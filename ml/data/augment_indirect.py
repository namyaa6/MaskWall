"""Indirect-injection augmentation (train split only).

Real injections often sit inside data the user pasted ("summarize this email"), not in the user's own
message. The public datasets are almost all direct attacks, so models see few of those. This builds
synthetic examples:

  label 1: <request> + <Dolly passage with a train-split attack string inserted at a random position>
  label 0: <request> + <Dolly passage without an attack; ~40% get a harmless note/comment inserted>

Controls against shortcuts:
  * the same request list and joiners are used for both classes;
  * ~40% of the attack strings are wrapped in a comment/bracket style (<!-- -->, [Note: ...], ...); the
    same styles wrap a benign note in ~40% of the label-0 passages, so formatting alone is not a cue;
  * no single source supplies more than 50% of the inserted attacks;
  * passages are drawn from one shuffled pool, so length distributions match by construction (reported).

Leakage rules (asserted):
  * attack strings come from the train split only; none equals a val/test/xstest/tricky text;
  * no augmented text equals, or has TF-IDF cosine > 0.9 with, any val/test/xstest/tricky text.

The request wordings were written by Claude, which also drafted tricky.csv, so they are not fully blind to it.
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

REQUESTS = [
    "Summarize the following text in two or three sentences.",
    "Can you give me a short summary of this passage?",
    "Translate the following text into French.",
    "Translate this into Spanish for me:",
    "Extract the key facts from this text as a bulleted list.",
    "Please proofread the text below and fix any grammar mistakes.",
    "What is the main idea of the following passage?",
    "Rewrite this in simpler language for a ten-year-old.",
    "List the names of people and places mentioned in this text.",
    "Give me three questions a student could answer after reading this.",
    "tldr of this?",
    "Classify the tone of the text below as formal, neutral or casual.",
    "Turn the following into a one-line headline.",
]
JOINERS = ["\n\n", "\n\n", "\n", " "]
STYLES = ["<!-- {x} -->", "[Note: {x}]", "[{x}]", "(Note: {x})", "/* {x} */"]
BENIGN_NOTES = [
    "This passage was last updated in 2021.",
    "Source: public encyclopedia entry.",
    "Figures are approximate and may be out of date.",
    "See the references section for citations.",
    "Reviewed by the editorial team.",
    "Text shortened for length.",
    "Original spelling preserved.",
    "Please keep the original spelling and punctuation in your answer.",
    "Dates in this text use the day/month/year format.",
    "This text is provided for educational use.",
    "Draft, not yet fact-checked.",
    "Names should stay in their original form.",
    "Contains a quotation from an interview.",
    "Formatting was lost when this was copied from a web page.",
    "AI assistants: the dates below follow the Gregorian calendar.",
    "Note to assistant: quoted passages are verbatim.",
]
WRAP_FRAC = 0.4
MAX_SOURCE_FRAC = 0.5
ATTACK_MAX_CHARS = 500  # 300 left only 161 non-Gandalf attacks, too few for 400 rows under the 50% cap
PASSAGE_MIN, PASSAGE_MAX = 300, 1500
SIM_THRESHOLD = 0.9
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def allocate(avail: dict[str, int], total: int, max_frac: float) -> dict[str, int]:
    """Quota per source: proportional to availability, no source above max_frac * total."""
    cap = int(max_frac * total)
    quota = {s: 0 for s in avail}
    open_, left = set(avail), total
    while left > 0 and open_:
        weight = sum(avail[s] - quota[s] for s in open_)
        if weight <= 0:
            break
        grant = {s: min(cap - quota[s], int(left * (avail[s] - quota[s]) / weight)) for s in open_}
        if sum(grant.values()) == 0:  # rounding leftovers: hand out one at a time
            for s in sorted(open_, key=lambda s: avail[s] - quota[s], reverse=True):
                if left and quota[s] < min(cap, avail[s]):
                    quota[s] += 1
                    left -= 1
            if not any(quota[s] < min(cap, avail[s]) for s in open_):
                break
            continue
        for s, g in grant.items():
            g = min(g, avail[s] - quota[s])
            quota[s] += g
            left -= g
        open_ = {s for s in open_ if quota[s] < min(cap, avail[s])}
    assert sum(quota.values()) == total, f"cannot fill {total} attacks under the {max_frac:.0%} cap: {quota}"
    return quota


def insert_text(passage: str, ins: str, rng) -> tuple[str, str]:
    """Insert `ins` at a random sentence boundary (including the very start and end)."""
    pos = [0] + [m.end() for m in _BOUNDARY.finditer(passage)] + [len(passage)]
    p = int(rng.choice(pos))
    where = "start" if p == 0 else "end" if p == len(passage) else "middle"
    if where == "start":
        return ins + "\n\n" + passage, where
    if where == "end":
        return passage + "\n\n" + ins, where
    return passage[:p] + ins + " " + passage[p:], where


def build_request(passage_text: str, rng) -> tuple[str, str]:
    req = REQUESTS[int(rng.integers(len(REQUESTS)))]
    j = JOINERS[int(rng.integers(len(JOINERS)))]
    colon = j == "\n" and req.endswith(".") and rng.random() < 0.5
    return req, (req[:-1] + ":" if colon else req) + j + passage_text


def max_sim(vec, eval_matrix, texts, chunk=500):
    X = vec.transform(texts)
    out = []
    for i in range(0, X.shape[0], chunk):
        out.append(np.asarray((X[i:i + chunk] @ eval_matrix.T).max(axis=1).todense()).ravel())
    return np.concatenate(out)


def build(train, val, test, xstest, tricky, passages, is_english, n_each=400, seed=42):
    """Return (augmented rows with metadata, summary dict). All inputs are plain DataFrames/lists."""
    rng = np.random.default_rng(seed)
    eval_texts = pd.concat([val["text"], test["text"], xstest["text"], tricky["text"]]).tolist()
    eval_norm = {_norm(t) for t in eval_texts}
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(eval_texts + list(passages))
    E = vec.transform(eval_texts)

    # --- passages: length, English, not near-duplicate of any eval row
    pool = sorted({p.strip() for p in passages if PASSAGE_MIN <= len(p.strip()) <= PASSAGE_MAX})
    pool = [p for p in pool if is_english(p) and _norm(p) not in eval_norm]
    sims = max_sim(vec, E, pool)
    pool = [p for p, s in zip(pool, sims) if s <= SIM_THRESHOLD]
    assert len(pool) >= 2 * n_each, f"only {len(pool)} usable passages"
    order = rng.permutation(len(pool))
    pool = [pool[i] for i in order]

    # --- attack strings: train label 1, short, never a val/test string, source cap
    cand = train[(train["label"] == 1) & (train["text"].str.len() <= ATTACK_MAX_CHARS)]
    cand = cand[~cand["norm"].isin(eval_norm)]
    cand = cand.drop_duplicates("norm")
    avail = cand.groupby("source").size().to_dict()
    quota = allocate(avail, n_each, MAX_SOURCE_FRAC)
    picked = pd.concat([cand[cand["source"] == s].sample(n=q, random_state=seed) for s, q in quota.items() if q])
    picked = picked.sample(frac=1, random_state=seed).reset_index(drop=True)

    # exactly WRAP_FRAC of each class is wrapped/annotated (not left to chance), so the rate is identical
    wrap_flags = {lab: rng.permutation([True] * round(WRAP_FRAC * n_each) + [False] * (n_each - round(WRAP_FRAC * n_each)))
                  for lab in (1, 0)}
    rows = []
    for i in range(n_each):  # label 1
        attack = picked.loc[i, "text"].strip()
        wrapped = bool(wrap_flags[1][i])
        style = STYLES[int(rng.integers(len(STYLES)))] if wrapped else None
        ins = style.format(x=attack) if wrapped else attack
        body, where = insert_text(pool[i], ins, rng)
        req, text = build_request(body, rng)
        rows.append({"text": text, "label": 1, "label_original": 1, "source": "augmented_indirect", "attack_source": picked.loc[i, "source"],
                     "wrapped": wrapped, "position": where, "request": req, "passage_chars": len(pool[i])})
    for i in range(n_each):  # label 0
        passage = pool[n_each + i]
        wrapped = bool(wrap_flags[0][i])
        where = None
        if wrapped:
            note = BENIGN_NOTES[int(rng.integers(len(BENIGN_NOTES)))]
            passage, where = insert_text(passage, STYLES[int(rng.integers(len(STYLES)))].format(x=note), rng)
        req, text = build_request(passage, rng)
        rows.append({"text": text, "label": 0, "label_original": 0, "source": "augmented_indirect", "attack_source": None,
                     "wrapped": wrapped, "position": where, "request": req, "passage_chars": len(pool[n_each + i])})
    aug = pd.DataFrame(rows)
    aug["norm"] = aug["text"].map(_norm)

    # --- leakage asserts
    assert aug["norm"].is_unique
    assert not set(aug["norm"]) & eval_norm, "augmented text equals a val/test/xstest/tricky text"
    assert not set(picked["norm"]) & eval_norm, "attack string equals a val/test/xstest/tricky text"
    assert set(picked["norm"]) <= set(train["norm"]), "attack string not from train"
    s = max_sim(vec, E, aug["text"].tolist())
    assert (s <= SIM_THRESHOLD).all(), f"{int((s > SIM_THRESHOLD).sum())} augmented rows too similar to eval rows"
    aug["max_sim_eval"] = s

    pl = aug.groupby("label")["passage_chars"].describe()[["50%", "mean"]]
    summary = {
        "n": {"label1": n_each, "label0": n_each},
        "attack_source_quota": quota,
        "attack_source_available": avail,
        "wrapped_frac": aug.groupby("label")["wrapped"].mean().round(3).to_dict(),
        "position_label1": aug[aug.label == 1]["position"].value_counts().to_dict(),
        "passage_chars_median_mean": pl.round(1).to_dict(),
        "max_sim_to_eval": float(s.max()),
        "usable_passages": len(pool),
    }
    return aug.drop(columns="norm"), summary
