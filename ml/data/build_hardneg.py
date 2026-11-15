"""Build the hard-negative TRAINING and VALIDATION data from the cached Groq generations (gen_hardneg_groq.py) and Hinglish-TOP.

Tiers (docs/decisions.md Day 2.6 proposal, approved by the owner 2026-11-15; Day 2.8):
  T1  benign prompt-talk, Groq (en + hinglish, 6 categories)
  T2  attacks, Groq (hinglish: english_keywords / hindi_only / roleplay / indirect; en: attacks that mention "system prompt")
  T3  HUMAN benign short commands: Hinglish-TOP TRAIN split (Apache-2.0) + their English originals. The 104 evaluation rows (E1) come from
      the TEST split; checked here for exact and near-duplicate overlap in both directions (cs and en columns). The new validation
      slice takes T3 rows from the Hinglish-TOP VALIDATION split (a third, disjoint split).
Filters (all counted in ml/results/hardneg_build_report.md): Devanagari, dialogue markers, wrong language, category rules (e.g. a "hindi_only"
attack must contain none of the English attack keywords), benign rows that contain a bare attack imperative (except the two categories that
legitimately quote attack phrases), exact duplicates, near-duplicates (char 3-5-gram TF-IDF cosine > 0.9) inside the data, against EVERY
evaluation set and against the existing train/val/test, and cross-class near-duplicates.
Writes ml/data/hardneg/hardneg_train.csv, hardneg_val.csv, ml/results/hardneg_build_report.md and the 50-row hand-review samples.
Run from the repo root: python ml/data/build_hardneg.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).parent
ROOT = DATA.parents[1]
PROC = DATA / "processed"
RAW = ROOT / "ml" / "cache" / "hardneg_raw"
OUT = DATA / "hardneg"
RES = ROOT / "ml" / "results"
SEED = 42
COS = 0.9
N_T3_TRAIN_PAIRS, N_T3_VAL_PAIRS = 200, 60
sys.path.insert(0, str(DATA))
from gen_hardneg_groq import TARGETS  # noqa: E402

norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
words = lambda t: len(t.split())
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
ROLE_MARK = re.compile(r"(?im)^\s*(user|assistant|chatgpt|bot|ai|system|human)\s*:")
# unambiguous romanised-Hindi function words (words that are also English, e.g. to/me/do/ho/par, are left out on purpose)
HINDI = set("hai hain kya kaise kaisa kyun kyon mujhe mera meri mere tum tumhare tumhari tumhe aap aapka aapki aapke karo karna kare kijiye "
            "nahi nahin bhi aur liye hoon hu tha thi bata batao batana dijiye chahiye chahta chahti chahte sakta sakte sakti yeh woh jao "
            "pehle saare sabhi sab kuch bhai yaar toh bhool apna apne apni unko isko uska uski kaun kitna kitni "
            "ka ke ki ko se ne pe ek hota hoga hogi raha rahi rahe karke karne karun karu likho likhna dekho dekhna suno bolo samjhao samjha dena dene lena lene "
            "zara bas abhi phir kyunki lekin magar isliye jaise kahan kab sirf bilkul wala wali wale hum hamare uske iske unka inka tera teri tere "
            "pichle pichli purane purana naya nayi hisaab dhyan rakhte hue maine mene tumne usne ab jo mein banao bana do de dena dikhao dikha "
            "sath saath bina wahi yahi isse usse kabhi hamesha zaroori matlab seedha seedhe jawab sawal gupt niyam nirdesh aadesh".split())
EN_ATTACK_KW = re.compile(r"\b(ignore|instructions?|system|prompts?|rules?|override|previous|developer|jailbreak|reveal|disregard|forget|dan)\b", re.I)
ATTACK_KW_ANY = re.compile(r"\b(ignore|instructions?|prompts?|rules?|override|jailbreak|developer|disregard|forget|restrictions?|guidelines?)\b", re.I)
SYSPROMPT = re.compile(r"system\s*(prompt|message|instruction)", re.I)
BARE_IMPERATIVE = re.compile(r"\b(ignore|disregard|forget|bhool\s+jao|nazarandaaz|ignore\s+karo)\b.{0,40}\b(previous|prior|above|earlier|all|saare|sabhi|pichle|pehle)\b.{0,25}"
                             r"\b(instructions?|rules?|prompts?|niyam|nirdesh)\b", re.I | re.S)
QUOTE = re.compile(r"[\"'“”‘’]")


def hindi_hits(t):
    return len(set(re.findall(r"[a-z]+", t.lower())) & HINDI)


def load_raw():
    rows = []
    for f in sorted(RAW.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for i, t in enumerate(d["items"]):
            rows.append(dict(text=t.strip(), label=int(d["tier"] == "T2"), tier=d["tier"], category=d["category"], language=d["language"],
                             bucket=d["bucket"], split=d["split"], call_id=d["id"], source=f"hardneg_groq_{d['tier'].lower()}", generator=d["model"]))
    return pd.DataFrame(rows)


def rule_filter(df, log):
    """Per-row validity rules. Returns the kept frame and records a reason for every dropped row."""
    keep, reasons = [], []
    for r in df.itertuples():
        t, why = r.text, None
        n = words(t)
        if not (4 <= n <= 110):
            why = "length"
        elif DEVANAGARI.search(t):
            why = "devanagari"
        elif ROLE_MARK.search(t):
            why = "dialogue_marker"
        elif r.language == "hinglish" and hindi_hits(t) < (3 if n > 30 else 2):
            why = "not_hinglish"
        elif r.language == "en" and hindi_hits(t) > 2:
            why = "not_english"
        elif r.tier == "T2" and r.category == "hindi_only" and EN_ATTACK_KW.search(t):
            why = "hindi_only_has_english_keyword"
        elif r.tier == "T2" and r.category == "english_keywords" and not ATTACK_KW_ANY.search(t):
            why = "keywords_missing"
        elif r.tier == "T2" and r.category == "system_prompt_attack" and not SYSPROMPT.search(t):
            why = "no_system_prompt_mention"
        elif r.tier == "T1" and r.category == "system_prompt_talk" and not re.search(r"system\s*prompt", t, re.I):
            why = "no_system_prompt_words"
        elif r.tier == "T1" and r.category == "quoted_phrase" and not QUOTE.search(t):
            why = "no_quote"
        elif r.tier == "T1" and r.category not in ("explain_concepts", "quoted_phrase") and BARE_IMPERATIVE.search(t):
            why = "benign_with_attack_imperative"
        keep.append(why is None)
        reasons.append(why)
    out = df.assign(drop_reason=reasons)
    log["rule_drops"] = out[~np.array(keep)].groupby(["tier", "category", "drop_reason"]).size().rename("n").reset_index()
    return out[keep].drop(columns="drop_reason")


def t3_rows():
    """Human Hinglish-TOP rows: (cs, en) pairs from TRAIN (training) and VALIDATION (the new val slice). Returns frames plus the E1 texts."""
    f = lambda name: pd.read_parquet(hf_hub_download("WillHeld/hinglish_top", name, repo_type="dataset"))
    tr = f("data/train-00000-of-00001-edd2764b201a5c1f.parquet")
    va = f("data/validation-00000-of-00001-5abcf62ed020ba5a.parquet")
    return tr, va


def pairs_to_rows(p, split):
    cs = pd.DataFrame(dict(text=p.cs_query.values, label=0, tier="T3", category="hinglish_top_cs", language="hinglish", bucket="short", split=split,
                           call_id="", source="hinglish_top_human", generator="human (Hinglish-TOP, Apache-2.0)", domain=p.domain.values))
    en = pd.DataFrame(dict(text=p.en_query.values, label=0, tier="T3", category="hinglish_top_en_original", language="en", bucket="short", split=split,
                           call_id="", source="hinglish_top_human", generator="human (TOPv2 English original, Apache-2.0)", domain=p.domain.values))
    return pd.concat([cs, en], ignore_index=True)


def main():
    OUT.mkdir(exist_ok=True)
    rep = {}
    raw = load_raw()
    assert len(raw), "no generations found: run ml/data/gen_hardneg_groq.py first"
    rep["raw_rows"] = {f"{a}/{b}": int(n) for (a, b), n in raw.groupby(["split", "tier"]).size().items()}
    gen = rule_filter(raw, rep)

    # ---- reference texts: every evaluation set and the existing train/val/test (nothing here may overlap them)
    ref_names = ["train", "train_augmented", "val", "test", "xstest_safe", "indirect_eval", "long_eval",
                 "hinglish_human_benign", "hinglish_attacks_ai", "prompt_talk_benign_ai", "manual7"]
    ref = pd.concat([pd.read_parquet(PROC / f"{n}.parquet")[["text"]].assign(set=n) for n in ref_names] +
                    [pd.read_csv(DATA / "tricky.csv")[["text"]].assign(set="tricky")], ignore_index=True)
    e1 = pd.read_parquet(PROC / "hinglish_human_benign.parquet")
    ref = pd.concat([ref, pd.DataFrame({"text": e1.en_control, "set": "E1_english_control"})], ignore_index=True)

    # ---- T3
    tr, va = t3_rows()
    tr = tr[tr.cs_query.str.split().str.len() >= 3].drop_duplicates("cs_query").drop_duplicates("en_query")
    va = va[va.cs_query.str.split().str.len() >= 3].drop_duplicates("cs_query").drop_duplicates("en_query")
    e1_norm = set(e1.text.map(norm)) | set(e1.en_control.map(norm))
    rep["t3_exact_overlap_with_E1"] = int(len(set(tr.cs_query.map(norm)) & e1_norm) + len(set(tr.en_query.map(norm)) & e1_norm))
    rep["t3_val_exact_overlap_with_train"] = int(len((set(va.cs_query.map(norm)) | set(va.en_query.map(norm))) & (set(tr.cs_query.map(norm)) | set(tr.en_query.map(norm)))))
    rng = np.random.default_rng(SEED)

    # one vectoriser for everything; fitted on all text so that out-of-vocabulary words count (the earlier filter fitted on train only and missed templates)
    allt = pd.concat([gen.text, ref.text, tr.cs_query, tr.en_query, va.cs_query, va.en_query]).tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(allt)
    R = vec.transform(ref.text)
    refn = set(ref.text.map(norm))

    def max_sim_to_ref(texts):
        X = vec.transform(texts)
        sims, who = [], []
        for i in range(0, X.shape[0], 400):                       # chunked dense product: one pass gives both the max and its argmax
            D = (X[i:i + 400] @ R.T).toarray()
            sims.append(D.max(axis=1)); who.append(D.argmax(axis=1))
        return np.concatenate(sims), ref["set"].to_numpy()[np.concatenate(who)]

    # T3 candidate pairs: drop pairs whose cs OR en text is a near-duplicate of any reference text (E1 especially)
    def clean_pairs(p, name):
        s1, w1 = max_sim_to_ref(p.cs_query.tolist())
        s2, w2 = max_sim_to_ref(p.en_query.tolist())
        bad = (s1 > COS) | (s2 > COS) | p.cs_query.map(norm).isin(refn).to_numpy() | p.en_query.map(norm).isin(refn).to_numpy()
        rep[f"t3_{name}_pairs_dropped_near_dup"] = int(bad.sum())
        rep[f"t3_{name}_dropped_by_set"] = pd.Series(np.where(s1[bad] >= s2[bad], w1[bad], w2[bad])).value_counts().to_dict()
        return p[~bad]
    tr_c, va_c = clean_pairs(tr, "train"), clean_pairs(va, "val")
    # train vs val pairs must not near-duplicate each other either
    Xv, Xt = vec.transform(va_c.cs_query), vec.transform(tr_c.cs_query)
    clash = np.concatenate([((Xt[i:i + 400] @ Xv.T).toarray().max(axis=1) > COS) for i in range(0, Xt.shape[0], 400)])   # per train row: too close to some val row
    rep["t3_train_pairs_dropped_near_dup_of_val"] = int(clash.sum())
    tr_c = tr_c[~clash]
    sample = lambda p, n: pd.concat([g.sample(n=min(len(g), int(np.ceil(n * len(g) / len(p)))), random_state=SEED) for _, g in p.groupby("domain")]).sample(n=n, random_state=SEED)
    t3_tr = pairs_to_rows(sample(tr_c, N_T3_TRAIN_PAIRS), "train")
    t3_va = pairs_to_rows(sample(va_c, N_T3_VAL_PAIRS), "val")
    s_tr, _ = max_sim_to_ref(t3_tr.text.tolist())
    e1_sim = (vec.transform(t3_tr.text) @ vec.transform(pd.concat([e1.text, e1.en_control])).T).max()
    rep["t3_train_max_cos_to_E1_cs_and_en"] = round(float(e1_sim), 3)
    rep["t3_train_max_cos_to_any_reference"] = round(float(s_tr.max()), 3)
    assert e1_sim <= COS and not set(t3_tr.text.map(norm)) & e1_norm and not set(t3_va.text.map(norm)) & e1_norm

    # ---- generated rows: leakage filter against references
    sims, who = max_sim_to_ref(gen.text.tolist())
    near = (sims > COS) | gen.text.map(norm).isin(refn).to_numpy()
    rep["gen_dropped_near_dup_of_reference"] = {"n": int(near.sum()), "by_set": pd.Series(who[near]).value_counts().to_dict()}
    gen = gen[~near].reset_index(drop=True)

    # ---- de-duplicate inside the whole new data (train + val + T3 together), keeping the first occurrence; cross-class pairs are dropped on both sides
    allx = pd.concat([gen, t3_tr, t3_va], ignore_index=True)
    allx["nt"] = allx.text.map(norm)
    X = vec.transform(allx.text)
    pairs = []
    for i in range(0, X.shape[0], 400):                           # chunked dense product, upper triangle only
        D = (X[i:i + 400] @ X.T).toarray()
        for r in range(D.shape[0]):
            D[r, : i + r + 1] = 0
        a, b = np.nonzero(D > COS)
        pairs += [(i + x, y) for x, y in zip(a, b)]
    ii = np.array([p[0] for p in pairs], dtype=int); jj = np.array([p[1] for p in pairs], dtype=int)
    drop = set()
    lab = allx.label.to_numpy()
    for a, b in zip(ii, jj):                                      # a < b always
        if lab[a] != lab[b]:
            drop |= {a, b}
        else:
            drop.add(b)
    rep["dedup_dropped"] = len(drop)
    rep["dedup_cross_class_pairs"] = int(sum(1 for a, b in zip(ii, jj) if lab[a] != lab[b]))
    allx = allx.drop(index=list(drop)).reset_index(drop=True)
    # exact duplicates that survive (identical normalised text): keep first
    allx = allx.drop_duplicates("nt").drop(columns="nt").reset_index(drop=True)

    # ---- hand-removed rows (none unless a review listed some)
    manual = DATA / "hardneg_manual_drops.txt"
    md = set(map(norm, manual.read_text(encoding="utf-8").splitlines())) if manual.exists() else set()
    rep["manual_drops"] = int(allx.text.map(norm).isin(md).sum())
    allx = allx[~allx.text.map(norm).isin(md)].reset_index(drop=True)

    # ---- select the final rows per cell, stratified by ACTUAL word count (the generator's requested length is only a request: benign "long" messages
    # came out much shorter than attack ones). One target mix of word bins per language, taken from the T1 benign pool and used for BOTH classes.
    allx["wbin"] = pd.cut(allx.text.map(words), [0, 14, 30, 10**6], labels=["short", "medium", "long"]).astype(str)
    mix = {lg: allx[(allx.tier == "T1") & (allx.language == lg) & (allx.split == "train")].wbin.value_counts(normalize=True).reindex(["short", "medium", "long"]).fillna(0) for lg in ("en", "hinglish")}
    rep["target_word_bin_mix"] = {lg: m.round(3).to_dict() for lg, m in mix.items()}
    parts = []
    short = {}
    for (tier, cat, lang), (n_tr, n_va) in TARGETS.items():
        for split, n in (("train", n_tr), ("val", n_va)):
            g = allx[(allx.tier == tier) & (allx.category == cat) & (allx.language == lang) & (allx.split == split)]
            want = {k: int(round(n * v)) for k, v in mix[lang].items()}
            want["short"] += n - sum(want.values())
            take = [g[g.wbin == k].sample(n=min((g.wbin == k).sum(), want[k]), random_state=SEED) for k in ("short", "medium", "long")]
            sel = pd.concat(take)
            if len(sel) < n:                                   # top up from any bin
                rest = g.drop(index=sel.index)
                sel = pd.concat([sel, rest.sample(n=min(len(rest), n - len(sel)), random_state=SEED)])
            short[f"{split}/{tier}/{cat}/{lang}"] = f"{len(sel)}/{n} (available {len(g)})"
            parts.append(sel)
    final = pd.concat(parts + [allx[allx.tier == "T3"]], ignore_index=True)
    rep["cells_selected_vs_wanted"] = short
    final["label_original"] = final["label"]
    final["words"] = final.text.map(words)
    cols = ["text", "label", "label_original", "tier", "category", "language", "bucket", "split", "source", "generator", "call_id", "words"]
    for split in ("train", "val"):
        d = final[final.split == split][cols].sample(frac=1, random_state=SEED).reset_index(drop=True)
        d.to_csv(OUT / f"hardneg_{split}.csv", index=False, encoding="utf-8")
        rep[f"final_{split}"] = d.groupby(["tier", "label"]).size().rename("n").reset_index().to_dict("records")

    # ---- integrity checks (asserted)
    tr_f, va_f = final[final.split == "train"], final[final.split == "val"]
    assert not set(tr_f.text.map(norm)) & set(va_f.text.map(norm)), "train/val overlap"
    assert not set(final.text.map(norm)) & refn, "overlap with existing data"
    fs, _ = max_sim_to_ref(final.text.tolist())
    assert (fs <= COS).all()
    rep["final_max_cos_to_any_reference"] = round(float(fs.max()), 3)

    # ---- diagnostics: style and length by class (a shortcut check, not a proof)
    d = tr_f[tr_f.tier.isin(["T1", "T2"])].copy()
    d["q"] = d.text.str.contains(r"\?")
    d["nl"] = d.text.str.contains("\n")
    d["sys"] = d.text.map(lambda t: bool(SYSPROMPT.search(t)))
    d["ignore"] = d.text.str.contains(r"\b(ignore|bhool|forget)\b", case=False)
    diag = d.groupby(["language", "label"]).agg(n=("text", "size"), words_p25=("words", lambda x: x.quantile(.25)), words_median=("words", "median"),
                                                 words_p75=("words", lambda x: x.quantile(.75)), question_mark=("q", "mean"), has_newline=("nl", "mean"),
                                                 mentions_system_prompt=("sys", "mean"), has_ignore_forget=("ignore", "mean")).round(2)
    bucket_mix = tr_f[tr_f.tier.isin(["T1", "T2"])].assign(wbin=lambda x: pd.cut(x.words, [0, 14, 30, 10**6], labels=["short", "medium", "long"]).astype(str)).groupby(["language", "label"]).wbin.value_counts(normalize=True).unstack().round(2)

    old = pd.read_parquet(PROC / "train_augmented.parquet")
    old_sys = {int(l): round(float(old[old.label == l].text.map(lambda t: bool(SYSPROMPT.search(t))).mean()), 4) for l in (0, 1)}

    L = ["# Hard-negative data build report (generated by ml/data/build_hardneg.py)\n",
         "Counts after every filter; thresholds: char 3-5-gram TF-IDF cosine > 0.9 or exact normalised match = duplicate.\n",
         "## Rows", "```", json.dumps({k: v for k, v in rep.items() if not isinstance(v, pd.DataFrame)}, indent=1, ensure_ascii=False, default=str), "```\n",
         "## Rows dropped by the validity rules (tier, category, reason)\n", rep["rule_drops"].to_markdown(index=False), "",
         "## Style and length by class, train, T1+T2 only (generator-style shortcut check)\n", diag.to_markdown(), "",
         "Word-count bin mix (short <= 14, medium 15 to 30, long > 30 words; share of rows):\n", bucket_mix.to_markdown(), "",
         f"For scale: in the existing train_augmented data, 'system prompt/message/instruction' appears in {old_sys[0]:.1%} of benign and {old_sys[1]:.1%} of attack rows.\n"]
    (RES / "hardneg_build_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:1]), json.dumps({k: v for k, v in rep.items() if k not in ('rule_drops',)}, indent=1, ensure_ascii=False, default=str))
    print(diag.to_string())

    # ---- 50-row hand-review samples (read by a human; rows judged wrong go into hardneg_manual_drops.txt and the build is rerun)
    samp = []
    for name, g in {"T1_benign_train": tr_f[tr_f.tier == "T1"], "T2_attack_train": tr_f[tr_f.tier == "T2"], "val_new_generated": va_f[va_f.tier.isin(["T1", "T2"])]}.items():
        samp.append(g.sample(n=min(50, len(g)), random_state=SEED).assign(review_set=name)[["review_set", "label", "tier", "category", "language", "text"]])
    pd.concat(samp).to_csv(RES / "hardneg_review_sample.csv", index=False, encoding="utf-8")
    tr_f[tr_f.category == "quoted_phrase"][["language", "text"]].to_csv(RES / "hardneg_quoted_phrase_rows.csv", index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
