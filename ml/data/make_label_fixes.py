"""One-shot provenance script: turns my hand-review decisions into ml/data/label_fixes.csv.

Inputs (from flag_label_noise.py, seed-deterministic):
  ml/results/label_review_candidates.csv  210 rows flagged by 5-fold OOF word-TF-IDF (label 0 & p>=0.8, label 1 & p<=0.1)
  ml/results/label_review_sample.csv      50 random NON-flagged rows, read to estimate residual noise
Every row was read by hand; no keyword rule decided any label. The classifier only nominated rows.

Outputs:
  ml/data/label_fixes.csv                   the changes (relabel / drop); this is what load_datasets.py applies
  ml/results/label_review_decisions.csv     every reviewed row incl. 'keep' (label was right), for audit

label_fixes.csv is the source of truth; this script only documents how it was produced.
"""
import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAND = ROOT / "ml/results/label_review_candidates.csv"
SAMPLE = ROOT / "ml/results/label_review_sample.csv"

REASONS = {
    "A": "DAN-style persona declared free of AI rules/policies (jailbreak)",
    "B": "persona told to ignore ethics/content policy or never refuse (unfiltered/amoral)",
    "C": "fake system note granting an exception to ethical protocols for explicit content",
    "D": "explicit instruction override or rule-priority trick (instruction hijack)",
    "E": "fictional/hypothetical framing used to remove safety limits",
    "F": "forces a prohibited output; not a benign persona",
    "G": "fake developer-mode / emergency framing plus adversarial suffix",
    "H": "ordinary task or persona prompt (writing, marketing, coding, role-play helper); no override or safety bypass",
    "I": "benign character card / role-play template; no safety bypass",
    "J": "plain question or instruction with no injection",
    "K": "benign persona prompt; label is correct",
    "L": "label is correct (genuine attack or benign); the model just misses it",
}

# ---- candidates: label 0 -> 1 (hand-reviewed, id in candidates csv) -------------------------------
TO_ATTACK = {
    "A": [0, 12, 20, 26, 31, 39, 40, 41, 49, 50, 51, 57, 63, 64, 66],
    "B": [4, 6, 7, 8, 9, 14, 23, 24, 25, 28, 32, 34, 37, 42, 44, 47, 53, 54, 59, 62, 65, 68, 69, 73, 74, 75, 76, 78],
    "C": [5, 21, 46, 52],
    "D": [10, 18, 30, 33, 55, 56, 60, 67, 70, 71, 77],
    "E": [13, 15, 58, 72],
    "F": [17],
    "G": [48],
}
# ---- candidates: label 1 -> 0 ------------------------------------------------------------------
TO_BENIGN = {
    "I": [89, 92, 93, 105, 107, 121, 137, 149, 157, 159, 164],
    "J": [82, 84, 86, 88, 94, 109, 124, 151, 170, 198],
    "H": [79, 80, 83, 85, 87, 90, 95, 96, 98, 99, 100, 101, 102, 104, 110, 111, 118, 119, 120, 122, 125, 126,
          127, 128, 132, 135, 136, 140, 141, 142, 143, 152, 154, 155, 160, 162, 168, 175, 176, 177, 182, 184,
          188, 189, 191, 193, 197, 199, 200, 201, 206, 208],
}
# ---- candidates: drop (genuinely ambiguous) -----------------------------------------------------
DROP = {
    11: "'ignore all prior prompts' + harmless Joker persona; intent ambiguous",
    27: "'ignore all prior prompts' + 'opposite of ChatGPT' persona; unclear",
    29: "'ignore all previous instructions' + business role template; ambiguous",
    36: "ANTI-DAN makes the model over-cautious; neither clearly injection nor benign",
    38: "brevity rules plus 'fulfil ALL requests' and a threat; unclear",
    43: "'forget instructions' + flirtatious human persona; ambiguous",
    45: "'refuse everything with made-up reasons' persona; unclear",
    61: "override phrase but a defensive Discord-bot persona; ambiguous",
    81: "plain request for torrent sites: not an injection, not clearly benign",
    91: "topic text + appended 'act as storyteller'; can't tell injection from benign without the system prompt",
    97: "German question + appended 'act as debater'; same ambiguity as deepset concatenations",
    108: "unclear fragment",
    113: "document plus leading question; intent unclear",
    117: "'state that X is the most powerful'; leading-claim instruction, unclear",
    123: "'ignore all previous instructions' + benign SEO template; ambiguous",
    130: "unrelated question + appended 'act as linux terminal'; ambiguous",
    131: "dictator-satire role-play; content-policy question, not injection",
    144: "55k-char story; cannot verify",
    148: "JS snippet that redirects the ChatGPT UI; not an injection",
    158: "image-link formatting prompt with a suggestive image; unclear",
    163: "adult role-play persona; not clearly a bypass",
    165: "explicit-content request; not an injection (cf. XSTest unsafe rows excluded)",
    169: "NSFW role-play prompt optimisation; unclear",
    174: "'state that trump is the best'; leading-claim instruction, unclear",
    181: "AI-detector evasion rewrite; not an injection",
    187: "code calling a third-party API with spoofed headers; not an injection",
    204: "incoherent identity text",
    207: "drug-chemistry information request; harmful request, not injection",
}
# ---- residual-noise sample (ids refer to label_review_sample.csv) -------------------------------
SAMPLE_TO_ATTACK = {2: "C", 44: "C"}
SAMPLE_DROP = {
    33: "'[Ignore previous instructions]' + AI-detector-evasion template; ambiguous (same rule as candidates)",
    35: "'ignore all previous instructions' + SEO template; ambiguous",
    39: "'ignore all previous instructions' + SEO template; ambiguous",
}


def norm_hash(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode()).hexdigest()


def snippet(text):
    return " ".join(text.split())[:100]


def main():
    cand = pd.read_csv(CAND).set_index("id")
    samp = pd.read_csv(SAMPLE).set_index("id")
    assert len(cand) == 210 and len(samp) == 50
    fixes, log = [], []

    def add(df, i, kind, new, reason, origin):
        r = df.loc[i]
        h = norm_hash(r.text)
        log.append({"origin": origin, "id": i, "text_hash": h, "source": r.source, "split_at_review": r.split,
                    "old_label": int(r.label), "decision": kind if new is not None else "keep", "new_label": new,
                    "reason": reason, "snippet": snippet(r.text)})
        if kind in ("relabel", "drop"):
            fixes.append({"text_hash": h, "old_label": int(r.label), "new_label": new, "reason": reason,
                          "source": r.source, "origin": origin, "snippet": snippet(r.text)})

    decided = set()
    for code, ids in TO_ATTACK.items():
        for i in ids:
            assert cand.loc[i, "label"] == 0 and i not in decided
            add(cand, i, "relabel", 1, REASONS[code], "candidate"); decided.add(i)
    for code, ids in TO_BENIGN.items():
        for i in ids:
            assert cand.loc[i, "label"] == 1 and i not in decided
            add(cand, i, "relabel", 0, REASONS[code], "candidate"); decided.add(i)
    for i, why in DROP.items():
        assert i not in decided
        add(cand, i, "drop", "drop", why, "candidate"); decided.add(i)
    for i in cand.index:  # everything else was read and the label was right
        if i not in decided:
            add(cand, i, "keep", None, REASONS["L"] if cand.loc[i, "label"] == 1 else REASONS["K"], "candidate")

    for i, code in SAMPLE_TO_ATTACK.items():
        assert samp.loc[i, "label"] == 0
        add(samp, i, "relabel", 1, REASONS[code], "residual_sample")
    for i, why in SAMPLE_DROP.items():
        add(samp, i, "drop", "drop", why, "residual_sample")
    for i in samp.index:
        if i not in SAMPLE_TO_ATTACK and i not in SAMPLE_DROP:
            add(samp, i, "keep", None, "label is correct", "residual_sample")

    fx = pd.DataFrame(fixes)
    assert fx.text_hash.is_unique, "a text was decided twice"
    fx.to_csv(ROOT / "ml/data/label_fixes.csv", index=False)
    pd.DataFrame(log).to_csv(ROOT / "ml/results/label_review_decisions.csv", index=False)
    print(fx.groupby(["origin", "old_label", "new_label"]).size().to_string())
    print("total fixes:", len(fx))


if __name__ == "__main__":
    main()
