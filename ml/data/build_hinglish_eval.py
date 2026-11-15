"""Hinglish / code-mixed evaluation sets (EVAL ONLY, never trained on). Writes ml/data/processed/:

  hinglish_human_benign.parquet   104 HUMAN-WRITTEN code-switched Hindi-English assistant queries (label 0), domain-stratified sample of
                                  the human-annotated test split of Hinglish-TOP (Google Research, Apache-2.0,
                                  https://github.com/google-research-datasets/Hinglish-TOP-Dataset, HF mirror WillHeld/hinglish_top).
                                  Human-annotated = crowdsourced code-switched renderings of English TOPv2 queries; the English originals
                                  (en_query) are kept as a paired control. NOT the 170K synthetic CST5 data.
  hinglish_attacks_ai.parquet     24 AI-DRAFTED Hinglish attacks (label 1) from ml/data/hinglish_attacks_ai.csv (Gemini; no public
                                  human-written Hinglish attack set exists)
  manual7.parquet                 the owner's 7 hand-tested live prompts (ml/data/manual7.csv), exact text
  prompt_talk_benign_ai.parquet   100 AI-DRAFTED benign prompts (50 English, 50 Hinglish) that mention system prompts / instructions
                                  (Gemini; 5 categories x 10 per language; read row by row, 0 removed), for the phrase-specific false-alarm rate
Leakage checks (asserted): no exact or char-n-gram cosine > 0.9 match against train, train_augmented, val, test, xstest, indirect_eval,
long_eval or the other sets built here; the human and AI sets also never overlap tricky.csv. manual7 may (2 of its 7 prompts ARE tricky.csv rows):
that is flagged in the column in_tricky_set.
These are short voice-assistant queries (median 8 words): they measure general false alarms on code-mixed text, NOT prompts about
system prompts. Run from the repo root: python ml/data/build_hinglish_eval.py (needs the Hugging Face datasets cache / network).
"""
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).parent
PROC = DATA / "processed"
SEED, PER_DOMAIN = 42, 13
norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()


def human_benign() -> pd.DataFrame:
    p = hf_hub_download("WillHeld/hinglish_top", "data/test-00000-of-00001-cf691a6b0d345ef6.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    df = df[df.cs_query.str.split().str.len() >= 3].drop_duplicates("cs_query").reset_index(drop=True)
    parts = [g.sample(n=min(PER_DOMAIN, len(g)), random_state=SEED) for _, g in df.groupby("domain")]
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return pd.DataFrame({"text": s.cs_query, "label": 0, "source": "hinglish_top_human", "domain": s.domain, "en_control": s.en_query,
                         "origin": "human-written (crowdsourced code-switching of TOPv2 queries), Apache-2.0"})


def main():
    PROC.mkdir(exist_ok=True)
    hb = human_benign()
    at = pd.read_csv(DATA / "hinglish_attacks_ai.csv")
    m7 = pd.read_csv(DATA / "manual7.csv")
    ref = pd.concat([pd.read_parquet(PROC / f"{n}.parquet")[["text"]] for n in
                     ["train", "train_augmented", "val", "test", "xstest_safe", "indirect_eval", "long_eval"]])
    tricky_n = set(pd.read_csv(DATA / "tricky.csv").text.map(norm))
    # two of the owner's manual prompts are identical to rows of the hand-written tricky set: recorded, not hidden
    m7["in_tricky_set"] = m7.text.map(norm).isin(tricky_n)
    pt = pd.read_csv(DATA / "prompt_talk_benign_ai.csv")
    sets = {"hinglish_human_benign": hb, "hinglish_attacks_ai": at, "manual7": m7, "prompt_talk_benign_ai": pt}
    refn = set(ref.text.map(norm))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(ref.text.tolist() + sum((d.text.tolist() for d in sets.values()), []))
    R = vec.transform(ref.text)
    for name, d in sets.items():
        assert d.text.map(norm).is_unique, f"{name}: duplicate texts"
        assert not set(d.text.map(norm)) & refn, f"{name}: exact overlap with training/val/test data"
        if name != "manual7":
            assert not set(d.text.map(norm)) & tricky_n, f"{name}: overlap with the tricky set"
        sim = (vec.transform(d.text) @ R.T).max(axis=1).toarray().ravel()
        assert (sim <= 0.9).all(), f"{name}: {int((sim > 0.9).sum())} rows with char-ngram cosine > 0.9 to existing data"
        print(f"{name}: {len(d)} rows, labels {d.label.value_counts().to_dict()}, max similarity to training/eval text {sim.max():.3f}")
        d.to_parquet(PROC / f"{name}.parquet", index=False)
    names = {n: set(d.text.map(norm)) for n, d in sets.items()}
    cross = {t for a in names for b in names if a < b for t in names[a] & names[b]}
    assert not cross, "overlap between the new sets"
    print("domains:", hb.domain.value_counts().to_dict())


if __name__ == "__main__":
    sys.exit(main())
