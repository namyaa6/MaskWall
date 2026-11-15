"""Sensitivity of the headline test numbers to templated / near-duplicate rows (ml/ review, finding 4).

The loader's near-duplicate filter fits TF-IDF on train only, never compares val with test, and public datasets contain templated families
("please ignore all previous instructions." appears 108 times in train, 8 in val, 7 in test). Question: do the test numbers change if
those rows are excluded? Baseline only (word TF-IDF + LR, C and threshold tuned on val exactly as in train_baseline.py), cleaned labels.
Reporting only. Run: python ml/leakage_sensitivity.py -> ml/results/leakage_sensitivity.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score

import train_baseline as tb

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "ml" / "data" / "processed"
norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()


def main():
    train, val, test = (pd.read_parquet(P / f"{n}.parquet") for n in ("train", "val", "test"))
    # model exactly as the baseline script selects it
    best_C = max(tb.C_GRID, key=lambda C: average_precision_score(val.label, tb.build_model("word_tfidf", C).fit(train.text, train.label).predict_proba(val.text)[:, 1]))
    model = tb.build_model("word_tfidf", best_C).fit(train.text, train.label)
    thr, _ = tb.tune_f1_threshold(val.label.to_numpy(), model.predict_proba(val.text)[:, 1])
    test_scores = model.predict_proba(test.text)[:, 1]

    tr_prefix = {norm(t)[:40] for t in train.text}
    prefix_shared = test.text.map(lambda t: norm(t)[:40] in tr_prefix).to_numpy()
    # similarity with the vocabulary of ALL splits (the loader fit on train only, so out-of-vocabulary words were ignored)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(pd.concat([train.text, val.text, test.text]))
    Xt, Xv, Xtr = vec.transform(test.text), vec.transform(val.text), vec.transform(train.text)
    sim_val = np.asarray((Xt @ Xv.T).max(axis=1).todense()).ravel()
    sim_train = np.asarray((Xt @ Xtr.T).max(axis=1).todense()).ravel()
    near_val = sim_val >= 0.9

    y = test.label.to_numpy()
    pred = test_scores >= thr
    out = {"C": best_C, "threshold": thr, "n_test": int(len(test)),
           "prefix_shared_with_train": int(prefix_shared.sum()), "test_rows_with_cosine>=0.9_to_a_val_row": int(near_val.sum()),
           "test_rows_with_full_vocab_cosine>=0.9_to_train": int((sim_train >= 0.9).sum()), "subsets": {}}
    for name, drop in {"all test rows": np.zeros(len(test), bool), "excluding rows sharing a 40-char prefix with train": prefix_shared,
                       "excluding rows with cosine >= 0.9 to a val row": near_val, "excluding both": prefix_shared | near_val}.items():
        keep = ~drop
        r = tb.rates(y[keep], pred[keep])
        out["subsets"][name] = {"n": int(keep.sum()), "attacks": int(y[keep].sum()), "f1": round(r["f1"], 4), "fpr": round(r["fpr"], 4), "recall": round(r["recall"], 4)}
    by_source = {}
    for s, g in test.assign(pred=pred, shared=prefix_shared).groupby("source"):
        by_source[s] = {"n": int(len(g)), "prefix_shared": int(g.shared.sum())}
    out["prefix_shared_by_source"] = by_source
    (ROOT / "ml" / "results" / "leakage_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "prefix_shared_by_source"}, indent=2))
    print(by_source)


if __name__ == "__main__":
    main()
