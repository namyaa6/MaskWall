"""Validate the hand-written tricky eval set (ml/data/tricky.csv).

Checks: parses as CSV, columns are text,label,category, labels are 0/1, categories
are from the known set, no exact or near-duplicates within the file, and no row
is a near-duplicate of the training set (TF-IDF fit on train, as in load_datasets).
Exits non-zero if any check fails. Rerun after adding rows.

Usage:
    python ml/data/validate_tricky.py [--threshold 0.9]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from load_datasets import max_similarity_to_train, normalize

DATA_DIR = Path(__file__).parent
CATEGORIES = {
    "benign_mentions_instructions": {0},
    "benign_document": {0},
    "benign_suspicious": {0},
    "long_benign": {0},
    "short_attack": {1},
    "sneaky_attack": {1},
    "hidden_injection_document": {1},
    "hinglish": {0, 1},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--file", type=Path, default=DATA_DIR / "tricky.csv")
    p.add_argument("--train", type=Path, default=DATA_DIR / "processed" / "train.parquet")
    args = p.parse_args()

    problems = []
    df = pd.read_csv(args.file, dtype={"text": str, "category": str})
    print(f"parsed {len(df)} rows from {args.file}")
    if list(df.columns) != ["text", "label", "category"]:
        sys.exit(f"FAIL: columns are {list(df.columns)}, expected text,label,category")
    if df.isna().any().any():
        problems.append(f"empty cells in rows {df[df.isna().any(axis=1)].index.tolist()}")
        df = df.dropna()

    # Labels and categories.
    bad_label = ~df["label"].isin([0, 1])
    if bad_label.any():
        problems.append(f"labels other than 0/1 in rows {df[bad_label].index.tolist()}")
    unknown = sorted(set(df["category"]) - set(CATEGORIES))
    if unknown:
        problems.append(f"unknown categories: {unknown}")
    for cat, allowed in CATEGORIES.items():
        wrong = df[(df["category"] == cat) & ~df["label"].isin(allowed)]
        if len(wrong):
            problems.append(f"{cat} rows {wrong.index.tolist()} have label outside {sorted(allowed)}")

    print("\n=== counts per category x label ===")
    counts = df.groupby(["category", "label"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts.loc["total"] = counts.sum()
    print(counts.to_string())

    # Duplicates within the file. Row numbers below are CSV line numbers (header = line 1).
    df["norm"] = df["text"].map(normalize)
    line = df.index + 2
    exact = df[df["norm"].duplicated(keep=False)]
    if len(exact):
        problems.append(f"exact duplicates within file at lines {(exact.index + 2).tolist()}")

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(df["norm"])
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0)
    i, j = np.triu_indices_from(sim, k=1)
    order = np.argsort(-sim[i, j])
    print("\n=== most similar pairs within file ===")
    for k in order[:5]:
        a, b = i[k], j[k]
        print(f"  {sim[a, b]:.2f}  line {line[a]} <-> line {line[b]}: {df['text'].iloc[a][:50]!r} | {df['text'].iloc[b][:50]!r}")
    near = [(line[a], line[b]) for a, b in zip(i, j) if sim[a, b] > args.threshold]
    if near:
        problems.append(f"near-duplicate pairs within file (> {args.threshold}): {near}")

    # Similarity to the training set.
    train = pd.read_parquet(args.train)
    train_norm = train["text"].map(normalize)
    tvec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X_train = tvec.fit_transform(train_norm)
    df["max_sim_train"] = max_similarity_to_train(tvec, X_train, df["norm"])
    print(f"\n=== similarity to train ({len(train)} rows) ===")
    print(f"exact overlap: {int(df['norm'].isin(set(train_norm)).sum())}")
    print(df.groupby("category")["max_sim_train"].max().round(2).to_string())
    top = df.nlargest(3, "max_sim_train")
    for idx, r in top.iterrows():
        print(f"  {r.max_sim_train:.2f}  line {idx + 2}: {r.text[:70]!r}")
    leaked = df[(df["max_sim_train"] > args.threshold) | df["norm"].isin(set(train_norm))]
    if len(leaked):
        problems.append(f"rows too similar to train at lines {(leaked.index + 2).tolist()}")

    print()
    if problems:
        for msg in problems:
            print("FAIL:", msg)
        sys.exit(1)
    print("OK: all checks passed")


if __name__ == "__main__":
    main()
