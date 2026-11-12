"""Flag possibly mislabeled rows using out-of-fold predictions (read-only; changes no data).

Every row in train+val+test is scored by a word-TF-IDF logistic regression that never saw it
(5-fold stratified, seed 42, C=10). xstest_safe is excluded: it is eval-only and has no label noise
concern of this kind. Rows whose label disagrees strongly with the OOF score are written to a
review file. The classifier only *nominates* rows; a human decides every label.

Usage: python ml/data/flag_label_noise.py [--hi 0.8] [--lo 0.1] [--out ml/results/label_review_candidates.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

DATA = Path(__file__).parent
SEED = 42


def oof_scores(df):
    # stratify on (source, label) like the real split, so every fold sees every source
    strata = df["source"] + "|" + df["label"].astype(str)
    oof = np.zeros(len(df))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, strata):
        m = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
            LogisticRegression(C=10, solver="liblinear", random_state=SEED, max_iter=2000),
        ).fit(df["text"].iloc[tr], df["label"].iloc[tr])
        oof[te] = m.predict_proba(df["text"].iloc[te])[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hi", type=float, default=0.8, help="flag label-0 rows with OOF p >= hi")
    ap.add_argument("--lo", type=float, default=0.1, help="flag label-1 rows with OOF p <= lo")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sample-out", type=Path, default=None, help="write a random sample of non-flagged rows")
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--sample-seed", type=int, default=303)
    args = ap.parse_args()

    parts = [pd.read_parquet(DATA / "processed" / f"{s}.parquet").assign(split=s) for s in ["train", "val", "test"]]
    df = pd.concat(parts, ignore_index=True)
    df["oof_p"] = oof_scores(df)

    print("rows:", len(df), "| label-0:", int((df.label == 0).sum()), "| label-1:", int((df.label == 1).sum()))
    print("\nflag counts by cutoff (label-0 with p >= x):")
    for x in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        print(f"  p>={x:<4} {int(((df.label == 0) & (df.oof_p >= x)).sum())}")
    print("flag counts by cutoff (label-1 with p <= x):")
    for x in [0.02, 0.05, 0.1, 0.2, 0.3]:
        print(f"  p<={x:<4} {int(((df.label == 1) & (df.oof_p <= x)).sum())}")

    flagged = df[((df.label == 0) & (df.oof_p >= args.hi)) | ((df.label == 1) & (df.oof_p <= args.lo))]
    print(f"\nflagged with hi={args.hi}, lo={args.lo}: {len(flagged)}")
    print(flagged.groupby(["label", "source", "split"]).size().to_string())
    if args.out:
        args.out.parent.mkdir(exist_ok=True, parents=True)
        flagged.sort_values(["label", "oof_p"]).to_csv(args.out, index=False)
        print("wrote", args.out)
    if args.sample_out:
        # random non-flagged rows, to estimate the label noise the model did NOT nominate
        rest = df.drop(flagged.index)
        rest.sample(args.sample, random_state=args.sample_seed).to_csv(args.sample_out, index=False)
        print("wrote", args.sample_out)


if __name__ == "__main__":
    main()
