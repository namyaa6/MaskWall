"""Build the prompt-injection / benign dataset for MaskWall's firewall classifier.

Labels: 0 = benign, 1 = injection or jailbreak (merged; the firewall blocks both).
Every row keeps a `source` column so metrics can be reported per source.

Pipeline: load -> clean -> apply hand-reviewed label fixes -> exact dedup -> sample benign
          -> remove rows dropped as ambiguous -> stratified split
          -> English-only filter -> drop near-duplicates of train from val/test
          -> save + report.

Label fixes (ml/data/label_fixes.csv, keyed by SHA-256 of the normalized text) come from a
hand review of rows an out-of-fold classifier flagged as possibly mislabeled; see
docs/decisions.md. Output has `label` (cleaned) and `label_original` (before fixes), so any
metric can be reported on either. Benign sampling uses `label_original`, so the sampled rows
don't change when labels do. Pass --no-label-fixes to reproduce the unfixed dataset.

See docs/decisions.md for why each dataset was included or excluded.

Usage:
    python ml/data/load_datasets.py [--benign-ratio 3] [--seed 42]
                                    [--near-dup-threshold 0.9] [--out-dir ml/data/processed]
                                    [--label-fixes ml/data/label_fixes.csv | --no-label-fixes]
                                    [--augment-indirect [--augment-n 400]]

--augment-indirect additionally writes train_augmented.parquet (train + synthetic indirect-injection
rows, see ml/data/augment_indirect.py) and augmented_indirect.parquet (the synthetic rows with metadata).
val, test, xstest and train.parquet are unchanged by the flag. It also writes indirect_eval.parquet, a held-out
evaluation-only set built from val/test attack strings and fresh passages/wordings (ml/data/indirect_eval.py).
"""

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DetectorFactory.seed = 0  # langdetect is randomized; make it deterministic

MIN_CHARS = 5
TRUSTAIRLAB = "TrustAIRLab/in-the-wild-jailbreak-prompts"

# langdetect is unreliable on short text ("Ignore all previous guidelines" -> it).
# Below SHORT_WORDS we rescue a "non-English" verdict if the text has no non-ASCII
# letters and enough common English words. Words shared with German ("was", "in"
# counts once, "die" is absent) are kept out of the list; attack words like
# "ignore" are deliberately absent too, so the rule stays label-agnostic.
SHORT_WORDS = 20
MIN_FUNCTION_WORD_RATIO = 0.2
EN_FUNCTION_WORDS = frozenset(
    "the a an is are were be been do does did i you he she it we they me my your our "
    "their this that these those what which who where when why how and or but if of to in "
    "on at for with from by about as all any not no can could will would should have has had "
    "there here tell give please now ever everything just said make write list explain describe".split()
)


# ---------------------------------------------------------------- loaders
# Each returns a DataFrame with columns: text, label, source.


def _all_splits(name: str, config: str | None = None) -> pd.DataFrame:
    """Concatenate every split of a HF dataset (we re-split ourselves)."""
    ds = load_dataset(name, config)
    return pd.concat([ds[s].to_pandas() for s in ds], ignore_index=True)


def load_deepset() -> pd.DataFrame:
    df = _all_splits("deepset/prompt-injections")
    return pd.DataFrame({"text": df["text"], "label": df["label"].astype(int), "source": "deepset"})


def load_jackhhao() -> pd.DataFrame:
    df = _all_splits("jackhhao/jailbreak-classification")
    label = df["type"].map({"benign": 0, "jailbreak": 1})
    assert label.notna().all(), f"unexpected types: {df['type'].unique()}"
    return pd.DataFrame({"text": df["prompt"], "label": label.astype(int), "source": "jackhhao"})


def load_gandalf() -> pd.DataFrame:
    df = _all_splits("Lakera/gandalf_ignore_instructions")
    return pd.DataFrame({"text": df["text"], "label": 1, "source": "gandalf"})


def load_trustairlab_jailbreak() -> pd.DataFrame:
    df = _all_splits(TRUSTAIRLAB, "jailbreak_2023_12_25")
    return pd.DataFrame({"text": df["prompt"], "label": 1, "source": "trustairlab_jailbreak"})


def load_trustairlab_regular() -> pd.DataFrame:
    df = _all_splits(TRUSTAIRLAB, "regular_2023_12_25")
    return pd.DataFrame({"text": df["prompt"], "label": 0, "source": "trustairlab_regular"})


def load_dolly() -> pd.DataFrame:
    df = _all_splits("databricks/databricks-dolly-15k")
    return pd.DataFrame({"text": df["instruction"], "label": 0, "source": "dolly"})


def load_dolly_contexts() -> list[str]:
    """Dolly context passages (Wikipedia-style text). Only used to build augmented_indirect rows."""
    df = _all_splits("databricks/databricks-dolly-15k")
    return [c for c in df["context"].dropna().astype(str) if c.strip()]


def load_xstest_safe() -> pd.DataFrame:
    """Eval-only: safe prompts that look dangerous. Never used for training."""
    df = _all_splits("Paul/XSTest")
    df = df[df["label"] == "safe"]
    return pd.DataFrame({"text": df["prompt"], "label": 0, "source": "xstest_safe"})


# ---------------------------------------------------------------- processing


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= MIN_CHARS]
    df["norm"] = df["text"].map(normalize)
    return df.reset_index(drop=True)


def apply_label_fixes(df: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    """Add `label_original`, apply relabels, and flag rows to drop (removed after benign sampling)."""
    df = df.assign(label_original=df["label"], drop_row=False)
    if path is None:
        return df
    fixes = pd.read_csv(path, dtype={"new_label": str})
    df["hash"] = df["norm"].map(lambda t: hashlib.sha256(t.encode()).hexdigest())
    missing = set(fixes["text_hash"]) - set(df["hash"])
    assert not missing, f"{len(missing)} label fixes match no row (text changed or source removed)"
    for _, f in fixes.iterrows():
        rows = df["hash"] == f["text_hash"]
        assert (df.loc[rows, "label_original"] == f["old_label"]).all(), "fix old_label disagrees with data"
        if f["new_label"] == "drop":
            df.loc[rows, "drop_row"] = True
        else:
            df.loc[rows, "label"] = int(f["new_label"])
    kinds = fixes["new_label"].map(lambda v: "drop" if v == "drop" else f"relabel->{v}").value_counts()
    print(f"label fixes applied from {path.name}: {len(fixes)} texts ({kinds.to_dict()}), "
          f"{int((df['label'] != df['label_original']).sum())} rows relabeled, {int(df['drop_row'].sum())} rows flagged to drop")
    return df.drop(columns="hash")


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Exact dedup on normalized text; drop texts seen with conflicting labels."""
    n_labels = df.groupby("norm")["label"].nunique()
    conflicting = n_labels[n_labels > 1].index
    print(f"dedup: dropping {len(conflicting)} texts with conflicting labels")
    df = df[~df["norm"].isin(conflicting)]
    before = len(df)
    df = df.drop_duplicates(subset="norm", keep="first")
    print(f"dedup: dropped {before - len(df)} exact duplicates")
    return df.reset_index(drop=True)


def is_english(text: str) -> bool:
    try:
        if detect(text) == "en":
            return True
    except LangDetectException:  # no detectable features, e.g. Morse code or symbols
        return False
    words = re.findall(r"[^\W\d_]+", text.lower())
    if not words or len(words) >= SHORT_WORDS:
        return False
    if any(not c.isascii() for w in words for c in w):  # ä, é, Cyrillic, CJK, ...
        return False
    function_ratio = sum(w in EN_FUNCTION_WORDS for w in words) / len(words)
    return function_ratio >= MIN_FUNCTION_WORD_RATIO


def filter_english(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Drop non-English rows from every split; report drops per split x source."""
    result, dropped = {}, {}
    for name, df in dfs.items():
        en = df["text"].map(is_english)
        dropped[name] = df[~en].groupby(["source", "label"]).size()
        result[name] = df[en].reset_index(drop=True)
    print("\n=== non-English rows dropped ===")
    report = pd.concat(dropped, axis=1).fillna(0).astype(int)
    report.loc[("total", ""), :] = report.sum()
    print(report.to_string())
    return result


def sample_benign(df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    """Keep all attacks and native benign rows; top up from the big benign pools
    (TrustAIRLab-regular, Dolly) evenly until benign ~= ratio * attacks."""
    pools = ["trustairlab_regular", "dolly"]
    # Sample on the ORIGINAL labels so label fixes don't change which rows are drawn.
    lab = df["label_original"] if "label_original" in df else df["label"]
    fixed = df[~df["source"].isin(pools)]
    n_attacks = int((lab == 1).sum())
    budget = int(ratio * n_attacks) - int((lab[fixed.index] == 0).sum())
    per_pool = max(budget, 0) // len(pools)
    sampled = [
        df[df["source"] == p].sample(n=min(per_pool, int((df["source"] == p).sum())), random_state=seed)
        for p in pools
    ]
    return pd.concat([fixed, *sampled], ignore_index=True)


def split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 80/10/10 on (source, label)."""
    strata = df["source"] + "_" + df["label"].astype(str)
    train, rest = train_test_split(df, test_size=0.2, stratify=strata, random_state=seed)
    rest_strata = rest["source"] + "_" + rest["label"].astype(str)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest_strata, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def max_similarity_to_train(vec: TfidfVectorizer, X_train, texts: pd.Series, chunk: int = 1000) -> np.ndarray:
    """Max cosine similarity of each text to any train row (TF-IDF rows are L2-normalized)."""
    out = []
    for i in range(0, len(texts), chunk):
        X = vec.transform(texts.iloc[i : i + chunk])
        out.append((X @ X_train.T).max(axis=1).toarray().ravel())
    return np.concatenate(out) if out else np.array([])


def drop_near_duplicates(train, evals: dict[str, pd.DataFrame], threshold: float, report_only: set[str]):
    """Fit TF-IDF on train only; drop eval rows whose max similarity to train > threshold."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X_train = vec.fit_transform(train["norm"])
    result = {}
    print(f"\n=== near-duplicates vs train (cosine > {threshold}) ===")
    for name, df in evals.items():
        sim = max_similarity_to_train(vec, X_train, df["norm"])
        df = df.assign(max_sim_train=sim)
        near = df["max_sim_train"] > threshold
        counts = df[near].groupby(["source", "label"]).size()
        action = "reported only" if name in report_only else "dropped"
        print(f"{name}: {int(near.sum())} / {len(df)} {action}")
        if len(counts):
            print(counts.to_string())
        result[name] = df if name in report_only else df[~near].reset_index(drop=True)
    return result


def length_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(chars=df["text"].str.len(), words=df["text"].str.split().str.len())
    g = d.groupby(["label", "source"])
    return pd.DataFrame(
        {
            "n": g.size(),
            "chars_med": g["chars"].median(),
            "chars_p90": g["chars"].quantile(0.9),
            "words_med": g["words"].median(),
            "words_p90": g["words"].quantile(0.9),
        }
    ).round(0).astype(int)


# ---------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benign-ratio", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--near-dup-threshold", type=float, default=0.9)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "processed")
    p.add_argument("--label-fixes", type=Path, default=Path(__file__).parent / "label_fixes.csv")
    p.add_argument("--augment-indirect", action="store_true",
                   help="also write train_augmented.parquet: train + synthetic indirect-injection rows (train only)")
    p.add_argument("--augment-n", type=int, default=400, help="rows per class for --augment-indirect")
    p.add_argument("--no-label-fixes", action="store_true", help="skip hand-reviewed label fixes")
    args = p.parse_args()

    loaders = [
        load_deepset,
        load_jackhhao,
        load_gandalf,
        load_trustairlab_jailbreak,
        load_trustairlab_regular,
        load_dolly,
    ]
    raw = pd.concat([f() for f in loaders], ignore_index=True)
    print(f"loaded {len(raw)} rows from {raw['source'].nunique()} sources")

    df = dedup(apply_label_fixes(clean(raw), None if args.no_label_fixes else args.label_fixes))
    df = sample_benign(df, args.benign_ratio, args.seed)
    n_dropped = int(df["drop_row"].sum())
    df = df[~df["drop_row"]].reset_index(drop=True)
    print(f"removed {n_dropped} sampled rows dropped as ambiguous by label review")
    train, val, test = split(df, args.seed)

    # XSTest: drop anything that already appears in the training pool, then report near-dups.
    xstest = clean(load_xstest_safe())
    xstest["label_original"] = xstest["label"]  # no label fixes for the eval-only safe set
    xstest = xstest[~xstest["norm"].isin(df["norm"])].reset_index(drop=True)

    # English only: DistilBERT/MiniLM are English models (see docs/decisions.md).
    english = filter_english({"train": train, "val": val, "test": test, "xstest_safe": xstest})
    train, val, test, xstest = english["train"], english["val"], english["test"], english["xstest_safe"]

    evals = drop_near_duplicates(
        train, {"val": val, "test": test, "xstest_safe": xstest}, args.near_dup_threshold, report_only={"xstest_safe"}
    )
    val, test, xstest = evals["val"], evals["test"], evals["xstest_safe"]

    # Leakage asserts.
    splits = {"train": set(train["norm"]), "val": set(val["norm"]), "test": set(test["norm"])}
    assert not splits["train"] & splits["val"], "exact overlap train/val"
    assert not splits["train"] & splits["test"], "exact overlap train/test"
    assert not splits["val"] & splits["test"], "exact overlap val/test"
    assert (val["max_sim_train"] <= args.near_dup_threshold).all()
    assert (test["max_sim_train"] <= args.near_dup_threshold).all()

    # Summary.
    print("\n=== counts per split x source x label ===")
    counts = pd.concat(
        {name: d.groupby(["source", "label"]).size() for name, d in [("train", train), ("val", val), ("test", test)]},
        axis=1,
    ).fillna(0).astype(int)
    print(counts.to_string())
    for name, d in [("train", train), ("val", val), ("test", test)]:
        n1 = int(d["label"].sum())
        print(f"{name}: {len(d)} rows, {n1} attacks, benign:attack = {(len(d) - n1) / max(n1, 1):.2f}")
    print(f"xstest_safe: {len(xstest)} rows (eval only)")
    print("\n=== label fixes in final splits (rows whose label != label_original) ===")
    for name, d in [("train", train), ("val", val), ("test", test)]:
        up, down = int(((d.label_original == 0) & (d.label == 1)).sum()), int(((d.label_original == 1) & (d.label == 0)).sum())
        print(f"{name}: relabeled 0->1: {up}, 1->0: {down}")

    print("\n=== text length per label x source (train+val+test) ===")
    print(length_stats(pd.concat([train, val, test])).to_string())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["text", "label", "source", "label_original"]
    for name, d in [("train", train), ("val", val), ("test", test), ("xstest_safe", xstest)]:
        d[cols].to_parquet(args.out_dir / f"{name}.parquet", index=False)
    print(f"\nsaved to {args.out_dir}")

    if args.augment_indirect:
        import json

        import augment_indirect as ai

        tricky_df = pd.read_csv(Path(__file__).parent / "tricky.csv")
        dolly_ctx = load_dolly_contexts()
        aug, summary = ai.build(train, val, test, xstest, tricky_df, dolly_ctx, is_english, n_each=args.augment_n, seed=args.seed)
        aug.to_parquet(args.out_dir / "augmented_indirect.parquet", index=False)
        train_aug = pd.concat([train[cols], aug[cols]], ignore_index=True)
        train_aug.to_parquet(args.out_dir / "train_augmented.parquet", index=False)
        print("\n=== indirect-injection augmentation (train only) ===")
        print(json.dumps(summary, indent=2, default=str))
        print(f"train_augmented: {len(train)} + {len(aug)} rows; val/test/xstest/tricky untouched")

        import indirect_eval as ie

        ev, ev_summary = ie.build_eval(train, train_aug, val, test, xstest, tricky_df, dolly_ctx, is_english,
                                       n_aug_each=args.augment_n, seed=args.seed)
        ev.to_parquet(args.out_dir / "indirect_eval.parquet", index=False)
        print("\n=== indirect_eval (held-out, evaluation only; attacks from val+test) ===")
        print(json.dumps(ev_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
