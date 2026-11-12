"""Provenance script: drop the whole "exception to AI usual ethical protocols" template cluster.

Why: the same fake-system-note template appears with both labels (hand-relabeled to 1 in the first
cleaning round, ~180 others still 0). Any label on it is inconsistent, and a model can memorize the
template instead of learning anything. Rather than review 263 near-identical rows, the whole cluster
is dropped (all splits, including rows the loader later drops via sampling/dedup, so the rule is closed).

The template is matched on its fixed phrase ("exception to <name> usual ethical protocols", which also
catches "Al's" and a stylized-name variant). Output: rows appended to ml/data/label_fixes.csv as drops;
existing relabel entries for the same text are replaced (hash is unique per text).

Run once from the repo root: python ml/data/make_template_drops.py
"""
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "data"))
import load_datasets as ld  # noqa: E402

PATTERN = re.compile(r"exception to .{1,40}? usual ethical protocols")
REASON = "templated cluster; inconsistent labels and memorization risk"
FIXES = ROOT / "ml/data/label_fixes.csv"


def main():
    raw = pd.concat([f() for f in [ld.load_deepset, ld.load_jackhhao, ld.load_gandalf,
                                   ld.load_trustairlab_jailbreak, ld.load_trustairlab_regular, ld.load_dolly]],
                    ignore_index=True)
    df = ld.clean(raw)
    df = df[df["norm"].map(lambda t: bool(PATTERN.search(t)))].copy()
    df["hash"] = df["norm"].map(lambda t: hashlib.sha256(t.encode()).hexdigest())
    # one entry per distinct text; old_label must be the original label (loader asserts all rows agree)
    g = df.groupby("hash").agg(old=("label", "first"), n_labels=("label", "nunique"), source=("source", "first"),
                               text=("text", "first")).reset_index()
    assert (g["n_labels"] == 1).all(), "same text carries two original labels"
    print("template rows in raw (cleaned, pre-dedup):", len(df), "| distinct texts:", len(g))
    print(g.groupby(["source", "old"]).size().to_string())

    fx = pd.read_csv(FIXES, dtype={"new_label": str})
    replaced = fx[fx["text_hash"].isin(g["hash"])]
    print("existing fixes replaced (previous decision):", replaced.groupby(["origin", "new_label"]).size().to_dict())
    fx = fx[~fx["text_hash"].isin(g["hash"])]
    new = pd.DataFrame({"text_hash": g["hash"], "old_label": g["old"], "new_label": "drop", "reason": REASON,
                        "source": g["source"], "origin": "template_cluster",
                        "snippet": g["text"].map(lambda t: " ".join(t.split())[:100])})
    out = pd.concat([fx, new], ignore_index=True)
    assert out["text_hash"].is_unique
    out.to_csv(FIXES, index=False)
    print("label_fixes.csv rows:", len(out), out.groupby("new_label").size().to_dict())


if __name__ == "__main__":
    main()
