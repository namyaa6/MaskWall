"""Split indirect_eval.parquet into indirect_val and indirect_test (decided after seeing indirect_eval results).

  indirect_val  = attacks built from VAL attack strings + a random half of the benign rows   -> used by the amended
                  server-model selection rule (docs/decisions.md Day 1.9), so selection can see indirect injection
  indirect_test = attacks built from TEST attack strings + the other half of the benign rows -> reporting only

Benign rows have no attack-string origin, so they are split at random (seed 42). Every model's threshold is still tuned on the
main val set only; indirect_val is only ever used to ask "does recall hold at that threshold".
Run from the repo root: python ml/data/split_indirect_eval.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path(__file__).parent / "processed"
SEED = 42


def main():
    ev = pd.read_parquet(PROC / "indirect_eval.parquet")
    benign = ev.index[ev["label"] == 0].to_numpy()
    half = np.random.default_rng(SEED).permutation(benign)
    to_val = set(half[: len(half) // 2])
    is_val = ev["attack_split"].eq("val") | ev.index.isin(to_val)
    is_test = ev["attack_split"].eq("test") | (ev["label"].eq(0) & ~ev.index.isin(to_val))
    assert (is_val ^ is_test).all(), "every row must land in exactly one part"
    val, test = ev[is_val].reset_index(drop=True), ev[is_test].reset_index(drop=True)
    assert not set(val["text"]) & set(test["text"])
    val.to_parquet(PROC / "indirect_val.parquet", index=False)
    test.to_parquet(PROC / "indirect_test.parquet", index=False)
    for name, d in [("indirect_val", val), ("indirect_test", test)]:
        print(f"{name}: {len(d)} rows, {int(d.label.sum())} attacks, {int((d.label == 0).sum())} benign, "
              f"wrapped {d.groupby('label').wrapped.mean().round(2).to_dict()}, attack sources {d[d.label == 1].attack_source.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
