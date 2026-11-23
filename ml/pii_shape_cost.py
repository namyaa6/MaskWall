"""Cost of hiding Aadhaar/PAN by SHAPE (Day 4.21): how often does the looser rule hide something harmless?

Compares, on every benign text we already have (label 0 in the injection splits, the tricky set, hard negatives, benign prompt
talk, the name-detection sample sentences):
  strict  = the rule before Day 4.21 (Aadhaar needs first digit 2-9 + Verhoeff; PAN needs a valid 4th letter), frozen here
  shape_421 = the Day 4.21 rule, frozen here: PAN shape allowed spaces/hyphens in any case
  shape   = the shipped rule (backend/app/scanners/pii_regex.py, Day 4.22): 12 digits / 5 letters + 4 digits + 1 letter; the PAN
            may contain spaces or hyphens only when written in UPPERCASE
Only Aadhaar and PAN are compared (email and phone rules did not change). No label or model is involved; nothing is tuned.
Writes ml/results/pii_shape_cost.json (the matched snippets are benign dataset text). Run: python ml/pii_shape_cost.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.normalize import normalize_text  # noqa: E402
from app.scanners.pii_regex import mask, verhoeff_valid  # noqa: E402

OLD_AADHAAR = re.compile(r"(?<!\d)(?<!\d[ -])([2-9]\d{3}[ -]?\d{4}[ -]?\d{4})(?![ -]?\d)")
OLD_PAN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{3}[PCHFATBLJGpchfatbljg][A-Za-z][ -]?\d{4}[ -]?[A-Za-z])(?![A-Za-z0-9])")
PAN_421 = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{5}[ -]?\d{4}[ -]?[A-Za-z])(?![A-Za-z0-9])")


def strict_hits(t: str) -> list[str]:
    out = [m.group(1) for m in OLD_AADHAAR.finditer(t) if verhoeff_valid(re.sub(r"\D", "", m.group(1)))]
    return out + [m.group(1) for m in OLD_PAN.finditer(t)]


def shape_hits(t: str) -> list[str]:
    return [t[f.start:f.end] for f in mask(t).findings if f.type in ("AADHAAR", "PAN")]


def shape_421_hits(t: str) -> list[str]:
    hits = [t[f.start:f.end] for f in mask(t).findings if f.type == "AADHAAR"]
    return hits + [m.group(1) for m in PAN_421.finditer(t)]


def load() -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    for n in ("train", "val", "test", "xstest_safe", "indirect_eval"):
        d = pd.read_parquet(ROOT / "colab_upload" / f"{n}.parquet")
        sets[f"{n} (benign)"] = d.loc[d["label"] == 0, "text"].astype(str).tolist()
    for name, path in [("tricky (benign)", "ml/data/tricky.csv"), ("hardneg_train (benign)", "ml/data/hardneg/hardneg_train.csv"),
                       ("hardneg_val (benign)", "ml/data/hardneg/hardneg_val.csv"), ("prompt_talk_benign_ai", "ml/data/prompt_talk_benign_ai.csv")]:
        d = pd.read_csv(ROOT / path)
        sets[name] = d.loc[d["label"] == 0, "text"].astype(str).tolist()
    for f in ("pii_names_sample", "pii_names_lowercase", "pii_names_lowercase_r2"):
        rows = [json.loads(line) for line in (ROOT / "ml/data" / f"{f}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        sets[f"{f} (name sentences)"] = [r["text"] for r in rows]
    return sets


def main() -> None:
    result, total = {}, {"texts": 0, "strict": 0, "shape": 0, "shape_421": 0}
    for name, texts in load().items():
        texts = [normalize_text(t).text for t in texts]
        s = [strict_hits(t) for t in texts]
        sh = [shape_hits(t) for t in texts]
        c = [shape_421_hits(t) for t in texts]
        new_only = [(t, h) for t, a, h in zip(texts, s, sh) if h and not a]
        row = {"texts": len(texts), "strict": sum(map(bool, s)), "shape": sum(map(bool, sh)), "shape_421": sum(map(bool, c)),
               "newly_hidden_examples": [{"matched": h, "context": t[max(0, t.find(h[0]) - 40): t.find(h[0]) + len(h[0]) + 40]} for t, h in new_only[:10]]}
        result[name] = row
        total["texts"] += len(texts)
        for k in ("strict", "shape", "shape_421"):
            total[k] += row[k]
        print(f"{name:38s} n={row['texts']:5d}  strict={row['strict']:3d}  shape={row['shape']:3d}  shape_421={row['shape_421']:3d}")
    result["TOTAL"] = total
    print(f"{'TOTAL':38s} n={total['texts']:5d}  strict={total['strict']:3d}  shape={total['shape']:3d}  shape_421={total['shape_421']:3d}")
    out = ROOT / "ml" / "results" / "pii_shape_cost.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))
    for name, row in result.items():
        for ex in row.get("newly_hidden_examples", []):
            print(f"  [{name}] {ex['matched']!r}  in  ...{ex['context']!r}...")


if __name__ == "__main__":
    main()
