"""Packs everything the Colab notebook ml/train_hardneg.ipynb needs into ONE zip: colab_upload/maskwall_hardneg_upload.zip (gitignored).
Run from the repo root after build_hardneg.py: python ml/pack_hardneg_upload.py"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "ml" / "data" / "processed"
FILES = {
    "train_baseline.py": ROOT / "ml" / "train_baseline.py",
    "data/tricky.csv": ROOT / "ml" / "data" / "tricky.csv",
    "data/hardneg/hardneg_train.csv": ROOT / "ml" / "data" / "hardneg" / "hardneg_train.csv",
    "data/hardneg/hardneg_val.csv": ROOT / "ml" / "data" / "hardneg" / "hardneg_val.csv",
    **{f"data/processed/{n}.parquet": PROC / f"{n}.parquet" for n in
       ["train_augmented", "val", "test", "xstest_safe", "indirect_test", "long_eval",
        "hinglish_human_benign", "hinglish_attacks_ai", "prompt_talk_benign_ai", "manual7"]},
}


def main():
    missing = [k for k, v in FILES.items() if not v.exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    out = ROOT / "colab_upload" / "maskwall_hardneg_upload.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, p in FILES.items():
            z.write(p, arc)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB):")
    for arc, p in FILES.items():
        print(f"  {arc:48s} {p.stat().st_size / 1e3:9.1f} kB")


if __name__ == "__main__":
    main()
