"""Local CPU smoke test of ml/train_hardneg.ipynb: executes the notebook's code cells on tiny subsets (MW_DRY_RUN=1: 1 seed, 1 epoch, 300 train rows).
It checks that the code runs end to end and writes sane files; the numbers it prints mean NOTHING. Needs the hard-negative CSVs (build_hardneg.py).
Run from the repo root: python ml/dry_run_hardneg_notebook.py"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="mw_dry_", dir=ROOT / "ml" / "cache"))
shutil.copy(ROOT / "ml" / "train_baseline.py", tmp / "train_baseline.py")
shutil.copytree(ROOT / "ml" / "data" / "processed", tmp / "data" / "processed")
shutil.copy(ROOT / "ml" / "data" / "tricky.csv", tmp / "data" / "tricky.csv")
shutil.copytree(ROOT / "ml" / "data" / "hardneg", tmp / "data" / "hardneg")
os.environ.update(MW_DRY_RUN="1", MW_ROOT=str(tmp), MW_OUT=str(tmp / "out"))
nb = json.loads((ROOT / "ml" / "train_hardneg.ipynb").read_text(encoding="utf-8"))
g = {"display": print, "__name__": "__main__"}
os.chdir(tmp)
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.lstrip().startswith("!pip"):
        continue
    print(f"--- cell {i}", flush=True)
    exec(compile(src, f"<cell {i}>", "exec"), g)
res = json.loads((tmp / "out" / "hardneg_results.json").read_text())
print("runs:", sorted(res["results"]))
print("dry run OK; scratch folder:", tmp)
