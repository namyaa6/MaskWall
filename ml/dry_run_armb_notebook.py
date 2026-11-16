"""Local CPU smoke test of ml/train_armB_seed42.ipynb (tiny data; the numbers mean NOTHING). Also tests the RESUME path:
  run 1: 2 epochs, the process 'dies' right after the first state file is written (simulated by making os.replace raise once, after it succeeded);
  run 2: same output folder -> must print RESUMING after epoch 1, finish epoch 2, write the final checkpoint;
  run 3: same folder again -> must report finished=True and train nothing.
Run from the repo root: python ml/dry_run_armb_notebook.py"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="mw_dry_armb_", dir=ROOT / "ml" / "cache"))
shutil.copy(ROOT / "ml" / "train_baseline.py", tmp / "train_baseline.py")
shutil.copytree(ROOT / "ml" / "data" / "processed", tmp / "data" / "processed")
shutil.copy(ROOT / "ml" / "data" / "tricky.csv", tmp / "data" / "tricky.csv")
shutil.copytree(ROOT / "ml" / "data" / "hardneg", tmp / "data" / "hardneg")
os.environ.update(MW_DRY_RUN="1", MW_ROOT=str(tmp), MW_OUT=str(tmp / "out"), MW_DRY_EPOCHS="2")
nb = json.loads((ROOT / "ml" / "train_armB_seed42.ipynb").read_text(encoding="utf-8"))
os.chdir(tmp)


def run(crash_after_first_state):
    g = {"display": print, "__name__": "__main__"}
    real_replace = os.replace
    n = {"state": 0}

    def flaky(a, b):
        real_replace(a, b)
        if str(b).endswith("state.pt"):
            n["state"] += 1
            if crash_after_first_state and n["state"] == 1:
                raise KeyboardInterrupt("simulated disconnect after epoch 1 was saved")
    os.replace = flaky
    try:
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"])
            if src.lstrip().startswith("!pip"):
                continue
            exec(compile(src, f"<cell {i}>", "exec"), g)
    finally:
        os.replace = real_replace
    return g


try:
    run(True)
    raise SystemExit("run 1 should have been interrupted")
except KeyboardInterrupt as e:
    print(">>> run 1 interrupted as planned:", e)
print("\n>>> run 2 (resume)")
run(False)
assert (tmp / "out" / "final_armB_seed42" / "model.safetensors").exists()
assert (tmp / "out" / "final_armB_seed42" / "colab_eval.json").exists()
print("\n>>> run 3 (already finished: must not train)")
run(False)
print("dry run OK:", tmp)
