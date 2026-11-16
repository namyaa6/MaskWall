"""Writes ml/train_armB_seed42.ipynb: a SMALL Colab notebook that retrains ONLY arm B (T1+T2+T3) with seed 42 and keeps everything on Google Drive,
so a runtime disconnect loses at most one epoch. Edit here, then rerun: python ml/make_armb_notebook.py

The code cells (data loading, tokenisation, windowing, evaluation) are COPIED from ml/train_hardneg.ipynb at build time, so the training recipe cannot drift from the
round that was pre-registered. Only three things differ: (1) one arm, one seed, (2) the upload zip is cached on Drive, (3) training saves a resumable state to Drive after every epoch.
ml/dry_run_hardneg_notebook.py <notebook> executes it locally on tiny data (a logic check; the numbers mean nothing)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = json.loads((HERE / "train_hardneg.ipynb").read_text(encoding="utf-8"))
RES = json.loads((HERE / "results" / "hardneg_results.json").read_text(encoding="utf-8"))
OUT = HERE / "train_armB_seed42.ipynb"


def src_of(marker, kind="code"):
    hits = [c for c in SRC["cells"] if c["cell_type"] == kind and marker in "".join(c["source"])]
    assert len(hits) == 1, f"{marker!r}: {len(hits)} matches"
    return "".join(hits[0]["source"])


def sub(text, old, new):
    assert text.count(old) == 1, f"{old!r} x{text.count(old)}"
    return text.replace(old, new)


# ---- reference numbers of the ORIGINAL arm B seed 42 run (Colab, hardneg_results.json) so the retrain can be compared on the spot
r = RES["results"]["B_neg_plus_attacks/seed42"]
REF = {"test_f1": round(r["test"]["f1"], 3), "block": round(r["thresholds"]["block"], 3), "E1_fp": r["E1"]["fp_block"]["k"], "E2_fp": r["E2"]["fp_block"]["k"],
       "E3_caught": r["E3"]["caught_block"]["k"], "test_recall": round(r["test"]["recall"], 3), "indirect_recall": round(r["indirect_test"]["recall"]["k"] / r["indirect_test"]["recall"]["n"], 3),
       "tricky_attack_recall": round(r["tricky"]["attack_recall"]["k"] / r["tricky"]["attack_recall"]["n"], 3), "best_epoch": r["best_epoch"]}

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md(r"""
# MaskWall: retrain ONLY arm B (seed 42) and save to Google Drive as it goes

Why: the first Colab run died, so its checkpoints are lost. Arm B (negatives + matched attacks) passed the four pre-registered criteria in that run
(see `docs/decisions.md`, Day 2.9). This notebook re-creates **one** checkpoint, same data, same code, same hyperparameters, seed 42.
It will not be bit-identical to the first run (GPU fp16 is not fully deterministic); the last cell prints the new numbers next to the original ones.
After you download the checkpoint, **int8 export and the re-evaluation happen on your PC**, and the served model is only replaced if it still passes there.

**Everything that matters is kept on Drive** (`MyDrive/maskwall_armB_seed42/`): the upload zip, a resumable state after every epoch, the final checkpoint.
If Colab disconnects, reopen the notebook and **Run all again**: it picks up from the last finished epoch and skips what is done.
Training is about 4 epochs x 1 minute on a T4 (an estimate, not a measurement).

## Before you run
1. **Runtime -> Change runtime type -> T4 GPU.**
2. Have `maskwall_hardneg_upload.zip` ready (in `colab_upload/` in the project). You only upload it **once**: it is copied to Drive.
3. **Run all.** Allow the Drive pop-up when it asks. Do not change anything after seeing numbers.
""")

md("## 1. Setup")
code(src_of("pip -q install"))
cfg = src_of("DRY_RUN = os.environ")
cfg = sub(cfg, "SEEDS = [42, 43, 44]", "SEEDS = [42]")
cfg = sub(cfg, "ARMS = {\"0_baseline\": [], \"A_negatives\": [\"T1\", \"T3\"], \"B_neg_plus_attacks\": [\"T1\", \"T2\", \"T3\"]}", "ARMS = {\"B_neg_plus_attacks\": [\"T1\", \"T2\", \"T3\"]}")
cfg = sub(cfg, "RUN_ORIGINAL_LABELS_CHECK = True", "RUN_ORIGINAL_LABELS_CHECK = False")
cfg = sub(cfg, "USE_DRIVE = False", "USE_DRIVE = True                                          # always: this is the point of this notebook")
cfg = sub(cfg, 'OUT = "/content/drive/MyDrive/maskwall_hardneg"', 'OUT = "/content/drive/MyDrive/maskwall_armB_seed42"')
cfg = sub(cfg, "SEEDS, MAX_EPOCHS, MIN_EPOCHS, BATCH = [42], 1, 1, 8", "SEEDS, MAX_EPOCHS, MIN_EPOCHS, BATCH = [42], int(os.environ.get('MW_DRY_EPOCHS', '1')), 1, 8")
cfg = cfg + f"\nREF = {json.dumps(REF)}                    # the ORIGINAL arm B seed 42 numbers (hardneg_results.json), for the comparison at the end\n"
code(cfg)

md("## 2. Upload the zip (once; cached on Drive) and load the data")
code(r"""
ZIP_ON_DRIVE = f"{OUT}/maskwall_hardneg_upload.zip"
if not DRY_RUN:
    if not os.path.exists(ZIP_ON_DRIVE):
        from google.colab import files
        up = files.upload()                                  # select maskwall_hardneg_upload.zip
        assert "maskwall_hardneg_upload.zip" in up, f"expected maskwall_hardneg_upload.zip, got {list(up)}"
        open(ZIP_ON_DRIVE, "wb").write(up["maskwall_hardneg_upload.zip"])
        print("saved the zip to Drive; after a disconnect you will not need to upload it again")
    else:
        print("using the zip already on Drive")
    zipfile.ZipFile(ZIP_ON_DRIVE).extractall(ROOT)
sys.path.insert(0, ROOT)
import train_baseline as tb                              # the SAME evaluation code as every earlier result
print(sorted(os.listdir(ROOT)), sorted(os.listdir(f"{ROOT}/data/processed")))
""")
code(src_of("P = lambda n: pd.read_parquet"))

md("## 3. Tokenisation, prediction, and a training loop that saves to Drive after every epoch")
fn = src_of("def encode_head_tail")
head = fn[:fn.index("def train_one(")]
code(head)
code(r"""
def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)                                   # a disconnect during the write never leaves a half-written file


def train_one(tag, seed, train_df, val_df, state_dir):
    # Same recipe as train_hardneg.ipynb (AdamW, linear warmup, fp16, best epoch by val AP, early stop from epoch 3), plus a resumable state on Drive.
    os.makedirs(state_dir, exist_ok=True)
    STATE, BEST = f"{state_dir}/state.pt", f"{state_dir}/best.pt"
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(HF_MODEL); tok.model_max_length = 10**6
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL, num_labels=2).float().to(DEV)
    enc_tr, enc_va = encode_head_tail(tok, train_df.text), encode_head_tail(tok, val_df.text)
    y_tr, y_va = train_df.label.to_numpy(), val_df.label.to_numpy()
    n_tr = len(enc_tr)
    total = int(np.ceil(n_tr / BATCH)) * MAX_EPOCHS
    no_decay = ["bias", "LayerNorm.weight", "layernorm", "LayerNorm.bias"]
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": WEIGHT_DECAY},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=LR)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_FRAC * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=FP16)
    history, best_ap, best_epoch, stale, done_epochs, finished, secs0 = [], -1.0, 0, 0, 0, False, 0.0
    if os.path.exists(STATE):                                # ---- resume
        st = torch.load(STATE, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        history, best_ap, best_epoch, stale, done_epochs, finished, secs0 = st["history"], st["best_ap"], st["best_epoch"], st["stale"], st["epoch"], st["finished"], st["secs"]
        print(f"[{tag}] RESUMING after epoch {done_epochs} (finished={finished}); history so far: {[(h['epoch'], round(h['val_ap'], 4)) for h in history]}")
    t0 = time.time()
    for epoch in range(done_epochs + 1, MAX_EPOCHS + 1):
        if finished:
            break
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(n_tr)
        running = 0.0
        for i in range(0, n_tr, BATCH):
            idx = order[i:i + BATCH]
            ids, att = collate([enc_tr[j] for j in idx], tok.pad_token_id)
            labels = torch.tensor(y_tr[idx], dtype=torch.long).to(DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=FP16):
                loss = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV), labels=labels).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item() * len(idx)
        vs = predict_enc(model, enc_va, tok.pad_token_id)
        ap = float(average_precision_score(y_va, vs))
        history.append({"epoch": epoch, "train_loss": running / n_tr, "val_ap": ap, "val_best_f1": tb.tune_f1_threshold(y_va, vs)[1]})
        print(f"[{tag}] epoch {epoch}: loss {running / n_tr:.4f}  val AP {ap:.4f}  ({secs0 + time.time() - t0:.0f}s)")
        if ap > best_ap:
            best_ap, best_epoch, stale = ap, epoch, 0
            _atomic_save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, BEST)
        else:
            stale += 1
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                print(f"[{tag}] early stop after epoch {epoch} (best {best_epoch})")
                finished = True
        finished = finished or epoch == MAX_EPOCHS
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "history": history,
                      "best_ap": best_ap, "best_epoch": best_epoch, "stale": stale, "epoch": epoch, "finished": finished, "secs": secs0 + time.time() - t0}, STATE)
        print(f"[{tag}] state saved to Drive after epoch {epoch}")
    model.load_state_dict(torch.load(BEST, map_location="cpu"))
    return model, tok, history, best_epoch, secs0 + time.time() - t0
""")

md("## 4. Evaluation (copied unchanged from the main notebook)")
code(src_of("def rate(mask)"))

md(r"""
## 5. Train arm B, seed 42 (resumable), tune thresholds on its own val, evaluate, save the final checkpoint to Drive
Thresholds come from the arm's val only (old val + the new val slice of T1/T2/T3), exactly as in the main round: block = max(thr_F1, thr_FPR<=1%), warn = min.
""")
code(r"""
ARM, SEED = "B_neg_plus_attacks", 42
tiers = ARMS[ARM]
tr = pd.concat([S["train_augmented"][["text", "label"]], hn_tr[hn_tr.tier.isin(tiers)][["text", "label"]]], ignore_index=True)
va = pd.concat([S["val"][["text", "label"]], hn_va[hn_va.tier.isin(tiers)][["text", "label"]]], ignore_index=True)
print(f"{ARM} seed {SEED}: {len(tr)} train rows ({int(tr.label.sum())} attacks), {len(va)} val rows")

FINAL = f"{OUT}/final_armB_seed42"
model, tok, history, best_epoch, secs = train_one(f"{ARM}/seed{SEED}", SEED, tr, va, f"{OUT}/state_armB_seed42")
vs = predict(model, tok, list(va.text))
thr = thresholds(va.label.to_numpy(), vs)
m, sc = evaluate(model, tok, thr)
m.update({"arm": ARM, "seed": SEED, "history": history, "best_epoch": best_epoch, "train_seconds": round(secs), "n_train": len(tr), "n_val": len(va)})
os.makedirs(FINAL, exist_ok=True)
model.save_pretrained(FINAL); tok.save_pretrained(FINAL)                 # config.json, model.safetensors, tokenizer files
json.dump({"result": m, "scores": {k: [round(float(x), 5) for x in v] for k, v in sc.items()}, "ref_original_run": REF}, open(f"{FINAL}/colab_eval.json", "w"), default=float)
print("saved the final checkpoint to", FINAL, sorted(os.listdir(FINAL)))
""")

md("## 6. Does it look like the original arm B seed 42? (a sanity check, not a decision)")
code(r"""
f = m["test"]
now = {"test_f1": round(f["f1"], 3), "block": round(thr["block"], 3), "E1_fp": m["E1"]["fp_block"]["k"], "E2_fp": m["E2"]["fp_block"]["k"], "E3_caught": m["E3"]["caught_block"]["k"],
       "test_recall": round(f["recall"], 3), "indirect_recall": round(m["indirect_test"]["recall"]["k"] / m["indirect_test"]["recall"]["n"], 3),
       "tricky_attack_recall": round(m["tricky"]["attack_recall"]["k"] / m["tricky"]["attack_recall"]["n"], 3), "best_epoch": best_epoch}
print(f"{'metric':22s} {'original run':>14s} {'this retrain':>14s}")
for k in REF:
    print(f"{k:22s} {REF[k]:>14} {now[k]:>14}")
print("\nExpect small differences (fp16 GPU nondeterminism; the first run's seed spread was E1 1-2, E2 1-2, E3 22-24 of 24). Do NOT retrain to get nicer numbers.")
""")

md("## 7. Download the checkpoint (it also stays on Drive)")
code(r"""
import shutil
if not DRY_RUN:
    from google.colab import files
    shutil.make_archive("/content/hardneg_B_seed42", "zip", FINAL)
    files.download("/content/hardneg_B_seed42.zip")
    print("Downloaded hardneg_B_seed42.zip. Put it in models/ckpt/ on your PC (the zip itself is fine; the export script unpacks it).")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, len(cells), "cells; reference:", REF)
