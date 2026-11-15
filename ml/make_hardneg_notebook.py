"""Writes ml/train_hardneg.ipynb (the single Colab notebook for the hard-negative round). Edit the cells HERE, then rerun:
    python ml/make_hardneg_notebook.py
ml/dry_run_hardneg_notebook.py executes the same code cells locally on CPU with tiny data (a syntax/logic check, not an experiment)."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "train_hardneg.ipynb"
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})


md(r"""
# MaskWall hard-negative round (Colab GPU): baseline x 3 seeds, arms A and B x 3 seeds, original-labels check

One notebook, MiniLM only (`sentence-transformers/all-MiniLM-L6-v2`), everything else exactly as in `train_transformer.ipynb`
(head+tail 256+254, lr 5e-5, batch 32, up to 4 epochs / early stopping from epoch 3 on val AP, fp16, linear warmup). Plan, label policy,
data policy and the **pre-registered success criteria** are in `docs/decisions.md` (Day 2.6 and Day 2.8). **Nothing is tuned on any evaluation set.**

| arm | training data | validation data (epoch choice and thresholds) |
|---|---|---|
| `0_baseline` | `train_augmented` (the data the served MiniLM was trained on) | old `val` |
| `A_negatives` | + T1 (benign prompt-talk, Groq) + T3 (human Hinglish-TOP commands + English originals) | old `val` + the new val slice of T1/T3 |
| `B_neg_plus_attacks` | + T1 + T2 (Groq attacks, matched to T1) + T3 | old `val` + the new val slice of T1/T2/T3 |
| `orig_labels` | `train_augmented`, **published** labels (`label_original`) | old `val`, published labels |

3 seeds each (42, 43, 44) = 12 runs, about 3 to 5 minutes each on a T4 (an estimate, not a measurement). The baseline additionally reports an
operating point tuned on the *new* val (`0_baseline@valB`), so a threshold shift can be told apart from a training effect.

**Operating point = the one the server uses:** block = max(thr_F1, thr_FPR<=1%) and warn = min of the two, both tuned on that arm's val only.
Evaluation-only sets (never trained on, never used for any choice): test, tricky, xstest_safe, indirect_test, E1 human Hinglish benign (104) + English
originals, E2 prompt-talk benign (100), E3 Hinglish attacks (24), E4 the owner's 7 prompts, long documents (228, scored with sliding windows, cap 6).
All raw scores are saved so the paired tests (McNemar, paired bootstrap) are done locally.

## Before you run
1. **Runtime -> Change runtime type -> T4 GPU.** The notebook stops if there is no GPU.
2. Upload **one file** when the upload cell asks: `maskwall_hardneg_upload.zip` (built locally by `python ml/pack_hardneg_upload.py`).
3. Run all cells top to bottom (the loop cell takes about 45 to 60 minutes; results are saved after every run, so a disconnect loses at most one run;
   set `USE_DRIVE = True` to also keep checkpoints on Drive).
4. At the end download `hardneg_results.json` and `hardneg_summary.md` into `ml/results/`. Do not edit numbers. **Do not rerun cells after looking at results
   to "improve" them.**
""")

md("## 1. Setup")
code(r"""
!pip -q install "transformers>=4.44" accelerate sentencepiece protobuf scikit-learn pandas pyarrow joblib tabulate
""")
code(r"""
import hashlib, json, os, platform, random, re, sys, time, zipfile
import numpy as np
import pandas as pd
import torch

DRY_RUN = os.environ.get("MW_DRY_RUN") == "1"          # local CPU smoke test only; always False in Colab
DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda" or DRY_RUN, "No GPU: Runtime -> Change runtime type -> T4 GPU"
print(torch.__version__, torch.cuda.get_device_name(0) if DEV == "cuda" else "CPU (dry run)")

# ------------------------------------------------------------------ config (fixed before any result)
SEEDS = [42, 43, 44]
HF_MODEL, LR, BATCH = "sentence-transformers/all-MiniLM-L6-v2", 5e-5, 32
HEAD, TAIL = 256, 254
MAX_EPOCHS, MIN_EPOCHS, PATIENCE = 4, 3, 1
WARMUP_FRAC, WEIGHT_DECAY, GRAD_CLIP = 0.1, 0.01, 1.0
FP16 = DEV == "cuda"
CONTENT, STRIDE, MAX_WINDOWS = 510, 255, 6              # same as backend/app/scanners/windowing.py and the served default (sliding, cap 6)
ARMS = {"0_baseline": [], "A_negatives": ["T1", "T3"], "B_neg_plus_attacks": ["T1", "T2", "T3"]}
RUN_ORIGINAL_LABELS_CHECK = True
USE_DRIVE = False

ROOT = os.environ.get("MW_ROOT", "/content/maskwall")
OUT = os.environ.get("MW_OUT", "/content/out")
if USE_DRIVE and not DRY_RUN:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/maskwall_hardneg"
os.makedirs(ROOT, exist_ok=True); os.makedirs(OUT, exist_ok=True)

if DRY_RUN:                                             # tiny and fast
    SEEDS, MAX_EPOCHS, MIN_EPOCHS, BATCH = [42], 1, 1, 8


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(s)
""")

md("## 2. Upload the zip and load the data")
code(r"""
if not DRY_RUN:
    from google.colab import files
    up = files.upload()                                  # select maskwall_hardneg_upload.zip
    assert "maskwall_hardneg_upload.zip" in up, f"expected maskwall_hardneg_upload.zip, got {list(up)}"
    open("/content/up.zip", "wb").write(up["maskwall_hardneg_upload.zip"])
    zipfile.ZipFile("/content/up.zip").extractall(ROOT)
sys.path.insert(0, ROOT)
import train_baseline as tb                              # the SAME evaluation code as every earlier result
print(sorted(os.listdir(ROOT)), sorted(os.listdir(f"{ROOT}/data/processed")))
""")
code(r"""
P = lambda n: pd.read_parquet(f"{ROOT}/data/processed/{n}.parquet")
S = {n: P(n) for n in ["train_augmented", "val", "test", "xstest_safe", "indirect_test", "long_eval",
                       "hinglish_human_benign", "hinglish_attacks_ai", "prompt_talk_benign_ai", "manual7"]}
tricky = pd.read_csv(f"{ROOT}/data/tricky.csv")
hn_tr = pd.read_csv(f"{ROOT}/data/hardneg/hardneg_train.csv")
hn_va = pd.read_csv(f"{ROOT}/data/hardneg/hardneg_val.csv")

for n, d in S.items():
    print(f"{n:24s} {len(d):5d} rows" + (f", {int(d.label.sum())} attacks" if "label" in d else ""))
print(f"{'tricky':24s} {len(tricky):5d} rows")
print("hardneg train:", hn_tr.groupby(["tier", "label"]).size().to_dict())
print("hardneg val:  ", hn_va.groupby(["tier", "label"]).size().to_dict())

# ---- leakage: exact-text overlap between the NEW training/val rows and every evaluation set (build_hardneg.py also checked near-duplicates)
norm = lambda t: " ".join(t.split()).lower()
eval_text = set()
for n in ["val", "test", "xstest_safe", "indirect_test", "long_eval", "hinglish_human_benign", "hinglish_attacks_ai", "prompt_talk_benign_ai", "manual7"]:
    eval_text |= set(map(norm, S[n].text))
eval_text |= set(map(norm, S["hinglish_human_benign"].en_control)) | set(map(norm, tricky.text))
assert not set(map(norm, hn_tr.text)) & eval_text, "hardneg train overlaps an evaluation set"
assert not set(map(norm, hn_va.text)) & eval_text, "hardneg val overlaps an evaluation set"
assert not set(map(norm, hn_tr.text)) & set(map(norm, hn_va.text)), "hardneg train overlaps hardneg val"
assert not set(map(norm, S["train_augmented"].text)) & set(map(norm, S["val"].text)) and not set(map(norm, S["train_augmented"].text)) & set(map(norm, S["test"].text))
print("no exact overlap between training data and any evaluation/validation set")

if DRY_RUN:
    S["train_augmented"] = S["train_augmented"][S["train_augmented"].text.str.len() < 600].sample(300, random_state=0).reset_index(drop=True)   # short rows only: CPU memory
    S["val"] = S["val"].sample(150, random_state=0).reset_index(drop=True)
    hn_tr = hn_tr.groupby("tier").head(60).reset_index(drop=True)
    hn_va = hn_va.groupby("tier").head(20).reset_index(drop=True)
    for _n, _k in [("test", 100), ("xstest_safe", 40), ("indirect_test", 30), ("long_eval", 8)]:      # keep the CPU smoke test fast and small in memory
        S[_n] = S[_n].sample(_k, random_state=0).reset_index(drop=True)
""")

md(r"""
## 3. Tokenisation, prediction (head+tail for short prompts, sliding windows for long documents), training loop
Identical to the Step 4 notebook except that the device is generic and the long documents can be scored with the server's sliding windows.
""")
code(r"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score, roc_auc_score


def encode_head_tail(tok, texts):
    ids_all = tok(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    out = []
    for ids in ids_all:
        if len(ids) > HEAD + TAIL:
            ids = ids[:HEAD] + ids[-TAIL:]
        out.append([tok.cls_token_id] + ids + [tok.sep_token_id])
    return out


def window_spans(n, content=CONTENT, stride=STRIDE):                 # copy of backend/app/scanners/windowing.py
    if n <= content:
        return [(0, n)]
    spans, s = [], 0
    while s + content < n:
        spans.append((s, s + content)); s += stride
    spans.append((n - content, n))
    return spans


def select_windows(spans, cap):
    if len(spans) <= cap:
        return spans, False
    return spans[:cap - 1] + [spans[-1]], True


def collate(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    return ids, att


@torch.no_grad()
def predict_enc(model, enc, pad_id, bs=64 if DEV == 'cuda' else 8):
    model.eval()
    order = np.argsort([len(x) for x in enc])
    probs = np.zeros(len(enc))
    for i in range(0, len(enc), bs):
        idx = order[i:i + bs]
        ids, att = collate([enc[j] for j in idx], pad_id)
        with torch.autocast(DEV, dtype=torch.float16, enabled=FP16):
            logits = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits
        probs[idx] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    return probs


def predict(model, tok, texts):
    return predict_enc(model, encode_head_tail(tok, texts), tok.pad_token_id)


def predict_sliding(model, tok, texts):
    ids_all = tok(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    flat, owner = [], []
    for d, ids in enumerate(ids_all):
        spans, _ = select_windows(window_spans(len(ids)), MAX_WINDOWS)
        for s, e in spans:
            flat.append([tok.cls_token_id] + ids[s:e] + [tok.sep_token_id]); owner.append(d)
    p = predict_enc(model, flat, tok.pad_token_id, bs=32 if DEV == 'cuda' else 4)
    out = np.zeros(len(ids_all))
    for pi, d in zip(p, owner):
        out[d] = max(out[d], pi)
    return out


def train_one(tag, seed, train_df, val_df):
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
    history, best_ap, best_state, best_epoch, stale = [], -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
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
        print(f"[{tag}] epoch {epoch}: loss {running / n_tr:.4f}  val AP {ap:.4f}  ({time.time() - t0:.0f}s)")
        if ap > best_ap:
            best_ap, best_epoch, stale = ap, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                print(f"[{tag}] early stop after epoch {epoch} (best {best_epoch})"); break
    model.load_state_dict(best_state)
    return model, tok, history, best_epoch, time.time() - t0
""")

md(r"""
## 4. Evaluation of one trained model
Thresholds come from the arm's **val only**. Everything else is scored once with those frozen thresholds. Rates are reported at **block** (the served
operating point) and at **warn or block**; `tb.evaluate_split` etc. are the baseline's own functions.
""")
code(r"""
def rate(mask):
    mask = np.asarray(mask, bool)
    return {"k": int(mask.sum()), "n": int(len(mask))}


def thresholds(y, scores):
    t_f1, f1 = tb.tune_f1_threshold(y, scores)
    t_fpr = tb.tune_fpr_threshold(y, scores)
    return {"thr_f1": t_f1, "thr_fpr1pct": t_fpr, "block": max(t_f1, t_fpr), "warn": min(t_f1, t_fpr), "val_f1": f1}


def sets_for_scoring():
    # name -> list of texts, in a fixed order (the saved raw scores are positional)
    h = S["hinglish_human_benign"]
    return {"test": S["test"].text, "tricky": tricky.text, "xstest_safe": S["xstest_safe"].text, "indirect_test": S["indirect_test"].text,
            "E1_hinglish": h.text, "E1_english_control": h.en_control, "E2_prompt_talk": S["prompt_talk_benign_ai"].text,
            "E3_hinglish_attacks": S["hinglish_attacks_ai"].text, "E4_manual7": S["manual7"].text}


def evaluate(model, tok, thr, label_col="label"):
    # Score every evaluation set with the frozen thresholds in `thr` (from thresholds()). Returns (metrics, raw scores).
    sc = {n: predict(model, tok, list(t)) for n, t in sets_for_scoring().items()}
    sc["long_eval"] = predict_sliding(model, tok, list(S["long_eval"].text))
    blk, wrn = thr["block"], thr["warn"]
    test_y = S["test"][label_col].to_numpy()
    m = {"thresholds": thr}
    m["test"] = {**tb.rates(test_y, sc["test"] >= blk), "roc_auc": float(roc_auc_score(test_y, sc["test"])), "pr_auc": float(average_precision_score(test_y, sc["test"])),
                 "at_f1_threshold": tb.rates(test_y, sc["test"] >= thr["thr_f1"])}
    t = S["test"].assign(label=test_y)
    m["test_full_eval_at_block"] = tb.evaluate_split(t, sc["test"], blk, thr["thr_fpr1pct"])
    tk = tricky.assign(pred=sc["tricky"] >= blk)
    att = tk[tk.label == 1]
    m["tricky"] = {"overall": int((tk.pred == tk.label).sum()), "n": len(tk), "attack_recall": rate(att.pred), "benign_fp": rate(tk[tk.label == 0].pred),
                   "by_category": {c: {"correct": int((g.pred == g.label).sum()), "n": len(g), "label": int(g.label.iloc[0])} for c, g in tk.groupby("category")}}
    m["xstest_safe_fp"] = rate(sc["xstest_safe"] >= blk)
    ie = S["indirect_test"]
    m["indirect_test"] = {"recall": rate((sc["indirect_test"] >= blk)[ie.label.to_numpy() == 1]), "fpr": rate((sc["indirect_test"] >= blk)[ie.label.to_numpy() == 0])}
    m["E1"] = {"fp_block": rate(sc["E1_hinglish"] >= blk), "fp_warn": rate(sc["E1_hinglish"] >= wrn), "english_control_fp_block": rate(sc["E1_english_control"] >= blk)}
    p = S["prompt_talk_benign_ai"]
    m["E2"] = {"fp_block": rate(sc["E2_prompt_talk"] >= blk), "fp_warn": rate(sc["E2_prompt_talk"] >= wrn),
               "by_language": {lg: rate((sc["E2_prompt_talk"] >= blk)[(p.language == lg).to_numpy()]) for lg in p.language.unique()},
               "by_category": {c: rate((sc["E2_prompt_talk"] >= blk)[(p.category == c).to_numpy()]) for c in p.category.unique()}}
    a = S["hinglish_attacks_ai"]
    m["E3"] = {"caught_block": rate(sc["E3_hinglish_attacks"] >= blk), "caught_warn": rate(sc["E3_hinglish_attacks"] >= wrn),
               "by_category": {c: rate((sc["E3_hinglish_attacks"] >= blk)[(a.category == c).to_numpy()]) for c in a.category.unique()}}
    m["E4"] = [{"text": x[:60], "label": int(l), "score": float(s)} for x, l, s in zip(S["manual7"].text, S["manual7"].label, sc["E4_manual7"])]
    lg = S["long_eval"]
    s_long = sc["long_eval"]
    m["long_eval_sliding"] = {"benign_fpr_block": rate((s_long >= blk)[lg.label.to_numpy() == 0]),
                              "recall_by_position": {pos: rate((s_long >= blk)[((lg.label == 1) & (lg.position == pos)).to_numpy()]) for pos in ["start", "middle", "end"]}}
    return m, sc
""")

md(r"""
## 5. The runs
For every seed: baseline, A, B and (optionally) the original-labels check. After each run the results and raw scores are written to disk.
The baseline model is also re-thresholded on the new val ("valB") as a diagnostic.
""")
code(r"""
results, scores = {}, {}
RES_PATH = f"{OUT}/hardneg_results.json"


def save_all():
    meta = {"seeds": SEEDS, "model": HF_MODEL, "lr": LR, "batch": BATCH, "head": HEAD, "tail": TAIL, "max_epochs": MAX_EPOCHS, "min_epochs": MIN_EPOCHS,
            "torch": torch.__version__, "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
            "rows": {k: len(v) for k, v in {**S, "tricky": tricky, "hardneg_train": hn_tr, "hardneg_val": hn_va}.items()},
            "set_hashes": {n: hashlib.sha1("\n".join(map(str, t)).encode()).hexdigest()[:12] for n, t in {**sets_for_scoring(), "long_eval": S["long_eval"].text}.items()},
            "dry_run": DRY_RUN}
    import transformers
    meta["transformers"] = transformers.__version__
    json.dump({"meta": meta, "results": results, "scores": scores}, open(RES_PATH, "w"), default=float)


def train_frames(arm):
    tiers = ARMS[arm]
    add_tr = hn_tr[hn_tr.tier.isin(tiers)][["text", "label"]]
    add_va = hn_va[hn_va.tier.isin(tiers)][["text", "label"]]
    tr = pd.concat([S["train_augmented"][["text", "label"]], add_tr], ignore_index=True)
    va = pd.concat([S["val"][["text", "label"]], add_va], ignore_index=True)
    return tr, va


def run(tag, arm, seed, label_col="label"):
    if label_col == "label":
        tr, va = train_frames(arm)
    else:                                                 # original-labels check: same rows, published labels
        tr = S["train_augmented"][["text", label_col]].rename(columns={label_col: "label"})
        va = S["val"][["text", label_col]].rename(columns={label_col: "label"})
    print(f"\n===== {tag}: {len(tr)} train rows ({int(tr.label.sum())} attacks), {len(va)} val rows =====")
    model, tok, history, best_epoch, secs = train_one(tag, seed, tr, va)
    vs = predict(model, tok, list(va.text))
    thr = thresholds(va.label.to_numpy(), vs)
    m, sc = evaluate(model, tok, thr, label_col)
    m.update({"arm": arm, "seed": seed, "label_col": label_col, "history": history, "best_epoch": best_epoch, "train_seconds": round(secs), "n_train": len(tr), "n_val": len(va)})
    # threshold-free test AUC under BOTH label versions (the labels differ on the rows flagged in the cleaning step)
    m["test_auc_cleaned_labels"] = float(roc_auc_score(S["test"].label, sc["test"]))
    m["test_auc_original_labels"] = float(roc_auc_score(S["test"].label_original, sc["test"]))
    if arm != "0_baseline" and label_col == "label":     # how the new val rows are flagged (diagnostic)
        sc_hn = predict(model, tok, list(hn_va.text))
        m["hardneg_val_flag_rates"] = {f"{t}/label{l}": {"n": int(len(g)), "flag_rate_block": float((sc_hn[g.index] >= thr["block"]).mean())} for (t, l), g in hn_va.groupby(["tier", "label"])}
    results[tag], scores[tag] = m, {k: [round(float(x), 5) for x in v] for k, v in sc.items()}
    f = m["test"]
    print(f"[{tag}] thr block {thr['block']:.3f} warn {thr['warn']:.3f} | test F1 {f['f1']:.3f} FPR {f['fpr']:.3f} R {f['recall']:.3f} | E1 FP {m['E1']['fp_block']['k']}/104 "
          f"E2 FP {m['E2']['fp_block']['k']}/100 E3 caught {m['E3']['caught_block']['k']}/24 | indirect R {m['indirect_test']['recall']['k']}/{m['indirect_test']['recall']['n']}")
    if arm == "0_baseline" and label_col == "label":     # diagnostic: same model, thresholds re-tuned on the NEW val (all three tiers)
        va_b = pd.concat([S["val"][["text", "label"]], hn_va[["text", "label"]]], ignore_index=True)
        thr_b = thresholds(va_b.label.to_numpy(), predict(model, tok, list(va_b.text)))
        mb, scb = evaluate(model, tok, thr_b)
        mb.update({"arm": "0_baseline@valB", "seed": seed, "label_col": "label", "note": "same model as 0_baseline, thresholds tuned on old val + all new val rows"})
        results[f"0_baseline@valB/seed{seed}"], scores[f"0_baseline@valB/seed{seed}"] = mb, {k: [round(float(x), 5) for x in v] for k, v in scb.items()}
    ckpt = f"{OUT}/ckpt/{tag.replace('/', '_')}"
    model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
    del model; torch.cuda.empty_cache() if DEV == "cuda" else None
    save_all()


for seed in SEEDS:
    for arm in ARMS:
        run(f"{arm}/seed{seed}", arm, seed)
    if RUN_ORIGINAL_LABELS_CHECK:
        run(f"orig_labels/seed{seed}", "0_baseline", seed, label_col="label_original")
print("\nall runs finished")
""")

md(r"""
## 6. Summary (mean and range over the seeds) and the pre-registered criteria
Criteria (docs/decisions.md Day 2.6, approved before any run), applied to the **retrained baseline's mean** (the pipeline here is fp16 PyTorch, so it is the like-for-like
"before"; the original int8 server numbers were E1 10/104 and E2 28/100):
1. E1 and E2 false positives each fall by >= 50% relative to the baseline mean. 2. Test F1 within -0.02 of the baseline (point estimate here; the paired bootstrap is run locally).
3. Recall on test, indirect_test and the tricky attack rows drops by <= 0.03. 4. E3 (Hinglish attacks) recall not below the baseline.
This table is a provisional read-out; the formal analysis (McNemar, paired bootstrap, per-seed detail) is done locally from the saved raw scores.
""")
code(r"""
def get(m, path):
    for p in path.split("."):
        m = m[p]
    return m

def frac(x):
    return x["k"] / x["n"]

METRICS = [("test F1", "test.f1", lambda v: v), ("test FPR", "test.fpr", lambda v: v), ("test recall", "test.recall", lambda v: v),
           ("indirect_test recall", "indirect_test.recall", frac), ("indirect_test FPR", "indirect_test.fpr", frac),
           ("tricky attack recall", "tricky.attack_recall", frac), ("xstest FP /250", "xstest_safe_fp", lambda x: x["k"]),
           ("E1 FP /104", "E1.fp_block", lambda x: x["k"]), ("E1 English-control FP /104", "E1.english_control_fp_block", lambda x: x["k"]),
           ("E2 FP /100", "E2.fp_block", lambda x: x["k"]), ("E3 caught /24", "E3.caught_block", lambda x: x["k"]),
           ("long benign FPR (sliding)", "long_eval_sliding.benign_fpr_block", frac),
           ("long middle recall (sliding)", "long_eval_sliding.recall_by_position.middle", frac), ("block threshold", "thresholds.block", lambda v: v)]
names = ["0_baseline", "0_baseline@valB", "A_negatives", "B_neg_plus_attacks", "orig_labels"]
rows = []
for label, path, f in METRICS:
    r = {"metric": label}
    for a in names:
        vals = [f(get(m, path)) for k, m in results.items() if k.split("/")[0] == a]
        r[a] = f"{np.mean(vals):.3f} [{min(vals):.3f}, {max(vals):.3f}]" if vals else "-"
    rows.append(r)
table = pd.DataFrame(rows).set_index("metric")
display(table)

def mean_of(a, path, f):
    return float(np.mean([f(get(m, path)) for k, m in results.items() if k.split("/")[0] == a]))

crit = []
for a in ["A_negatives", "B_neg_plus_attacks"]:
    b = lambda path, f: mean_of("0_baseline", path, f)
    x = lambda path, f: mean_of(a, path, f)
    kk = lambda v: v["k"]
    c1 = x("E1.fp_block", kk) <= 0.5 * b("E1.fp_block", kk) and x("E2.fp_block", kk) <= 0.5 * b("E2.fp_block", kk)
    c2 = x("test.f1", lambda v: v) >= b("test.f1", lambda v: v) - 0.02
    c3 = all(x(p, frac) >= b(p, frac) - 0.03 for p in ["indirect_test.recall", "tricky.attack_recall"]) and x("test.recall", lambda v: v) >= b("test.recall", lambda v: v) - 0.03
    c4 = x("E3.caught_block", kk) >= b("E3.caught_block", kk)
    crit.append({"arm": a, "1 FP halved (E1 and E2)": c1, "2 test F1 within -0.02": c2, "3 recalls within -0.03": c3, "4 E3 recall not below baseline": c4, "all four": c1 and c2 and c3 and c4})
crit = pd.DataFrame(crit).set_index("arm")
display(crit)
open("hardneg_summary.md", "w").write("# Hard-negative round, provisional summary (Colab)\n\nMean [min, max] over seeds.\n\n" + table.to_markdown() +
                                     "\n\n## Pre-registered criteria vs the retrained baseline mean (provisional)\n\n" + crit.to_markdown() + "\n")
save_all()
""")

md("## 7. Download the results (and, optionally, one checkpoint)")
code(r"""
import shutil
if not DRY_RUN:
    from google.colab import files
    shutil.copy(RES_PATH, "hardneg_results.json")
    files.download("hardneg_results.json"); files.download("hardneg_summary.md")
""")
code(r"""
# OPTIONAL: after you have looked at the summary, zip the checkpoint(s) you may want to export to ONNX (each is about 90 MB). The runtime and its disk disappear when the session ends.
ZIP_TAGS = []            # e.g. ["B_neg_plus_attacks/seed42"]
if ZIP_TAGS and not DRY_RUN:
    for t in ZIP_TAGS:
        name = t.replace("/", "_")
        shutil.make_archive(f"/content/{name}", "zip", f"{OUT}/ckpt/{name}")
        files.download(f"/content/{name}.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, len(cells), "cells")
