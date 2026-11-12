"""ONNX export + int8 quantization of the fine-tuned classifiers, with a numerical parity gate.

For every job (a fine-tuned checkpoint, or a pretrained stand-in of the same architecture):
  1. export fp32 ONNX (inputs input_ids, attention_mask; dynamic batch and sequence; output logits)
  2. PARITY GATE: PyTorch vs ONNX-fp32 logits on 200 val prompts at batch size 1 (their natural head+tail lengths);
     max |difference| must be <= 1e-3, otherwise the job stops (an export bug, not a modelling result)
  3. int8: onnxruntime dynamic quantization (QInt8 weights, per-channel, Gather/embeddings included); then an
     agreement check fp32 vs int8 (score correlation and decision agreement at 0.5)
Files go to models/onnx/<job>/{fp32,int8}.onnx (gitignored). A summary is merged into ml/results/export_report.json.

Jobs
  real      models/ckpt/<variant>/<model>   (variant: cleaned | cleaned_aug; model: minilm | deberta_v3_small)
  stand-in  --standin minilm|deberta_v3_small: pretrained base model with a RANDOM classification head. Size, latency and
            exportability depend on the architecture, not on fine-tuned weights, so these are valid for planning;
            accuracy of a stand-in is meaningless and is never reported as a result.

Usage (repo root):
    python ml/export_quantize.py --standin minilm deberta_v3_small
    python ml/export_quantize.py --real cleaned/minilm cleaned_aug/deberta_v3_small     # or --all-real
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from headtail import encode_head_tail

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "ckpt"
ONNX_DIR = ROOT / "models" / "onnx"
RESULTS = ROOT / "ml" / "results"
VAL = ROOT / "ml" / "data" / "processed" / "val.parquet"
HF_IDS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "deberta_v3_small": "microsoft/deberta-v3-small"}
REAL_JOBS = [f"{v}/{m}" for v in ("cleaned", "cleaned_aug") for m in ("minilm", "deberta_v3_small")]
PARITY_N, PARITY_TOL = 200, 1e-3
OPSET = 17
SEED = 42


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def load(job, standin):
    if standin:
        torch.manual_seed(SEED)
        src = HF_IDS[job]
        model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=2)
    else:
        src = str(CKPT / job)
        model = AutoModelForSequenceClassification.from_pretrained(src)
    model = model.float().eval()  # fp32 (the DeBERTa fp16 gradient-scaler crash came from fp16 weights)
    return model, AutoTokenizer.from_pretrained(src if not standin else HF_IDS[job])


def export_fp32(model, tok, path, dynamo=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = torch.tensor([[tok.cls_token_id, 1000, 2000, 3000, tok.sep_token_id]] * 2)
    att = torch.ones_like(ids)
    kwargs = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"],
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
                  opset_version=OPSET)
    # The wrapper must be in eval mode: torch.onnx.export restores the module's ORIGINAL train/eval flag afterwards, and
    # a fresh wrapper is in train mode, which would silently switch the wrapped model back to train mode (dropout on)
    # and break the PyTorch reference in the parity check.
    with torch.no_grad():
        torch.onnx.export(LogitsOnly(model).eval(), (ids, att), str(path), dynamo=dynamo, **kwargs)
    model.eval()
    onnx.checker.check_model(str(path))  # path form: works above the 2 GB protobuf limit too


def session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def val_encodings(tok, n=PARITY_N):
    texts = pd.read_parquet(VAL)["text"].sample(n=n, random_state=SEED).tolist()
    return encode_head_tail(tok, texts)


def run_ort(sess, seq):
    ids = np.array([seq], dtype=np.int64)
    return sess.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]


def parity(model, sess, enc):
    diffs = []
    with torch.no_grad():
        for s in enc:
            t = torch.tensor([s])
            ref = model(input_ids=t, attention_mask=torch.ones_like(t)).logits[0].numpy()
            diffs.append(np.abs(ref - run_ort(sess, s)).max())
    return float(np.max(diffs)), float(np.mean(diffs))


def softmax_p1(lg):
    e = np.exp(lg - lg.max())
    return float(e[1] / e.sum())


def quantize(fp32, int8):
    pre = fp32.with_name("fp32_pre.onnx")
    try:
        quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)  # shape inference + constant folding for quantization
        src, note = pre, "pre-processed"
    except Exception as e:  # pre-processing can fail on exotic graphs; quantizing the raw graph still works
        src, note = fp32, f"pre-process skipped ({type(e).__name__})"
    quantize_dynamic(str(src), str(int8), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gather"])
    if pre.exists():
        pre.unlink()
    return note


def mb(path):
    return round(path.stat().st_size / 1e6, 1)


def run_job(job, standin):
    out = ONNX_DIR / (f"_standin/{job}" if standin else job)
    rep = {"job": job, "standin": standin, "dir": str(out.relative_to(ROOT))}
    t0 = time.time()
    model, tok = load(job, standin)
    rep["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    fp32, int8 = out / "fp32.onnx", out / "int8.onnx"

    for dynamo in (False, True):  # legacy exporter first (more robust for these encoders), dynamo as the fallback
        try:
            export_fp32(model, tok, fp32, dynamo=dynamo)
            rep["exporter"] = "dynamo" if dynamo else "torchscript"
            break
        except Exception as e:
            rep.setdefault("export_errors", []).append(f"{'dynamo' if dynamo else 'torchscript'}: {type(e).__name__}: {str(e)[:300]}")
            print(f"[{job}] export with dynamo={dynamo} failed: {type(e).__name__}: {str(e)[:200]}")
    else:
        rep["status"] = "export_failed"
        return rep
    rep["fp32_mb"] = mb(fp32)

    enc = val_encodings(tok)
    sess32 = session(fp32)
    mx, mean = parity(model, sess32, enc)
    rep["parity_fp32"] = {"n": len(enc), "max_abs_logit_diff": mx, "mean_abs_logit_diff": mean, "tolerance": PARITY_TOL}
    print(f"[{job}] fp32 parity: max |diff| {mx:.2e} (tol {PARITY_TOL:.0e})")
    if mx > PARITY_TOL:
        rep["status"] = "parity_failed"
        return rep

    rep["quantize_note"] = quantize(fp32, int8)
    rep["int8_mb"] = mb(int8)
    sess8 = session(int8)
    p32 = np.array([softmax_p1(run_ort(sess32, s)) for s in enc])
    p8 = np.array([softmax_p1(run_ort(sess8, s)) for s in enc])
    rep["int8_vs_fp32"] = {"n": len(enc), "score_pearson": float(np.corrcoef(p32, p8)[0, 1]),
                           "decision_agreement_at_0.5": float(((p32 >= 0.5) == (p8 >= 0.5)).mean()),
                           "max_abs_score_diff": float(np.abs(p32 - p8).max())}
    print(f"[{job}] fp32 {rep['fp32_mb']} MB -> int8 {rep['int8_mb']} MB; int8 vs fp32 score r={rep['int8_vs_fp32']['score_pearson']:.4f}, "
          f"agreement {rep['int8_vs_fp32']['decision_agreement_at_0.5']:.3f}")
    rep["status"] = "ok"
    rep["seconds"] = round(time.time() - t0)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standin", nargs="*", default=[], choices=list(HF_IDS))
    ap.add_argument("--real", nargs="*", default=[], choices=REAL_JOBS)
    ap.add_argument("--all-real", action="store_true")
    args = ap.parse_args()
    jobs = [(j, True) for j in args.standin] + [(j, False) for j in (REAL_JOBS if args.all_real else args.real)]
    assert jobs, "nothing to do"
    path = RESULTS / "export_report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    for job, standin in jobs:
        if not standin and not (CKPT / job).exists():
            print(f"[{job}] checkpoint missing at {CKPT / job}; skipped")
            continue
        key = ("standin:" if standin else "") + job
        try:
            report[key] = run_job(job, standin)
        except Exception as e:
            report[key] = {"job": job, "standin": standin, "status": f"error: {type(e).__name__}: {str(e)[:300]}"}
            print(f"[{job}] ERROR {type(e).__name__}: {str(e)[:300]}")
        path.write_text(json.dumps(report, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
