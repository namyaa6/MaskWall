"""Benchmark the exported ONNX models: size, CPU latency, and (real checkpoints only) accuracy. fp32 vs int8.

Latency protocol (fixed before any number was seen; 2 intra-op threads to mimic a 2-vCPU Hugging Face Space):
  * onnxruntime CPUExecutionProvider, intra_op_num_threads=2, inter_op_num_threads=1, all graph optimisations
  * batch size 1; 20 warm-up runs; then 300 timed runs, each on a different random val prompt (seed 42) at its NATURAL
    head+tail length (the number the 50 ms selection rule uses: p50 over these 300)
  * plus a worst case: 100 timed runs on a full 512-token input
  * only session.run is timed (tokenization excluded; it is measured separately in the server step)
Accuracy (real checkpoints only): scores = softmax(logits)[1] at batch size 1 for val, test, indirect_val, indirect_test,
tricky, xstest_safe. The threshold is re-tuned on val F1 for every model/precision (an int8 model is a different scorer);
results at the frozen fp32 threshold from the Colab run are reported too, to show how much quantization shifts the scores.
Stand-ins (random heads) get size and latency only; their accuracy is meaningless and is not computed.

Usage (repo root):
    python ml/benchmark.py --standin minilm deberta_v3_small
    python ml/benchmark.py --all-real          # or --real cleaned/minilm ...
Results are merged into ml/results/benchmark.json.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer

import export_quantize as eq
import train_baseline as tb
from headtail import encode_head_tail

ROOT = eq.ROOT
PROC = ROOT / "ml" / "data" / "processed"
RESULTS = eq.RESULTS
THREADS, WARMUP, RUNS, RUNS_512, SEED = 2, 20, 300, 100, 42
SETS = ["val", "test", "indirect_val", "indirect_test", "xstest_safe"]


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def timeit(sess, seqs, warmup, runs):
    for s in seqs[:warmup] if len(seqs) >= warmup else [seqs[0]] * warmup:
        eq.run_ort(sess, s)
    ts = []
    for i in range(runs):
        s = seqs[i % len(seqs)]
        ids = np.array([s], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        t = time.perf_counter()
        sess.run(["logits"], feed)
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "mean_ms": float(np.mean(ts)), "n": runs}


def latency(sess, tok):
    texts = pd.read_parquet(PROC / "val.parquet")["text"].sample(n=RUNS, random_state=SEED).tolist()
    natural = encode_head_tail(tok, texts)
    rng = np.random.default_rng(SEED)
    ids = rng.integers(1000, 20000, 510).tolist()
    full = [[tok.cls_token_id] + ids + [tok.sep_token_id]]
    res = {"natural": timeit(sess, natural, WARMUP, RUNS), "len512": timeit(sess, full, 5, RUNS_512),
           "natural_tokens_median": float(np.median([len(s) for s in natural]))}
    return res


def scores(sess, tok, texts):
    enc = encode_head_tail(tok, texts)
    return np.array([eq.softmax_p1(eq.run_ort(sess, s)) for s in enc])


def load_sets():
    d = {n: pd.read_parquet(PROC / f"{n}.parquet") for n in SETS}
    d["tricky"] = pd.read_csv(ROOT / "ml" / "data" / "tricky.csv")
    return d


def accuracy(sess, tok, sets, frozen_thr):
    sc = {n: scores(sess, tok, d["text"].tolist()) for n, d in sets.items()}
    yv = sets["val"]["label"].to_numpy()
    thr, val_f1 = tb.tune_f1_threshold(yv, sc["val"])
    thr_fpr = tb.tune_fpr_threshold(yv, sc["val"])

    def at(t):
        out = {"val": tb.rates(yv, sc["val"] >= t)}
        for n in ("indirect_val", "indirect_test"):
            r = tb.rates(sets[n]["label"].to_numpy(), sc[n] >= t)
            out[n] = {"recall": r["recall"], "fpr": r["fpr"], "f1": r["f1"], "confusion_matrix": r["confusion_matrix"]}
        out["test"] = tb.evaluate_split(sets["test"], sc["test"], t, thr_fpr)
        out["tricky"] = tb.evaluate_tricky(sets["tricky"], sc["tricky"], t)
        out["xstest_fp"] = int((sc["xstest_safe"] >= t).sum())
        return out

    res = {"threshold_val_tuned": thr, "val_f1": val_f1, "val_pr_auc": float(tb.average_precision_score(yv, sc["val"])),
           "at_val_threshold": at(thr)}
    if frozen_thr is not None:
        res["frozen_fp32_threshold"] = frozen_thr
        res["at_frozen_threshold"] = at(frozen_thr)
    return res


def run_job(job, standin, sets, frozen, smoke=False):
    out = eq.ONNX_DIR / (f"_standin/{job}" if standin else job)
    fp32, int8 = out / "fp32.onnx", out / "int8.onnx"
    if not (fp32.exists() and int8.exists()):
        return {"job": job, "standin": standin, "status": "onnx files missing (run export_quantize.py first)"}
    tok = AutoTokenizer.from_pretrained(eq.HF_IDS[job] if standin else str(eq.CKPT / job))
    res = {"job": job, "standin": standin, "status": "ok"}
    for name, path in [("fp32", fp32), ("int8", int8)]:
        s = session(path)
        entry = {"size_mb": eq.mb(path), "latency": latency(s, tok)}
        if not standin or smoke:
            entry["accuracy"] = accuracy(s, tok, sets, frozen.get(job))
        res[name] = entry
        lat = entry["latency"]
        print(f"[{job}{' (stand-in)' if standin else ''}] {name}: {entry['size_mb']} MB, natural p50 {lat['natural']['p50_ms']:.1f} ms "
              f"(p95 {lat['natural']['p95_ms']:.1f}), 512 tok p50 {lat['len512']['p50_ms']:.1f} ms"
              + ("" if standin and not smoke else f", val F1 {entry['accuracy']['val_f1']:.3f}"), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standin", nargs="*", default=[], choices=list(eq.HF_IDS))
    ap.add_argument("--real", nargs="*", default=[], choices=eq.REAL_JOBS)
    ap.add_argument("--all-real", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="code-path test only: run the accuracy code on a stand-in over small subsets; writes ml/cache/bench_smoke.json, never results/")
    args = ap.parse_args()
    jobs = [(j, True) for j in args.standin] + [(j, False) for j in (eq.REAL_JOBS if args.all_real else args.real)]
    assert jobs, "nothing to do"
    mt = RESULTS / "metrics_transformers.json"
    frozen = {}
    if mt.exists():
        for v, ms in json.loads(mt.read_text())["transformers"].items():
            for m, r in ms.items():
                frozen[f"{v}/{m}"] = r["threshold"]
    sets = load_sets() if (args.smoke or any(not s for _, s in jobs)) else None
    if args.smoke:  # tiny subsets: this run exercises code paths, its numbers mean nothing
        sets = {n: (d.groupby("label").head(20) if n in ("val", "test", "indirect_val", "indirect_test") else d.head(40) if n == "xstest_safe" else d)
                for n, d in sets.items()}
    path = (ROOT / "ml" / "cache" / "bench_smoke.json") if args.smoke else RESULTS / "benchmark.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["env"] = {"cpu": platform.processor(), "cores": os.cpu_count(), "onnxruntime": ort.__version__, "threads": THREADS,
                     "python": platform.python_version(), "protocol": {"warmup": WARMUP, "runs": RUNS, "runs_512": RUNS_512, "batch": 1}}
    for job, standin in jobs:
        key = ("standin:" if standin else "") + job
        report[key] = run_job(job, standin, sets, frozen, args.smoke)
        path.write_text(json.dumps(report, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
