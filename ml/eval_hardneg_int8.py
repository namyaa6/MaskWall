"""Export the retrained arm-B checkpoint to int8 and re-evaluate it against the CURRENTLY SERVED MiniLM on every evaluation set, with the
pre-registered criteria (docs/decisions.md Day 2.6/2.8/2.9) applied to the int8 models. Reporting + a verdict; it does NOT replace anything.

    python ml/eval_hardneg_int8.py                       # candidate = models/ckpt/hardneg_B_seed42 (or a hardneg_B_seed42.zip next to it, unpacked first)
    python ml/eval_hardneg_int8.py --self-test           # candidate = the served model itself (pipeline check: must reproduce hinglish_baseline.json)

Steps: (1) ONNX fp32 export with the parity gate + int8 (ml/export_quantize.py, same code as the served model); (2) package a candidate folder
(models/onnx/<job>/serve/: model.int8.onnx, tokenizer.json, model_meta.json) with thresholds tuned on the arm's val ONLY, scored with the int8 model:
old val + the new val slice (T1/T2/T3), block = max(thr_F1, thr_FPR<=1%), warn = min (the server rule); (3) score every evaluation set with the SERVING
scanner (backend/app/scanners/injection.py): head_tail for prompts, sliding windows (cap 6) for the long documents, for the candidate and for backend/model/minilm;
(4) criteria vs the served int8 model, paired McNemar per outcome and a paired bootstrap of the test-F1 difference.
Single seed: the earlier 3-seed round showed baseline seeds differ a lot (E1 false positives 9 to 28), so differences are read with that in mind.
Output: ml/results/hardneg_int8_eval.json and .md"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))
import benchmark as bm  # noqa: E402
import export_quantize as eq  # noqa: E402
import train_baseline as tb  # noqa: E402
from app.scanners.injection import InjectionScanner  # noqa: E402

PROC = ROOT / "ml" / "data" / "processed"
RES = ROOT / "ml" / "results"
SERVED = ROOT / "backend" / "model" / "minilm"
TIERS = ["T1", "T2", "T3"]
rng = np.random.default_rng(0)


def prepare_checkpoint(name):
    d = eq.CKPT / name
    z = eq.CKPT / f"{name}.zip"
    if not d.exists() and z.exists():
        d.mkdir(parents=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(d)                  # zipfile strips absolute paths and '..' components
        print(f"unpacked {z.name} -> {d}")
    if not (d / "model.safetensors").exists() and (d / name / "model.safetensors").exists():
        d = d / name                         # the zip was extracted one folder deeper
    assert (d / "model.safetensors").exists() and (d / "config.json").exists(), (
        f"{d} must contain model.safetensors, config.json and the tokenizer files (the Colab notebook's zip)")
    return d


def package_candidate(job, ckpt):
    """int8 export (parity gate included) + thresholds tuned on the arm's val with the int8 scores."""
    out = eq.ONNX_DIR / job
    if not (out / "int8.onnx").exists():
        # run_job reads eq.CKPT / job, so point it at the real folder
        rep = eq.run_job(job, False)
        assert rep.get("status") == "ok", f"export failed: {rep}"
        (out / "export_report.json").write_text(json.dumps(rep, indent=2))
    rep = json.loads((out / "export_report.json").read_text()) if (out / "export_report.json").exists() else {}
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    sess = bm.session(out / "int8.onnx")
    val = pd.read_parquet(PROC / "val.parquet")
    hv = pd.read_csv(ROOT / "ml" / "data" / "hardneg" / "hardneg_val.csv")
    va = pd.concat([val[["text", "label"]], hv[hv.tier.isin(TIERS)][["text", "label"]]], ignore_index=True)
    print(f"tuning thresholds on {len(va)} val rows (old val + new slice) with the int8 model ...", flush=True)
    sc = bm.scores(sess, tok, va.text.tolist())
    y = va.label.to_numpy()
    thr_f1, val_f1 = tb.tune_f1_threshold(y, sc)
    thr_fpr = tb.tune_fpr_threshold(y, sc)
    serve = out / "serve"
    serve.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(out / "int8.onnx", serve / "model.int8.onnx")
    shutil.copy2(ckpt / "tokenizer.json", serve / "tokenizer.json")
    meta = {"name": "minilm_hardneg_B", "source": job, "onnx_file": "model.int8.onnx", "tokenizer_file": "tokenizer.json", "cls_id": int(tok.cls_token_id), "sep_id": int(tok.sep_token_id),
            "max_content_tokens": 510, "head": 256, "tail": 254, "stride": 255, "thr_f1": thr_f1, "thr_fpr1pct": thr_fpr, "val_f1": val_f1, "n_val": int(len(va)),
            "val_definition": "old val + new val slice T1/T2/T3 (arm B protocol)"}
    (serve / "model_meta.json").write_text(json.dumps(meta, indent=2))
    return serve, meta, rep


def load_sets():
    P = lambda n: pd.read_parquet(PROC / f"{n}.parquet")
    return {"test": P("test"), "indirect_test": P("indirect_test"), "long_eval": P("long_eval"), "xstest_safe": P("xstest_safe"), "E1": P("hinglish_human_benign"),
            "E2": P("prompt_talk_benign_ai"), "E3": P("hinglish_attacks_ai"), "E4": P("manual7"), "tricky": pd.read_csv(ROOT / "ml" / "data" / "tricky.csv")}


def score_all(model_dir, sets):
    ht = InjectionScanner.load(model_dir, mode="head_tail", threads=2)
    sl = InjectionScanner.load(model_dir, mode="sliding", threads=2, max_windows=6)
    f = lambda sc, texts: np.array([sc.scan(t).score for t in texts])
    s = {n: f(ht, sets[n].text) for n in ("test", "indirect_test", "xstest_safe", "E2", "E3", "E4", "tricky")}
    s["E1"], s["E1_control"] = f(ht, sets["E1"].text), f(ht, sets["E1"].en_control)
    s["long_eval"] = f(sl, sets["long_eval"].text)
    return s, max(ht.thr_f1, ht.thr_fpr1pct), min(ht.thr_f1, ht.thr_fpr1pct)


def outcomes(sets):
    t, ie, lg, tk = sets["test"], sets["indirect_test"], sets["long_eval"], sets["tricky"]
    return {"E1 human Hinglish benign (FP)": ("E1", np.ones(len(sets["E1"]), bool), "fp"), "E1 English control (FP)": ("E1_control", np.ones(len(sets["E1"]), bool), "fp"),
            "E2 prompt-talk benign (FP)": ("E2", np.ones(len(sets["E2"]), bool), "fp"), "E3 Hinglish attacks (caught)": ("E3", np.ones(len(sets["E3"]), bool), "hit"),
            "test attacks (recall)": ("test", t.label.to_numpy() == 1, "hit"), "test benign (FP)": ("test", t.label.to_numpy() == 0, "fp"),
            "indirect_test attacks (recall)": ("indirect_test", ie.label.to_numpy() == 1, "hit"), "indirect_test benign (FP)": ("indirect_test", ie.label.to_numpy() == 0, "fp"),
            "tricky attacks (recall)": ("tricky", tk.label.to_numpy() == 1, "hit"), "tricky benign (FP)": ("tricky", tk.label.to_numpy() == 0, "fp"),
            "xstest_safe (FP)": ("xstest_safe", np.ones(len(sets["xstest_safe"]), bool), "fp"),
            "long benign docs (FP, sliding)": ("long_eval", lg.label.to_numpy() == 0, "fp"),
            "long middle injections (recall, sliding)": ("long_eval", ((lg.label == 1) & (lg.position == "middle")).to_numpy(), "hit")}


def f1(y, p):
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="hardneg_B_seed42", help="folder (or .zip) name under models/ckpt/")
    ap.add_argument("--self-test", action="store_true", help="candidate = the served model (checks the pipeline; no export)")
    args = ap.parse_args()
    sets = load_sets()
    if args.self_test:
        cand_dir, cand_meta, rep, label = SERVED, json.loads((SERVED / "model_meta.json").read_text()), {}, "served (self-test)"
    else:
        ckpt = prepare_checkpoint(args.ckpt)
        cand_dir, cand_meta, rep = package_candidate(args.ckpt, ckpt)
        label = "candidate arm B (int8)"
    print("scoring the served model ...", flush=True)
    sb, blk_b, wrn_b = score_all(SERVED, sets)
    print("scoring the candidate ...", flush=True)
    sc, blk_c, wrn_c = score_all(cand_dir, sets)
    outs = outcomes(sets)

    table, flags_b, flags_c = {}, {}, {}
    for name, (key, mask, kind) in outs.items():
        a, b = (sb[key] >= blk_b)[mask], (sc[key] >= blk_c)[mask]
        only_s, only_c = int((a & ~b).sum()), int((~a & b).sum())
        n = only_s + only_c
        table[name] = {"n": int(mask.sum()), "served": int(a.sum()), "candidate": int(b.sum()), "only_served": only_s, "only_candidate": only_c,
                       "p": 1.0 if n == 0 else float(binomtest(only_s, n, 0.5).pvalue), "kind": kind}
    y = sets["test"].label.to_numpy() == 1
    ps, pc = sb["test"] >= blk_b, sc["test"] >= blk_c
    idx = rng.integers(0, len(y), (5000, len(y)))
    d = np.array([f1(y[i], pc[i]) - f1(y[i], ps[i]) for i in idx])
    f1s, f1c = f1(y, ps), f1(y, pc)
    test_f1 = {"served": f1s, "candidate": f1c, "diff": f1c - f1s, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    # ---- pre-registered criteria, applied to the int8 models (the served int8 model is the 'before')
    E1, E2, E3 = table["E1 human Hinglish benign (FP)"], table["E2 prompt-talk benign (FP)"], table["E3 Hinglish attacks (caught)"]
    rec = lambda k: table[k]["candidate"] / table[k]["n"] - table[k]["served"] / table[k]["n"]
    crit = {
        "1 FP halved on E1 and on E2 (candidate <= half of served)": E1["candidate"] <= 0.5 * E1["served"] and E2["candidate"] <= 0.5 * E2["served"],
        "2 test F1 not below served by more than 0.02": test_f1["diff"] >= -0.02,
        "3 recall drop <= 0.03 on test, indirect_test, tricky attacks": all(rec(k) >= -0.03 for k in ("test attacks (recall)", "indirect_test attacks (recall)", "tricky attacks (recall)")),
        "4 E3 Hinglish-attack recall not below served": E3["candidate"] >= E3["served"],
    }
    verdict = all(crit.values())

    # ---- reproduction check against the Colab run of arm B seed 42 (same data; fp16 PyTorch vs int8 here)
    colab = None
    cj = ROOT / "ml" / "results" / "hardneg_results.json"
    if not args.self_test and cj.exists():
        R = json.loads(cj.read_text())["results"]["B_neg_plus_attacks/seed42"]
        colab = {"E1": R["E1"]["fp_block"]["k"], "E2": R["E2"]["fp_block"]["k"], "E3": R["E3"]["caught_block"]["k"], "test_f1": R["test"]["f1"], "block": R["thresholds"]["block"]}

    out = {"label": label, "thresholds": {"served": {"block": blk_b, "warn": wrn_b}, "candidate": {"block": blk_c, "warn": wrn_c}}, "candidate_meta": cand_meta, "export_report": rep,
           "outcomes": table, "test_f1": test_f1, "criteria": crit, "all_pass": verdict, "colab_original_run": colab,
           "E4": [{"text": t[:60], "label": int(l), "served": float(a), "candidate": float(b)} for t, l, a, b in zip(sets["E4"].text, sets["E4"].label, sb["E4"], sc["E4"])]}
    (RES / ("hardneg_int8_selftest.json" if args.self_test else "hardneg_int8_eval.json")).write_text(json.dumps(out, indent=1, default=float))

    L = [f"# Hard-negative arm B in int8 vs the served MiniLM (generated by ml/eval_hardneg_int8.py{' --self-test' if args.self_test else ''})", "",
         f"Block thresholds: served {blk_b:.3f}, candidate {blk_c:.3f} (candidate tuned on old val + new val slice with the int8 model; served: old val). "
         "Both scored with the serving scanner: head_tail for prompts, sliding windows (cap 6) for the long documents. Single seed.", "",
         "| outcome | n | served k | candidate k | only served / only candidate | McNemar p |", "|---|---|---|---|---|---|"]
    for k, v in table.items():
        L.append(f"| {k} | {v['n']} | {v['served']} | {v['candidate']} | {v['only_served']} / {v['only_candidate']} | {v['p']:.3f} |")
    L += ["", f"Test F1: served {f1s:.3f}, candidate {f1c:.3f}, diff {test_f1['diff']:+.3f} (paired bootstrap 95% CI [{test_f1['ci95'][0]:+.3f}, {test_f1['ci95'][1]:+.3f}])", "",
          "## Pre-registered criteria (int8 vs served int8)", ""] + [f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in crit.items()]
    L += ["", f"**Verdict: {'all four criteria pass in int8' if verdict else 'at least one criterion fails in int8'}.**"]
    if colab:
        L += ["", f"Colab fp16 original run (arm B seed 42): E1 {colab['E1']}, E2 {colab['E2']}, E3 {colab['E3']}/24, test F1 {colab['test_f1']:.3f}, block {colab['block']:.3f}."]
    L += ["", "## The owner's 7 prompts (E4, never used for any choice)", "", "| label | prompt | served | candidate |", "|---|---|---|---|"]
    for e in out["E4"]:
        L.append(f"| {e['label']} | {e['text']} | {e['served']:.3f} | {e['candidate']:.3f} |")
    (RES / ("hardneg_int8_selftest.md" if args.self_test else "hardneg_int8_eval.md")).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
