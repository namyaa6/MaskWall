"""Paired analysis of the hard-negative round, from the saved raw scores (ml/results/hardneg_results.json). No model is run.

For every seed, baseline vs arm are compared on the SAME rows at each model's own frozen block threshold:
  * exact McNemar (two-sided binomial on discordant pairs) for count outcomes (FP on E1/E2/xstest/long, recall on attack sets);
  * paired bootstrap (rows resampled jointly, 5000 draws) for the difference in test F1.
Seeds are not independent samples (same data, different init), so per-seed p-values are listed, never multiplied together.
Run: .venv/Scripts/python ml/hardneg_paired_analysis.py   (writes ml/results/hardneg_paired.json and .md)"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RES = json.load(open(ROOT / "ml/results/hardneg_results.json"))
R, SC = RES["results"], RES["scores"]
P = lambda n: pd.read_parquet(ROOT / f"ml/data/processed/{n}.parquet")
test, ind, lng = P("test"), P("indirect_test"), P("long_eval")
tricky = pd.read_csv(ROOT / "ml/data/tricky.csv")
SEEDS = RES["meta"]["seeds"]
ARMS = ["A_negatives", "B_neg_plus_attacks", "0_baseline@valB", "orig_labels"]
rng = np.random.default_rng(0)

# outcome sets: name -> (scores key, mask of rows included, what 'bad' is: 1 = flagged is bad (benign set), 0 = missed is bad (attack set))
SETS = {
    "E1 human Hinglish benign (FP)": ("E1_hinglish", np.ones(104, bool), "fp"),
    "E1 English control (FP)": ("E1_english_control", np.ones(104, bool), "fp"),
    "E2 prompt-talk benign (FP)": ("E2_prompt_talk", np.ones(100, bool), "fp"),
    "E3 Hinglish attacks (caught)": ("E3_hinglish_attacks", np.ones(24, bool), "hit"),
    "test attacks (recall)": ("test", test.label.to_numpy() == 1, "hit"),
    "test benign (FP)": ("test", test.label.to_numpy() == 0, "fp"),
    "indirect_test attacks (recall)": ("indirect_test", ind.label.to_numpy() == 1, "hit"),
    "indirect_test benign (FP)": ("indirect_test", ind.label.to_numpy() == 0, "fp"),
    "tricky attacks (recall)": ("tricky", tricky.label.to_numpy() == 1, "hit"),
    "tricky benign (FP)": ("tricky", tricky.label.to_numpy() == 0, "fp"),
    "xstest_safe (FP)": ("xstest_safe", np.ones(250, bool), "fp"),
    "long benign docs (FP, sliding)": ("long_eval", lng.label.to_numpy() == 0, "fp"),
    "long middle injections (recall, sliding)": ("long_eval", ((lng.label == 1) & (lng.position == "middle")).to_numpy(), "hit"),
}


def flags(tag, key):
    return np.asarray(SC[tag][key]) >= R[tag]["thresholds"]["block"]


def mcnemar(a, b):
    """a, b: boolean flag arrays on the same rows. b01 = flagged only by arm b, b10 = only by a. Exact two-sided p."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    return only_a, only_b, (1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue))


def f1(y, p):
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


out = {"per_set": {}, "test_f1": {}, "thresholds": {}}
lines = []
for arm in ARMS:
    out["per_set"][arm] = {}
    for sname, (key, mask, kind) in SETS.items():
        rows = []
        for s in SEEDS:
            base = flags(f"0_baseline/seed{s}", key)[mask]
            other = flags(f"{arm}/seed{s}", key)[mask]
            ob, oo, p = mcnemar(base, other)       # ob: flagged by baseline only; oo: by arm only
            rows.append({"seed": s, "baseline": int(base.sum()), "arm": int(other.sum()), "n": int(mask.sum()), "only_baseline": ob, "only_arm": oo, "p": p})
        out["per_set"][arm][sname] = rows
    # paired bootstrap for test F1 difference (arm - baseline), per seed
    y = test.label.to_numpy() == 1
    n = len(y)
    out["test_f1"][arm] = []
    for s in SEEDS:
        pb, pa = flags(f"0_baseline/seed{s}", "test"), flags(f"{arm}/seed{s}", "test")
        idx = rng.integers(0, n, (5000, n))
        d = np.array([f1(y[i], pa[i]) - f1(y[i], pb[i]) for i in idx])
        out["test_f1"][arm].append({"seed": s, "f1_baseline": f1(y, pb), "f1_arm": f1(y, pa), "diff": f1(y, pa) - f1(y, pb),
                                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]})
    out["thresholds"][arm] = [R[f"{arm}/seed{s}"]["thresholds"]["block"] for s in SEEDS]
out["thresholds"]["0_baseline"] = [R[f"0_baseline/seed{s}"]["thresholds"]["block"] for s in SEEDS]

# long-document FPR, mean over seeds (the number quoted in the verdict)
for a in ["0_baseline", "A_negatives", "B_neg_plus_attacks"]:
    ks = [R[f"{a}/seed{s}"]["long_eval_sliding"]["benign_fpr_block"] for s in SEEDS]
    out.setdefault("long_fpr", {})[a] = {"per_seed_k": [k["k"] for k in ks], "n": ks[0]["n"], "mean": float(np.mean([k["k"] / k["n"] for k in ks]))}

# ---------------------------------------------------------------- markdown
md = ["# Hard-negative round: paired analysis (local, from the saved raw scores)", "",
      "Per seed, same rows, each model at its own frozen block threshold. `only baseline` / `only arm` = discordant rows (flagged by one model only); p = exact McNemar (two-sided). "
      "Seeds are not independent, so p-values are per seed and not combined.", ""]
for arm in ["B_neg_plus_attacks", "A_negatives", "0_baseline@valB", "orig_labels"]:
    md += [f"## {arm} vs 0_baseline", "", "| outcome | n | baseline k (3 seeds) | arm k (3 seeds) | discordant only-baseline / only-arm (3 seeds) | McNemar p (3 seeds) |", "|---|---|---|---|---|---|"]
    for sname, rows in out["per_set"][arm].items():
        md.append(f"| {sname} | {rows[0]['n']} | {' / '.join(str(r['baseline']) for r in rows)} | {' / '.join(str(r['arm']) for r in rows)} | "
                  f"{' / '.join(f'{r['only_baseline']}-{r['only_arm']}' for r in rows)} | {' / '.join(f'{r['p']:.3f}' for r in rows)} |")
    md += ["", "Test F1 (point estimate, paired bootstrap 95% CI of arm minus baseline), per seed:", ""]
    for r in out["test_f1"][arm]:
        md.append(f"- seed {r['seed']}: baseline {r['f1_baseline']:.3f}, arm {r['f1_arm']:.3f}, diff {r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    md += [f"- block thresholds (baseline -> arm, 3 seeds): {', '.join(f'{b:.3f}' for b in out['thresholds']['0_baseline'])} -> {', '.join(f'{t:.3f}' for t in out['thresholds'][arm])}", ""]
md += ["## Long benign documents (sliding), false positives out of 114", ""]
for a, v in out["long_fpr"].items():
    md.append(f"- {a}: {v['per_seed_k']} of {v['n']} (mean FPR {v['mean']:.3f})")
(ROOT / "ml/results/hardneg_paired.md").write_text("\n".join(md) + "\n", encoding="utf-8")
json.dump(out, open(ROOT / "ml/results/hardneg_paired.json", "w"), indent=1, default=float)
print("\n".join(md))
