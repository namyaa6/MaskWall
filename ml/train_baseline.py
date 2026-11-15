"""Baseline injection classifiers (scikit-learn), all on the same splits.

Models:
  length_lr    log(chars), log(words) -> LR           (floor: how far does length alone get you?)
  word_tfidf   word 1-2 gram TF-IDF -> LR
  wordchar     word 1-2 gram + char_wb 3-5 gram TF-IDF -> LR
  minilm_emb   all-MiniLM-L6-v2 sentence embeddings (first 256 tokens) -> LR

Every model is run on two label versions over the SAME rows:
  cleaned          `label` column (after the hand-reviewed fixes in ml/data/label_fixes.csv)
  original_labels  `label_original` column (labels as published, rows the review dropped stay dropped)
Train, val and test labels switch together; tricky.csv and xstest_safe are identical in both.

Selection rules (val only; test, tricky and xstest are never used for any choice):
  * C (inverse regularisation) from {0.1, 1, 10, 100}: best val average precision.
  * Decision threshold: maximises val F1, ties go to the higher threshold (fewer false alarms).
  * "recall @ FPR<=1%": a second threshold, the lowest one whose val FPR is <= 1%.
    Reported only, never used to pick a model.

Usage:
    python ml/train_baseline.py                      # all models
    python ml/train_baseline.py --models word_tfidf  # a subset
Results are merged into ml/results/metrics.json under "baselines" -> {cleaned, original_labels} -> model.
"""
import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

ML_DIR = Path(__file__).parent
CACHE = ML_DIR / "cache"
ROOT = ML_DIR.parent
DATA = ML_DIR / "data"
RESULTS = ML_DIR / "results"
MODELS_DIR = ROOT / "models"

SEED = 42
C_GRID = [0.1, 1, 10, 100]
FPR_TARGET = 0.01
N_ERRORS = 15
SNIPPET = 200
BOOT_B = 1000  # bootstrap resamples for test CIs


# --------------------------------------------------------------------------- models
class LengthFeatures(BaseEstimator, TransformerMixin):
    """log1p(characters), log1p(words). Nothing about content."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([[np.log1p(len(t)), np.log1p(len(t.split()))] for t in X])


class MiniLMEmbedder(BaseEstimator, TransformerMixin):
    """all-MiniLM-L6-v2 sentence embeddings (L2-normalised). Cached on disk by text so the C grid and the
    two label variants never re-embed. The encoder is frozen, so no label ever touches it."""

    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_TOKENS = 256

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import hashlib

        import joblib as jl

        path = CACHE / "minilm_all_minilm_l6_v2_256.joblib"
        cache = jl.load(path) if path.exists() else {}
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in X]
        todo = {k: t for k, t in zip(keys, X) if k not in cache}
        if todo:
            from sentence_transformers import SentenceTransformer

            enc = SentenceTransformer(self.MODEL, device="cpu")
            enc.max_seq_length = self.MAX_TOKENS
            vecs = enc.encode(list(todo.values()), batch_size=32, normalize_embeddings=True, show_progress_bar=True)
            cache.update(dict(zip(todo.keys(), vecs)))
            CACHE.mkdir(exist_ok=True)
            jl.dump(cache, path)
        return np.stack([cache[k] for k in keys])


def build_model(name, C):
    lr = LogisticRegression(C=C, solver="liblinear", random_state=SEED, max_iter=2000)
    if name == "length_lr":
        return make_pipeline(LengthFeatures(), StandardScaler(), lr)
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    if name == "word_tfidf":
        return make_pipeline(word, lr)
    if name == "wordchar":
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        return make_pipeline(FeatureUnion([("word", word), ("char", char)]), lr)
    if name == "minilm_emb":
        return make_pipeline(MiniLMEmbedder(), lr)
    raise ValueError(name)


MODEL_NAMES = ["length_lr", "word_tfidf", "wordchar", "minilm_emb"]
DESCRIPTIONS = {
    "length_lr": "length-only LR (floor)",
    "word_tfidf": "word 1-2gram TF-IDF + LR",
    "wordchar": "word + char_wb 3-5gram TF-IDF + LR",
    "minilm_emb": "all-MiniLM-L6-v2 embeddings (256 tok) + LR",
}
# Errors are listed only for content-based models; length_lr's errors are just "long" or "short".
LIST_ERRORS = {"word_tfidf", "wordchar", "minilm_emb"}


# --------------------------------------------------------------------------- thresholds
def tune_f1_threshold(y, scores):
    """Threshold (score >= t => attack) that maximises F1 on y; ties -> higher threshold."""
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    best = np.flatnonzero(f1 == f1.max())[-1]  # thresholds ascend, so the last tie is the highest
    return float(thr[best]), float(f1[best])


def tune_fpr_threshold(y, scores, target=FPR_TARGET):
    """Lowest threshold (score >= t) whose FPR on y is <= target."""
    neg = np.sort(scores[y == 0])[::-1]
    k = int(np.floor(target * len(neg)))  # at most k benign rows may score above t
    return float(np.nextafter(neg[k], np.inf))


# --------------------------------------------------------------------------- evaluation
def rates(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_metrics(y, preds, n_boot=BOOT_B, seed=SEED):
    """Percentile bootstrap over test rows at FROZEN thresholds (no retuning per resample).
    All predictions share the same resamples, so differences between two models/variants are paired.
    Returns one (f1, fpr) pair of length-n_boot arrays per prediction vector."""
    y = np.asarray(y).astype(bool)
    idx = np.random.default_rng(seed).integers(0, len(y), (n_boot, len(y)))
    yy = y[idx]
    out = []
    for pred in preds:
        pp = np.asarray(pred).astype(bool)[idx]
        tp, fp = (yy & pp).sum(1), (~yy & pp).sum(1)
        fn, tn = (yy & ~pp).sum(1), (~yy & ~pp).sum(1)
        out.append((2 * tp / np.maximum(2 * tp + fp + fn, 1), fp / np.maximum(fp + tn, 1)))
    return out


def ci95(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def evaluate_split(df, scores, thr, thr_fpr):
    y = df["label"].to_numpy()
    out = rates(y, scores >= thr)
    f1_b, fpr_b = bootstrap_metrics(y, [scores >= thr])[0]
    out["ci95"] = {"f1": ci95(f1_b), "fpr": ci95(fpr_b), "n_boot": BOOT_B}
    out["roc_auc"] = float(roc_auc_score(y, scores))
    out["pr_auc"] = float(average_precision_score(y, scores))
    at_fpr = rates(y, scores >= thr_fpr)
    out["at_val_fpr_1pct"] = {"recall": at_fpr["recall"], "fpr": at_fpr["fpr"]}
    out["per_source"] = {}
    for src, g in df.assign(score=scores).groupby("source"):
        pred = g["score"].to_numpy() >= thr
        yy = g["label"].to_numpy()
        entry = {"n": len(g), "n_attack": int(yy.sum())}
        if yy.sum():
            entry["recall"] = float(pred[yy == 1].mean())
        if (yy == 0).sum():
            entry["fpr"] = float(pred[yy == 0].mean())
        out["per_source"][src] = entry
    return out


def evaluate_indirect(df, scores, thr, thr_fpr):
    """Held-out indirect-injection set: split metrics + recall/FPR by whether the insertion was annotated
    (comment/bracket) or bare, so a formatting shortcut would show up as a gap."""
    out = evaluate_split(df, scores, thr, thr_fpr)
    pred = scores >= thr
    by = {}
    for (lab, wr), g in df.assign(pred=pred).groupby(["label", "wrapped"]):
        by[f"label{lab}_{'annotated' if wr else 'bare'}"] = {"n": len(g), "flagged": int(g["pred"].sum()), "rate": float(g["pred"].mean())}
    out["by_annotation"] = by
    return out


def evaluate_tricky(df, scores, thr):
    pred = scores >= thr
    out = {}
    for cat, g in df.assign(pred=pred).groupby("category"):
        ok = int((g["pred"] == g["label"]).sum())
        out[cat] = {"correct": ok, "total": len(g), "label": int(g["label"].iloc[0])}
    out["_overall"] = {"correct": int((pred == df["label"]).sum()), "total": len(df)}
    return out


def snippet(text):
    t = " ".join(text.split())
    return t if len(t) <= SNIPPET else t[:SNIPPET] + "..."


def top_errors(df, scores, thr):
    """Most confident wrong predictions on val: highest-scored benign, lowest-scored attacks."""
    d = df.assign(score=scores)
    fp = d[(d.label == 0) & (d.score >= thr)].sort_values("score", ascending=False).head(N_ERRORS)
    fn = d[(d.label == 1) & (d.score < thr)].sort_values("score").head(N_ERRORS)
    return fp, fn


# --------------------------------------------------------------------------- main
def run_model(name, variant, data, errors_md):
    train, val = data["train"], data["val"]

    # 1. C by val average precision (threshold-free, so it doesn't interact with step 2)
    ap_by_c = {}
    for C in C_GRID:
        m = build_model(name, C).fit(train["text"], train["label"])
        ap_by_c[C] = average_precision_score(val["label"], m.predict_proba(val["text"])[:, 1])
    best_C = max(ap_by_c, key=ap_by_c.get)
    model = build_model(name, best_C).fit(train["text"], train["label"])

    def score(texts):
        return model.predict_proba(texts)[:, 1]

    # 2. thresholds on val only
    val_scores = score(val["text"])
    thr, val_f1 = tune_f1_threshold(val["label"].to_numpy(), val_scores)
    thr_fpr = tune_fpr_threshold(val["label"].to_numpy(), val_scores)

    # 3. evaluate everything with those frozen thresholds
    result = {
        "description": DESCRIPTIONS[name],
        "C": best_C,
        "val_ap_by_C": {str(c): v for c, v in ap_by_c.items()},
        "threshold": thr,
        "threshold_fpr1pct": thr_fpr,
        "val": {**rates(val["label"].to_numpy(), val_scores >= thr), "pr_auc": float(average_precision_score(val["label"], val_scores))},
        "test": evaluate_split(data["test"], score(data["test"]["text"]), thr, thr_fpr),
        "tricky": evaluate_tricky(data["tricky"], score(data["tricky"]["text"]), thr),
        "indirect_eval": evaluate_indirect(data["indirect_eval"], score(data["indirect_eval"]["text"]), thr, thr_fpr),
    }
    xs = data["xstest_safe"]
    xs_scores = score(xs["text"])
    result["xstest_safe"] = {
        "n": len(xs),
        "false_positives": int((xs_scores >= thr).sum()),
        "fpr": float((xs_scores >= thr).mean()),
        "fpr_at_val_fpr_1pct_threshold": float((xs_scores >= thr_fpr).mean()),
    }

    if name in LIST_ERRORS and variant == "cleaned":
        fp, fn = top_errors(val, val_scores, thr)
        errors_md.append(f"## {name} ({DESCRIPTIONS[name]}), val threshold {thr:.3f}\n")
        for title, frame in [("Most confident false positives (benign flagged)", fp),
                             ("Most confident false negatives (attacks missed)", fn)]:
            errors_md.append(f"### {title} ({len(frame)})\n")
            errors_md.append("| score | source | text |\n|---|---|---|")
            for _, r in frame.iterrows():
                errors_md.append(f"| {r.score:.3f} | {r.source} | {snippet(r.text).replace('|', '/')} |")
            errors_md.append("")

    PREDS[(variant, name)] = score(data["test"]["text"]) >= thr
    PREDS[(variant, name, "ie")] = score(data["indirect_eval"]["text"]) >= thr
    if name == "wordchar" and variant in ("cleaned", "cleaned_aug"):  # candidate browser pre-check model (Step 5)
        MODELS_DIR.mkdir(exist_ok=True)
        suffix = "" if variant == "cleaned" else "_aug"
        joblib.dump({"model": model, "threshold": thr}, MODELS_DIR / f"baseline_wordchar{suffix}.joblib")
    return result


# variant -> (label column, train file). "_aug" = train + indirect-injection augmentation (ml/data/augment_indirect.py).
# Val, test, tricky and xstest are identical across variants; label columns switch together.
def paired_effect(test, models):
    """aug - plain on the same test rows, same bootstrap resamples (paired), frozen thresholds."""
    out = {}
    for label_version in ["cleaned", "original_labels"]:
        y = test["label" if label_version == "cleaned" else "label_original"].to_numpy()
        for name in models:
            (f_a, r_a), (f_b, r_b) = bootstrap_metrics(y, [PREDS[(label_version, name)], PREDS[(f"{label_version}_aug", name)]])
            out.setdefault(label_version, {})[name] = {
                "delta_f1": float(rates(y, PREDS[(f"{label_version}_aug", name)])["f1"] - rates(y, PREDS[(label_version, name)])["f1"]),
                "delta_f1_ci95": ci95(f_b - f_a),
                "delta_fpr": float(rates(y, PREDS[(f"{label_version}_aug", name)])["fpr"] - rates(y, PREDS[(label_version, name)])["fpr"]),
                "delta_fpr_ci95": ci95(r_b - r_a),
            }
    ie_y = INDIRECT_Y
    for label_version in ["cleaned", "original_labels"]:
        for name in models:
            (f_a, r_a), (f_b, r_b) = bootstrap_metrics(ie_y, [PREDS[(label_version, name, "ie")], PREDS[(f"{label_version}_aug", name, "ie")]])
            a, b = rates(ie_y, PREDS[(label_version, name, "ie")]), rates(ie_y, PREDS[(f"{label_version}_aug", name, "ie")])
            out.setdefault("indirect_eval_" + label_version, {})[name] = {
                "delta_f1": float(b["f1"] - a["f1"]), "delta_f1_ci95": ci95(f_b - f_a),
                "delta_fpr": float(b["fpr"] - a["fpr"]), "delta_fpr_ci95": ci95(r_b - r_a),
            }
    return out


TRICKY_ORDER = ["hidden_injection_document", "benign_document", "benign_mentions_instructions", "benign_suspicious",
                "sneaky_attack", "short_attack", "hinglish", "long_benign"]


def write_comparison(baselines, effect, tricky, models):
    """ml/results/augmentation_comparison.md: with vs without indirect-injection augmentation."""
    cats = [c for c in TRICKY_ORDER if c in set(tricky["category"])] + sorted(set(tricky["category"]) - set(TRICKY_ORDER))
    L = ["# Indirect-injection augmentation: with vs without (generated by ml/train_baseline.py)\n",
         "Test thresholds frozen from val. Brackets are bootstrap 95% CIs over test rows (1,000 resamples). "
         "Delta rows use the same resamples for both variants (paired).\n"]
    for lv, title in [("cleaned", "Cleaned labels"), ("original_labels", "Original labels (same rows)")]:
        L += [f"## Test, {title}\n", "| model | train | F1 [CI] | FPR [CI] | recall | precision | tricky | xstest FP |", "|---|---|---|---|---|---|---|---|"]
        for name in models:
            for suf, tag in [("", "plain"), ("_aug", "+aug")]:
                r = baselines[lv + suf][name]
                t = r["test"]
                L.append(f"| {name} | {tag} | {t['f1']:.3f} [{t['ci95']['f1'][0]:.3f}, {t['ci95']['f1'][1]:.3f}] | "
                         f"{t['fpr']:.3f} [{t['ci95']['fpr'][0]:.3f}, {t['ci95']['fpr'][1]:.3f}] | {t['recall']:.3f} | {t['precision']:.3f} | "
                         f"{r['tricky']['_overall']['correct']}/{r['tricky']['_overall']['total']} | {r['xstest_safe']['false_positives']}/{r['xstest_safe']['n']} |")
            e = effect[lv][name]
            L.append(f"| {name} | **delta** | {e['delta_f1']:+.3f} [{e['delta_f1_ci95'][0]:+.3f}, {e['delta_f1_ci95'][1]:+.3f}] | "
                     f"{e['delta_fpr']:+.3f} [{e['delta_fpr_ci95'][0]:+.3f}, {e['delta_fpr_ci95'][1]:+.3f}] | | | | |")
        L.append("")
        L += [f"## Tricky set per category, {title} (correct/total; plain -> +aug)\n",
              "| category | label | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
        lab = tricky.groupby("category")["label"].agg(lambda x: int(x.iloc[0]) if x.nunique() == 1 else "mixed")   # hinglish holds both labels
        for c in cats:
            cells = []
            for name in models:
                a, b = baselines[lv][name]["tricky"][c], baselines[lv + "_aug"][name]["tricky"][c]
                cells.append(f"{a['correct']}/{a['total']} -> {b['correct']}/{b['total']}")
            L.append(f"| {c} | {lab[c]} | " + " | ".join(cells) + " |")
        L.append("")
        L += [f"## Held-out indirect eval (indirect_eval.parquet), {title}\n",
              "Built the same way as the augmentation (partly in-distribution), from val/test attack strings, fresh passages and fresh wordings. "
              "Annotated = insertion wrapped in a comment/bracket style; bare = plain sentence. CIs are bootstrap 95%.\n",
              "| model | train | F1 [CI] | FPR [CI] | recall | precision | attacks caught: bare / annotated | benign flagged: bare / annotated |",
              "|---|---|---|---|---|---|---|---|"]
        for name in models:
            for suf, tag in [("", "plain"), ("_aug", "+aug")]:
                t = baselines[lv + suf][name]["indirect_eval"]
                b = t["by_annotation"]
                L.append(f"| {name} | {tag} | {t['f1']:.3f} [{t['ci95']['f1'][0]:.3f}, {t['ci95']['f1'][1]:.3f}] | "
                         f"{t['fpr']:.3f} [{t['ci95']['fpr'][0]:.3f}, {t['ci95']['fpr'][1]:.3f}] | {t['recall']:.3f} | {t['precision']:.3f} | "
                         f"{b['label1_bare']['flagged']}/{b['label1_bare']['n']} / {b['label1_annotated']['flagged']}/{b['label1_annotated']['n']} | "
                         f"{b['label0_bare']['flagged']}/{b['label0_bare']['n']} / {b['label0_annotated']['flagged']}/{b['label0_annotated']['n']} |")
            e = effect["indirect_eval_" + lv][name]
            L.append(f"| {name} | **delta** | {e['delta_f1']:+.3f} [{e['delta_f1_ci95'][0]:+.3f}, {e['delta_f1_ci95'][1]:+.3f}] | "
                     f"{e['delta_fpr']:+.3f} [{e['delta_fpr_ci95'][0]:+.3f}, {e['delta_fpr_ci95'][1]:+.3f}] | | | | |")
        L.append("")
    (RESULTS / "augmentation_comparison.md").write_text("\n".join(L), encoding="utf-8")


VARIANTS = {
    "cleaned": ("label", "train"),
    "cleaned_aug": ("label", "train_augmented"),
    "original_labels": ("label_original", "train"),
    "original_labels_aug": ("label_original", "train_augmented"),
}
PREDS = {}  # (variant, model[, "ie"]) -> test / indirect_eval predictions, for the paired bootstrap
INDIRECT_Y = None  # labels of indirect_eval (same for both label versions), set in main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    args = ap.parse_args()

    proc = DATA / "processed"
    raw = {s: pd.read_parquet(proc / f"{s}.parquet")
           for s in ["train", "train_augmented", "val", "test", "xstest_safe", "indirect_eval"]}
    global INDIRECT_Y
    INDIRECT_Y = raw["indirect_eval"]["label"].to_numpy()
    tricky = pd.read_csv(DATA / "tricky.csv")
    RESULTS.mkdir(exist_ok=True)
    metrics_path = RESULTS / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    old = metrics.get("baselines", {})
    if old and "cleaned" not in old:  # first run (original splits, C grid up to 10): keep it for reference
        metrics["baselines_run1_original_splits"] = old
    elif old and "cleaned_aug" not in old:  # second run: before the template-cluster drop and augmentation
        metrics["baselines_run2_before_template_drop"] = old
        old = {}
    baselines = {k: v for k, v in old.items() if k in VARIANTS}
    metrics["baselines"] = baselines
    metrics["meta"] = {
        **metrics.get("meta", {}),
        "seed": SEED,
        "rows": {k: len(v) for k, v in {**raw, "tricky": tricky}.items()},
        "augmented_rows_by_label": {int(k): int(v) for k, v in raw["train_augmented"].query("source == 'augmented_indirect'")["label"].value_counts().items()},
        "label_changes_in_splits": {
            s: {"relabeled_0_to_1": int(((raw[s].label_original == 0) & (raw[s].label == 1)).sum()),
                "relabeled_1_to_0": int(((raw[s].label_original == 1) & (raw[s].label == 0)).sum())}
            for s in ["train", "val", "test"]
        },
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "threshold_rule": "max val F1, ties -> higher threshold; C by val average precision over {0.1,1,10,100}",
    }

    errors_md = [f"# Baseline val errors, cleaned labels (top {N_ERRORS} per direction)\n",
                 "Val only. Test errors are deliberately not inspected.\n"]
    for variant, (col, train_file) in VARIANTS.items():
        data = {"train": raw[train_file].assign(label=raw[train_file][col]),
                **{s: raw[s].assign(label=raw[s][col]) for s in ["val", "test"]}}
        data["tricky"], data["xstest_safe"], data["indirect_eval"] = tricky, raw["xstest_safe"], raw["indirect_eval"]
        for name in args.models:
            print(f"[{variant}] training {name} ...", flush=True)
            baselines.setdefault(variant, {})[name] = run_model(name, variant, data, errors_md)

    if all(v in baselines for v in VARIANTS):
        metrics["augmentation_effect"] = paired_effect(raw["test"], args.models)
        write_comparison(baselines, metrics["augmentation_effect"], tricky, args.models)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    errs = RESULTS / "baseline_val_errors.md"
    if any(m in LIST_ERRORS for m in args.models):
        old_md = errs.read_text(encoding="utf-8") if errs.exists() else ""
        # keep the sections of models not rerun now; the sections just produced come from errors_md
        keep = [blk for blk in old_md.split("\n## ")[1:] if blk.split(" ")[0] not in args.models]
        errs.write_text("\n".join(errors_md) + "".join("\n## " + b for b in keep), encoding="utf-8")

    for variant in VARIANTS:
        print(f"\n== {variant} ==")
        print(f"{'model':<11}{'C':>5}{'thr':>7} | {'P':>6}{'R':>6}{'F1':>6}{'FPR':>6}{'AUC':>6} | {'R@1%':>6}{'FPR':>6} | tricky | xstest FP")
        for name in args.models:
            r = baselines[variant][name]
            t, a = r["test"], r["test"]["at_val_fpr_1pct"]
            print(f"{name:<11}{r['C']:>5}{r['threshold']:>7.3f} | {t['precision']:>6.3f}{t['recall']:>6.3f}{t['f1']:>6.3f}"
                  f"{t['fpr']:>6.3f}{t['roc_auc']:>6.3f} | {a['recall']:>6.3f}{a['fpr']:>6.3f} | "
                  f"{r['tricky']['_overall']['correct']}/{r['tricky']['_overall']['total']}    | "
                  f"{r['xstest_safe']['false_positives']}/{r['xstest_safe']['n']}")


if __name__ == "__main__":
    main()
