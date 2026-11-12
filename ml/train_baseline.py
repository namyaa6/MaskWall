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


def evaluate_split(df, scores, thr, thr_fpr):
    y = df["label"].to_numpy()
    out = rates(y, scores >= thr)
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

    if name == "wordchar" and variant == "cleaned":  # candidate browser pre-check model (Step 5)
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": model, "threshold": thr}, MODELS_DIR / "baseline_wordchar.joblib")
    return result


VARIANTS = {"cleaned": "label", "original_labels": "label_original"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    args = ap.parse_args()

    raw = {s: pd.read_parquet(DATA / "processed" / f"{s}.parquet") for s in ["train", "val", "test", "xstest_safe"]}
    tricky = pd.read_csv(DATA / "tricky.csv")
    RESULTS.mkdir(exist_ok=True)
    metrics_path = RESULTS / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    old = metrics.get("baselines", {})
    if old and "cleaned" not in old:  # first run (original splits, C grid up to 10): keep it for reference
        metrics["baselines_run1_original_splits"] = old
    baselines = {k: v for k, v in old.items() if k in VARIANTS}
    metrics["baselines"] = baselines
    metrics["meta"] = {
        **metrics.get("meta", {}),
        "seed": SEED,
        "rows": {k: len(v) for k, v in {**raw, "tricky": tricky}.items()},
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
    for variant, col in VARIANTS.items():
        data = {s: raw[s].assign(label=raw[s][col]) for s in ["train", "val", "test"]}
        data["tricky"], data["xstest_safe"] = tricky, raw["xstest_safe"]
        for name in args.models:
            print(f"[{variant}] training {name} ...", flush=True)
            baselines.setdefault(variant, {})[name] = run_model(name, variant, data, errors_md)

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
