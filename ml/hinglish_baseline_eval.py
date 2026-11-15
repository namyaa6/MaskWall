"""BASELINE of the two served int8 models on the Hinglish / prompt-talk evaluation sets (before any hard-negative training).

Sets (all evaluation-only; see ml/data/build_hinglish_eval.py and docs/decisions.md Day 2.6):
  hinglish_human_benign   104 HUMAN-written Hinglish assistant queries (Hinglish-TOP, Apache-2.0) + their English originals as a paired control
  hinglish_attacks_ai     24 AI-drafted Hinglish attacks (Gemini), 4 categories
  prompt_talk_benign_ai   100 AI-drafted benign prompts that mention system prompts / instructions (50 English, 50 Hinglish; Gemini)
  manual7                 the owner's 7 hand-tested prompts; the tricky.csv Hinglish rows are listed too
Thresholds are the SERVER's (backend/model/<name>/model_meta.json): block = max(thr_f1, thr_fpr1pct), warn = min(...). Scoring is
the serving scanner (backend/app/scanners/injection.py, head_tail mode; all prompts here are far below one window).
Reporting only. Run: python ml/hinglish_baseline_eval.py -> ml/results/hinglish_baseline.json and .md
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.scanners.injection import InjectionScanner  # noqa: E402

PROC = ROOT / "ml" / "data" / "processed"
RES = ROOT / "ml" / "results"


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": round(k / max(n, 1), 3), "ci95": wilson(k, n)}


def main():
    sets = {n: pd.read_parquet(PROC / f"{n}.parquet") for n in ["hinglish_human_benign", "hinglish_attacks_ai", "prompt_talk_benign_ai", "manual7"]}
    tricky = pd.read_csv(ROOT / "ml" / "data" / "tricky.csv")
    report = {}
    for name in ("minilm", "deberta"):
        sc = InjectionScanner.load(ROOT / "backend" / "model" / name, mode="head_tail")
        warn, block = min(sc.thr_f1, sc.thr_fpr1pct), max(sc.thr_f1, sc.thr_fpr1pct)
        score = lambda texts: pd.Series([sc.scan(t).score for t in texts])
        r = {"warn": warn, "block": block}
        h = sets["hinglish_human_benign"]
        hs, es = score(h.text), score(h.en_control)
        r["human_hinglish_benign"] = {"block": rate((hs >= block).sum(), len(hs)), "warn_or_block": rate((hs >= warn).sum(), len(hs)), "median_score": round(float(hs.median()), 4),
                                      "max_score": round(float(hs.max()), 4),
                                      "english_control_block": rate((es >= block).sum(), len(es)), "english_control_warn_or_block": rate((es >= warn).sum(), len(es)),
                                      "flagged_examples": h.assign(score=hs)[hs >= warn].sort_values("score", ascending=False).head(6)[["score", "text", "en_control"]].round(3).to_dict("records")}
        a = sets["hinglish_attacks_ai"]
        as_ = score(a.text)
        r["ai_hinglish_attacks"] = {"block": rate((as_ >= block).sum(), len(as_)), "warn_or_block": rate((as_ >= warn).sum(), len(as_)),
                                    "by_category": {c: {"block": rate((as_[a.category == c] >= block).sum(), (a.category == c).sum()),
                                                        "warn_or_block": rate((as_[a.category == c] >= warn).sum(), (a.category == c).sum())} for c in a.category.unique()},
                                    "missed_at_block": a.assign(score=as_)[as_ < block][["category", "score", "text"]].round(3).to_dict("records")}
        p = sets["prompt_talk_benign_ai"]
        ps = score(p.text)
        r["prompt_talk_benign_ai"] = {"block": rate((ps >= block).sum(), len(ps)), "warn_or_block": rate((ps >= warn).sum(), len(ps)),
                                      "by_language": {lg: {"block": rate((ps[p.language == lg] >= block).sum(), (p.language == lg).sum()),
                                                           "warn_or_block": rate((ps[p.language == lg] >= warn).sum(), (p.language == lg).sum())} for lg in p.language.unique()},
                                      "by_category": {c: {"block": rate((ps[p.category == c] >= block).sum(), (p.category == c).sum())} for c in p.category.unique()},
                                      "by_language_category_block": {f"{lg}/{c}": rate((ps[(p.language == lg) & (p.category == c)] >= block).sum(), ((p.language == lg) & (p.category == c)).sum())
                                                                     for lg in p.language.unique() for c in p.category.unique()},
                                      "flagged_at_block": p.assign(score=ps)[ps >= block][["language", "category", "score", "text"]].round(3).to_dict("records")}
        m = sets["manual7"]
        ms = score(m.text)
        r["manual7"] = [{"text": t[:70], "label": int(l), "score": round(float(s_), 4), "verdict": "block" if s_ >= block else "warn" if s_ >= warn else "allow", "in_tricky_set": bool(it)}
                        for t, l, s_, it in zip(m.text, m.label, ms, m.in_tricky_set)]
        th = tricky[tricky.category == "hinglish"]
        ts = score(th.text)
        r["tricky_hinglish_rows"] = [{"text": t[:70], "label": int(l), "score": round(float(s_), 4), "verdict": "block" if s_ >= block else "warn" if s_ >= warn else "allow"}
                                     for t, l, s_ in zip(th.text, th.label, ts)]
        report[name] = r
        print(f"[{name}] human Hinglish benign FPR@block {r['human_hinglish_benign']['block']['k']}/104 (English control {r['human_hinglish_benign']['english_control_block']['k']}/104); "
              f"AI Hinglish attacks recall@block {r['ai_hinglish_attacks']['block']['k']}/24; prompt-talk benign FPR@block {r['prompt_talk_benign_ai']['block']['k']}/100", flush=True)
    (RES / "hinglish_baseline.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    f = lambda x: f"{x['k']}/{x['n']} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]"
    L = ["# Baseline on the Hinglish / prompt-talk evaluation sets (generated by ml/hinglish_baseline_eval.py)\n",
         "Served thresholds (block = max, warn = min of the model's val tuning). Rates k/n with Wilson 95% intervals. Human Hinglish = Hinglish-TOP (Apache-2.0), human-written; the two AI-drafted sets are Gemini-generated.\n",
         "| set | metric | MiniLM int8 | DeBERTa int8 |", "|---|---|---|---|"]
    g = lambda fn: [fn(report[n]) for n in ("minilm", "deberta")]
    rows = [("human Hinglish benign (104)", "false positives at block", lambda r: f(r["human_hinglish_benign"]["block"])),
            ("", "flagged at warn or block", lambda r: f(r["human_hinglish_benign"]["warn_or_block"])),
            ("", "English originals (control), at block", lambda r: f(r["human_hinglish_benign"]["english_control_block"])),
            ("AI-drafted Hinglish attacks (24)", "caught at block", lambda r: f(r["ai_hinglish_attacks"]["block"])),
            ("", "caught at warn or block", lambda r: f(r["ai_hinglish_attacks"]["warn_or_block"]))]
    for c in ("english_keywords", "hindi_only", "roleplay", "indirect"):
        rows.append(("", f"  {c}: caught at block", lambda r, c=c: f(r["ai_hinglish_attacks"]["by_category"][c]["block"])))
    rows += [("prompt-talk benign (100)", "false positives at block", lambda r: f(r["prompt_talk_benign_ai"]["block"])),
             ("", "  English (50)", lambda r: f(r["prompt_talk_benign_ai"]["by_language"]["en"]["block"])),
             ("", "  Hinglish (50)", lambda r: f(r["prompt_talk_benign_ai"]["by_language"]["hinglish"]["block"]))]
    for c in ("write_system_prompt", "explain_concepts", "process_instructions", "task_steps", "refer_earlier"):
        rows.append(("", f"  {c}: false positives at block", lambda r, c=c: f(r["prompt_talk_benign_ai"]["by_category"][c]["block"])))
    for a, b, fn in rows:
        L.append(f"| {a} | {b} | " + " | ".join(g(fn)) + " |")
    L += ["", "## The owner's 7 manual prompts\n", "| # | label | prompt | MiniLM score / verdict | DeBERTa score / verdict | also in tricky.csv |", "|---|---|---|---|---|---|"]
    for i, (a, b) in enumerate(zip(report["minilm"]["manual7"], report["deberta"]["manual7"])):
        L.append(f"| {i + 1} | {a['label']} | {a['text']} | {a['score']:.4f} / {a['verdict']} | {b['score']:.4f} / {b['verdict']} | {'yes' if a['in_tricky_set'] else ''} |")
    L += ["", "## tricky.csv `hinglish` rows (6 benign + 5 attacks)\n", "| label | prompt | MiniLM | DeBERTa |", "|---|---|---|---|"]
    for a, b in zip(report["minilm"]["tricky_hinglish_rows"], report["deberta"]["tricky_hinglish_rows"]):
        L.append(f"| {a['label']} | {a['text']} | {a['score']:.3f} / {a['verdict']} | {b['score']:.3f} / {b['verdict']} |")
    (RES / "hinglish_baseline.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote hinglish_baseline.json / .md")


if __name__ == "__main__":
    main()
