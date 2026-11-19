"""Create the two Hugging Face model repos and upload the model files + generated model cards.

    python scripts/upload_models.py --render-only      # write the cards to scripts/_cards/ and stop (review them first)
    python scripts/upload_models.py --cards-only       # re-upload ONLY the README model cards (after a wording correction)
    python scripts/upload_models.py                    # create/update the repos and upload everything

Uses the login from `hf auth login` (the token is never read or printed here). Numbers in the cards are read from
frontend/src/data/results.json (itself generated from ml/results), never typed by hand.

Repos (under the logged-in user):
  maskwall-injection-minilm   SERVED model. Backend layout (model.int8.onnx, tokenizer.json, model_meta.json) AND the
                              Transformers.js layout (onnx/model_quantized.onnx, config.json, tokenizer_config.json) side by
                              side at the repo root; it is the same ONNX file, which the Hub stores once.
  maskwall-injection-deberta  int8 DeBERTa-v3-small, for the benchmark / reference only. NOT served.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "frontend" / "src" / "data" / "results.json").read_text(encoding="utf-8"))
SERVED_DIR = ROOT / "backend" / "model" / "minilm"
BROWSER_DIR = ROOT / "frontend" / "public" / "models" / "maskwall-minilm"
DEBERTA_DIR = ROOT / "backend" / "model" / "deberta"
CARDS = Path(__file__).resolve().parent / "_cards"
REPO_MINILM, REPO_DEBERTA = "maskwall-injection-minilm", "maskwall-injection-deberta"
GITHUB = "https://github.com/namyaa6/MaskWall"


def frac(d: dict) -> str:
    return f"{d['k']}/{d['n']} ({100 * d['k'] / d['n']:.1f}%)"


def minilm_card(user: str) -> str:
    s = RESULTS["served"]
    meta = json.loads((SERVED_DIR / "model_meta.json").read_text(encoding="utf-8"))
    return f"""---
license: apache-2.0
base_model: sentence-transformers/all-MiniLM-L6-v2
library_name: transformers.js
pipeline_tag: text-classification
language: [en]
tags: [prompt-injection, jailbreak, llm-firewall, onnx, int8, transformers.js]
---

# MaskWall injection classifier (MiniLM-L6, int8 ONNX)

Binary classifier used by [MaskWall]({GITHUB}), a privacy-first LLM firewall. It labels a prompt **benign (0)** or
**prompt injection / jailbreak (1)**. The same file is used twice: on the server (onnxruntime) as the real check, and in the
browser (Transformers.js, WASM) as an instant early warning.

> **Not a guarantee.** This is a statistical classifier with the measured error rates below. It misses attacks and it flags
> some harmless prompts. Use it as one layer, not as the only defence.

## Files
| file | used by |
|---|---|
| `model.int8.onnx`, `tokenizer.json`, `model_meta.json` | the FastAPI backend (`MASKWALL_SERVER_MODEL=minilm`) |
| `onnx/model_quantized.onnx`, `config.json`, `tokenizer_config.json`, `tokenizer.json` | Transformers.js in the browser (`AutoModelForSequenceClassification`, `dtype: "q8"`) |

Both layouts hold the **same** weights (dynamic int8 quantisation, {s['int8_mb']} MB; fp32 was {s['fp32_mb']} MB, {s['params_m']} M parameters).

## How it is meant to be run
Texts longer than the model window are scored in overlapping 510-token windows and the **maximum** score is used (a
"sliding window"). Two thresholds on the attack probability: **block at {s['block']:.3f}**, **warn at {s['warn']:.3f}**
(both tuned on the validation split only; see `model_meta.json`). `max_content_tokens` 510, window stride 255.

## Training data
Fine-tuned from `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) on English prompts from public datasets, after
cleaning (exact and near-duplicate removal across sources, conflicting labels dropped, label fixes after manual review):
`deepset/prompt-injections` (Apache-2.0), `jackhhao/jailbreak-classification` (Apache-2.0),
`Lakera/gandalf_ignore_instructions` (MIT), `TrustAIRLab/in-the-wild-jailbreak-prompts` (MIT),
`databricks/databricks-dolly-15k` (CC BY-SA 3.0, sampled benign rows), plus 800 synthetic training rows (half attacks, half
matched benign) that hide an instruction inside text the user pasted ("summarize this...") because the public data barely
contains that case. **Hard-negative round:** benign prompts that *talk
about* attacks or system prompts (English and Hinglish), generated with an LLM with matched templates for both classes, plus
human Hinglish assistant queries from Hinglish-TOP (train split). `Paul/XSTest` (CC BY 4.0) and all evaluation sets were
never trained on. Full licence table: [NOTICE.md]({GITHUB}/blob/main/NOTICE.md). Share-alike (Dolly, CC BY-SA 3.0) applies to
that dataset's text; whether it reaches a model trained on a sample of it is not settled, so it is stated here.

## Measured results (served int8 model, at the thresholds above)
Source of truth: `ml/results/` in the repository; this table is generated from it. Counts are shown so the rates are checkable.

| set | result |
|---|---|
| public test set, attacks caught (recall) | {frac(s['test_recall'])} |
| public test set, benign flagged (false positives) | {frac(s['test_fp'])} |
| test F1 | {s['test_f1']:.3f} |
| XSTest safe prompts flagged | {frac(s['xstest_fp'])} |
| hidden-instruction-in-document attacks caught (synthetic, built with the same recipe as 800 training rows, so an upper bound) | {frac(s['indirect_recall'])} |
| human-written Hinglish assistant queries flagged (same domain as some training rows) | {frac(s['hinglish_benign_fp'])} |
| LLM-written "prompt-talk" benign flagged | {frac(s['prompt_talk_fp'])} |
| Hinglish attacks caught (24 AI-written prompts) | {frac(s['hinglish_attacks'])} |
| tricky set (49 prompts drafted with AI help, so not blind): attacks caught / benign flagged | {frac(s['tricky_attacks'])} / {frac(s['tricky_benign_fp'])} |
| long documents (114 benign / 38 with a hidden attack): benign flagged / hidden attack caught | {frac(s['long_benign_fp'])} / {frac(s['long_middle_caught'])} |

Latency depends on the CPU and on text length; see the Benchmarks page of the live app for the figures measured on the
deployment hardware.

## Limitations
- **English and Hinglish only.** Other languages are not evaluated. A German-language filter step removed them from training.
- The label "attack" covers instruction overrides, jailbreak personas and instructions hidden in documents. It does **not**
  judge whether a request is harmful.
- It still flags some benign prompts that quote attack phrases or discuss system prompts (the hard-negative round reduced,
  not removed, this), and long benign documents are flagged more often than short prompts.
- Test and tricky sets are small (e.g. 21 tricky attacks); the confidence intervals are wide and several differences are
  within noise. The thresholds were tuned on validation only.
- Adaptive attackers who can see the model can evade it. Paraphrased or obfuscated attacks are a known weakness.
- Some training rows were written by an LLM; a style shortcut was checked for (same generator and templates for both classes)
  but cannot be ruled out entirely.

Code, training notebooks, decisions and all numbers: {GITHUB}
"""


def deberta_card() -> str:
    q = {(r["model"], r["precision"]): r for r in RESULTS["quantization"]}
    t = {r["model"]: r for r in RESULTS["transformers"]}["DeBERTa-v3-small fine-tuned"]
    i8 = q[("DeBERTa-v3-small", "int8")]
    return f"""---
license: mit
base_model: microsoft/deberta-v3-small
pipeline_tag: text-classification
language: [en]
tags: [prompt-injection, jailbreak, onnx, int8, benchmark]
---

# MaskWall comparison model: DeBERTa-v3-small (int8 ONNX), benchmark only

This is the larger comparison model from [MaskWall]({GITHUB}). It is **not served** by the live app: the served model is
[`{{user}}/{REPO_MINILM}`](https://huggingface.co/{{user}}/{REPO_MINILM}). It is hosted as a benchmark reference: it did not fit next to the app in the 512 MB of the deployment host, so only
laptop latency exists for it.

- Fine-tuned from `microsoft/deberta-v3-small` (MIT) on the **first-round cleaned and augmented training set** (before the hard-negative round).
- int8 file {i8['size_mb']} MB. Round-1 test results (at its own validation-tuned threshold): F1 {t['f1']:.3f}, false-positive
  rate {t['fpr']:.3f} ({t['fp']}/{t['benign']}), XSTest safe flagged {t['xstest_fp']}/{t['xstest_n']}.
- Same limitations as the MiniLM model. It flags more benign prompts (higher false-positive rate) and is larger and slower; the final choice of
  MiniLM rested on the served-threshold and Hinglish analysis in `docs/decisions.md` (the original selection rule was amended after results
  were seen, and the record says so).
- Training data and licences: see [NOTICE.md]({GITHUB}/blob/main/NOTICE.md).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--cards-only", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    user = api.whoami()["name"]            # the token stays inside huggingface_hub
    CARDS.mkdir(exist_ok=True)
    (CARDS / "minilm.md").write_text(minilm_card(user), encoding="utf-8")
    (CARDS / "deberta.md").write_text(deberta_card().replace("{user}", user), encoding="utf-8")
    print(f"cards written to {CARDS}")
    if args.render_only:
        return

    plan = {
        f"{user}/{REPO_MINILM}": [
            (SERVED_DIR / "model.int8.onnx", "model.int8.onnx"),
            (SERVED_DIR / "tokenizer.json", "tokenizer.json"),
            (SERVED_DIR / "model_meta.json", "model_meta.json"),
            (BROWSER_DIR / "onnx" / "model_quantized.onnx", "onnx/model_quantized.onnx"),
            (BROWSER_DIR / "config.json", "config.json"),
            (BROWSER_DIR / "tokenizer_config.json", "tokenizer_config.json"),
            (CARDS / "minilm.md", "README.md"),
        ],
        f"{user}/{REPO_DEBERTA}": [
            (DEBERTA_DIR / "model.int8.onnx", "model.int8.onnx"),
            (DEBERTA_DIR / "tokenizer.json", "tokenizer.json"),
            (DEBERTA_DIR / "model_meta.json", "model_meta.json"),
            (CARDS / "deberta.md", "README.md"),
        ],
    }
    for repo, files in plan.items():
        api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
        for src, dest in files:
            if args.cards_only and dest != "README.md":
                continue
            if not src.exists():
                raise SystemExit(f"missing {src}")
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dest, repo_id=repo, repo_type="model",
                            commit_message=f"Add {dest}")
            print(f"uploaded {repo}/{dest}")
    print("done:", *(f"https://huggingface.co/{r}" for r in plan))


if __name__ == "__main__":
    main()
