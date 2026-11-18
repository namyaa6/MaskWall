# Notice and attribution

MaskWall's own code is MIT-licensed (see `LICENSE`). This file lists the third-party datasets, models and derived data the project uses, with the licence recorded for each and where it was checked. "Hub tag" means the licence field on the Hugging Face page, read on 2026-11-18. Where no tag exists, the licence comes from the dataset card text as recorded in `docs/decisions.md`; that is weaker evidence and is marked.

## Datasets used to train or evaluate the injection classifier
| Dataset | Licence | Use |
|---|---|---|
| `deepset/prompt-injections` | Apache-2.0 | train/val/test |
| `jackhhao/jailbreak-classification` | Apache-2.0 | train/val/test |
| `Lakera/gandalf_ignore_instructions` | MIT (Hub tag) | train/val/test |
| `TrustAIRLab/in-the-wild-jailbreak-prompts` | MIT | train/val/test |
| `databricks/databricks-dolly-15k` | **CC BY-SA 3.0** (Hub tag) | benign training rows (sampled) |
| `Paul/XSTest` (safe prompts) | CC BY 4.0 (Hub tag) | evaluation only |
| `WillHeld/hinglish_top` (Hinglish-TOP, Google Research) | Apache-2.0 per the dataset card as recorded in `docs/decisions.md` (**no Hub licence tag**) | Hinglish benign training negatives (train split) and an evaluation set (test split) |
| Synthetic hard negatives and Hinglish attacks | generated with Groq `gpt-oss-120b`; evaluation text with Gemini | training / evaluation, see `docs/decisions.md` Day 2.6 to 2.9 |

Share-alike note: Dolly's CC BY-SA 3.0 applies to the dataset text. Whether it extends to a classifier trained on a sample of it is not settled law; this project does not redistribute the dataset text, only model weights, and states the training sources on the model card. Nothing under a non-commercial licence is used (Alpaca was rejected for that reason).

## Base models
| Model | Licence (Hub tag) | Use |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | fine-tuned into the served injection classifier (int8 ONNX) and used for embeddings in a baseline |
| `microsoft/deberta-v3-small` | MIT | fine-tuned comparison model (benchmark only; not served) |
| `dslim/distilbert-NER` (browser build `onnx-community/distilbert-NER-ONNX`) | Apache-2.0 for `dslim/distilbert-NER`; the ONNX re-upload has **no Hub tag** | person-name detection in the browser, loaded from the Hugging Face Hub at run time. Trained on CoNLL-2003 (English news) |
| Meta Llama Prompt Guard 2 | Llama 4 Community License | **evaluation reference only**; weights are not redistributed and are kept out of the repository |

## Derived data shipped in the repository
- `frontend/src/data/commonWordsB2.json` (5,450 words, used only by the opt-in "Strict name hiding"). Built from word counts of Hinglish-TOP (train split, Apache-2.0) and **WikiText-103 (Wikipedia text, CC BY-SA 3.0 and GFDL)**. This file is therefore **shared under CC BY-SA 3.0** with attribution to Wikipedia contributors and the WikiText-103 authors (Merity et al., Salesforce). The share-alike condition applies to this word list, not to the rest of the code.

## Software
Runtime dependencies keep their own licences (React, Vite, Transformers.js and ONNX Runtime Web, FastAPI, Pydantic, slowapi, onnxruntime, tokenizers and others); see each package's metadata (`frontend/package-lock.json`, `backend/requirements.lock.txt`). Fonts: Inter and JetBrains Mono through `@fontsource` (SIL Open Font License 1.1).

## Services
Language-model answers come from Groq and Google Gemini through their public APIs under the providers' terms; no model output is redistributed as data.
