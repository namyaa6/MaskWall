# MaskWall — Privacy-First LLM Firewall

## What this project is
MaskWall is a two-layer security system for LLM applications:

1. **Private AI layer (in the browser):** a quantized ONNX model runs on the user's device and masks personal data (names, phone numbers, emails, Aadhaar, PAN) *before* any text leaves the device. The server only ever sees placeholders like `[PHONE_1]`.
2. **LLM firewall (on the server):** a FastAPI proxy that detects prompt injection / jailbreaks before the prompt reaches the LLM, and scans the LLM's output for leaks (system prompt, secrets) before returning it.

Tagline: *Defense in depth, private by design.*

This is a portfolio project. It must be deployed, measurable, well-documented, and fully explainable by its author in an interview.

## Architecture
```
User types prompt
      │
      ▼
[Layer 1 — Browser]  quantized ONNX models via Transformers.js (WASM backend)
  • Detect + mask PII (NER model + regex/checksum for Aadhaar & PAN)
  • Quick injection pre-check (instant warning, works offline)
      │  only masked text is sent
      ▼
[Layer 2 — FastAPI firewall]
  • Fine-tuned injection/jailbreak classifier
  • Policy engine (block / warn / allow), rate limiting, logging
      │
      ▼
[LLM]  Groq API (Gemini as fallback)
      │
      ▼
[Layer 3 — Output scan]
  • Block system-prompt leakage, secrets, API keys
  • Unmask placeholders on the client side only
      │
      ▼
Response shown to user
```

## Tech stack
- **ML:** Python 3.13, scikit-learn (baseline), Hugging Face Transformers + Datasets, fine-tune DistilBERT or MiniLM on Google Colab GPU
- **Optimization:** ONNX export + int8 quantization (Hugging Face Optimum / ONNX Runtime)
- **Backend:** FastAPI, Pydantic, SQLite (logs), slowapi (rate limiting), pytest
- **Frontend:** React + Vite, Transformers.js (WASM backend, not WebGPU)
- **LLM:** Groq API (key in `.env` as `GROQ_API_KEY`)
- **Deploy:** frontend on Vercel, backend on Hugging Face Spaces (Docker), models on Hugging Face Hub
- **CI:** GitHub Actions running pytest + lint on every push

## Folder structure
```
maskwall/
├── ml/
│   ├── data/                 # dataset loading + preprocessing scripts
│   ├── train_baseline.py     # TF-IDF + logistic regression
│   ├── train_transformer.ipynb  # fine-tuning notebook (run on Colab)
│   ├── evaluate.py           # precision, recall, F1, FPR, confusion matrix
│   ├── export_quantize.py    # ONNX export + int8 quantization
│   └── benchmark.py          # fp32 vs int8: size, latency, F1
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app + /chat proxy endpoint
│   │   ├── scanners/         # injection.py, output.py, pii_regex.py
│   │   ├── policy.py         # block/warn/allow rules
│   │   ├── llm.py            # Groq client
│   │   └── logging_db.py     # SQLite event logging
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pii/              # in-browser model loading + masking + unmasking
│   │   ├── pages/            # Chat, AttackPlayground, Dashboard, Benchmarks
│   │   └── components/
│   └── package.json
├── models/                   # local ONNX files (gitignored; hosted on HF Hub)
├── docs/                     # architecture diagram, screenshots, demo GIF
├── .github/workflows/ci.yml
├── docker-compose.yml
├── .env                      # secrets (gitignored)
└── README.md
```

## Key features
- **Split-screen chat:** "What you typed" vs "What the server received" — the core demo.
- **Privacy mode toggle:** turns the in-browser masking on/off.
- **Attack playground:** preset injection/jailbreak attacks plus free-text; shows the verdict and confidence.
- **Dashboard:** attacks blocked, PII caught, latency per layer (from SQLite logs).
- **Benchmarks page:** classifier metrics and fp32 vs int8 comparison.

## Build plan

### Day 1 — Models (`ml/`)
1. Load prompt-injection / jailbreak datasets from Hugging Face; add plenty of benign prompts so false positives can be measured honestly. Check licenses.
2. Hold out a test set; also hand-write ~50 tricky examples (subtle attacks + benign prompts that *look* suspicious).
3. `ml/train_baseline.py`: four models on the same splits: (1) length-only LR, (2) word TF-IDF + LR, (3) word+char n-gram TF-IDF + LR, (4) all-MiniLM-L6-v2 embeddings + LR. Tune threshold on val only. Evaluate on test (precision, recall, F1, FPR, confusion matrix), tricky set (counts per category), xstest_safe (FPR). List the 15 most confident FPs and FNs on val. Save to `ml/results/metrics.json`.
4. `ml/train_transformer.ipynb` (Colab GPU): fine-tune MiniLM and DeBERTa-v3-small on the same splits; head+tail truncation (first 256 + last 256 tokens); 3–4 epochs with early stopping on val; fixed seed; threshold tuned on val; same evaluation as baselines.
5. `ml/export_quantize.py` + `ml/benchmark.py`: ONNX fp32 and int8 for both models; report size (MB), CPU latency p50/p95 at batch size 1, and F1 for each. Server model selection rule decided in advance: highest val F1 among int8 models with p50 CPU latency under 50 ms. Also pick a browser pre-check model: the most accurate model under ~30 MB int8 (fall back to the word+char TF-IDF model exported with skl2onnx). Record both selections in `docs/decisions.md`.
6. (Optional) Evaluate Meta's Prompt Guard on test, tricky, and xstest as an external reference only; skip if gated access causes friction.
7. Comparison table + chart in `ml/results/`, update `docs/decisions.md` and `docs/interview_notes.md`, then a review subagent checks the day's code for bugs and leakage.

### Day 2 — Firewall API (`backend/`)
1. `/chat` endpoint: input scan → policy → Groq call → output scan → response.
2. Injection scanner using the quantized ONNX model (onnxruntime).
3. Regex + checksum scanner for Aadhaar (Verhoeff) and PAN, as a server-side backup.
4. Output scanner: system-prompt leakage, API-key/secret patterns.
5. Rate limiting, SQLite logging, `/stats` endpoint for the dashboard.
6. pytest tests for every scanner and the policy engine.

### Day 3 — Frontend + browser layer (`frontend/`)
1. Pick a PII NER model with an ONNX version usable by Transformers.js, then load the quantized model in the browser with Transformers.js (WASM).
2. Mask PII to placeholders, keep the mapping in memory only, unmask the response client-side.
3. Build Chat (split screen), Attack Playground, Dashboard, Benchmarks pages.
4. **Fallback rule:** if in-browser loading blocks progress for more than ~2 hours, move masking to the server temporarily and continue. Do not let this layer sink the project.

### Day 4 — Ship
1. Dockerfile + docker-compose; deploy backend to HF Spaces, frontend to Vercel.
2. GitHub Actions CI (pytest + lint).
3. README: architecture diagram, demo GIF, metrics tables, design decisions, limitations, how to run locally.
4. Record a 2-minute demo video.
5. Rehearse explaining every design decision and metric.

## Metrics to report (README)
- Injection classifier: precision, recall, F1, **false-positive rate on benign prompts**, on both the public test set and the hand-written tricky set
- Baseline vs fine-tuned comparison
- Quantization: model size (MB), latency (ms), F1 — fp32 vs int8
- PII detection recall on a small labeled sample
- End-to-end latency added by each layer

## Working rules for Claude Code
- **Build one step at a time.** Finish and test each step before starting the next. Don't generate whole days of code in one go.
- **Explain decisions.** When choosing a model, library, threshold, or design, briefly say why, and add notable decisions to `docs/decisions.md` (used for the README and interviews).
- **Never hardcode secrets.** Read keys from `.env`. Never commit `.env` or model binaries.
- **Write tests** alongside backend code.
- **Be honest about metrics.** Never tune on the test set; report false positives; don't call it "production-grade".
- **Keep it simple.** Prefer working and well-explained over clever. Flag anything likely to take longer than planned before starting it.
- **Suggest a git commit** after each working step with a clear message.
- After each completed step, append to docs/interview_notes.md.
