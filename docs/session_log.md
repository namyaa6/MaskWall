# Session log: Day 3, frontend and browser privacy layer (2026-11-17)

The plan was approved by the owner with four additions. Later the owner said they would not review the names sample, and asked for the full plan (3.0 to 3.9) to be run without waiting. Each step was built, tested and committed on its own, with an entry in `docs/decisions.md` (Day 3.0 to 3.9) and in `docs/interview_notes.md`. That file is gitignored, so it was updated but not committed.

## What was built

| step | commit | what |
|---|---|---|
| 3.0 | `927fb4f` | `frontend/` scaffold (Vite 6, React 18, TypeScript strict, react-router 6, Vitest, ESLint; pinned versions), five routes, design tokens, contrast unit test, "Firewall online" indicator. The backend's PII format cases moved to the shared `backend/tests/fixtures/pii_cases.json` (49 PII + 12 normalisation cases). |
| 3.1 | `59be7d3` | TypeScript ports of normalisation and the Aadhaar (Verhoeff) / PAN / email / mobile detectors, held to the same shared cases as the server; `PlaceholderVault` (memory only, per conversation, `[EMAIL_1]` style, never unmasks the server's `[S_…]`, skips placeholders the user typed literally). |
| 3.2 | `d438f2d`, `89911b5` | Name (NER) model selection. The rule and the sample were committed **before** measuring. Three candidates were measured in headless Chromium (WASM, q8). Selected: `onnx-community/distilbert-NER-ONNX` (66 MB, 40 ms, 47/56 names). |
| 3.3 | `5a6735c` | In-browser attack pre-check: the served int8 MiniLM (23 MB) with the server's sliding windows and thresholds (block 0.610, warn 0.368), in a Web Worker. Parity checked in Node and in Chromium WASM. |
| 3.4 | `741b0e5` | Backend: `/stats` gains `recent` (last 20 events, whitelisted fields, no text, fingerprint or id) and `added_ms`. |
| 3.5 | `b3ff939` | Chat page: on-device masking before sending, unmasking on the device, parallel early warning, result labels with reason and next step, safe Markdown (no images), cut-off indicator, "How this message was handled" panel, download progress, a no-history note. |
| 3.6 | `ba2a2db` | Attack playground (`/scan` only): six tagged examples, a one-sentence verdict, score out of 1, a 3-zone meter at the live thresholds, PII / coverage / timing facts. |
| 3.7 | `06c4a33` | Dashboard: "message text is never saved" first, totals, recent-messages table. |
| 3.8 | `8bc2f74` | How it works and Benchmarks. Every number is generated from `ml/results` into the committed `src/data/results.json`, with a freshness test. |
| 3.9 | `f94f39a` | ONNX Runtime WASM self-hosted (the jsDelivr request is gone), phone menu, Playwright e2e against the production build. |

## Numbers worth remembering
- **Name model (browser, AI-drafted sample of 62 sentences / 56 names, not reviewed by the owner):** distilbert-NER 47/56 strict, 40 ms median, 66 MB; bert-base-NER 48/56, 73 ms, 109 MB (within the 2-name tie, so the smaller model won); multilingual 41/56 and over the size limit. **Lowercase names: 1/7 for every model**, the largest known gap.
- **Browser pre-check parity** on the 25 golden prompts: identical token counts, max score difference 9.4e-4 (Node and WASM alike), 0 verdict differences. Browser speed: 22 ms for a short prompt, about 0.9 s per 510-token window, 5.2 s for a 1,920-token document. That's why it never delays sending.
- **Verified on the wire:** the `/chat` body for "Email Rahul Verma at rahul.v@example.in or call 98765 43210 about the PAN ABCPE1234F." was `Email [NAME_1] at [EMAIL_1] or call [PHONE_1] about the PAN [PAN_1].`, and the reply was unmasked in the browser (one real Groq call, made by hand).
- **Tests:** backend 394 (was 372); frontend 113 Vitest (including 61 shared fixture cases, parity, contrast and results freshness); e2e 3/3 against the production build.

## Problems found and fixed along the way
1. **Parity test compared the wrong modes.** The golden fixture is head+tail; the first run compared it with sliding scores (diff 0.99 on long texts). Added a head+tail mode and compared like with like.
2. **Stats test assumed `total >= llm_ms`;** the stub LLM reports a fixed 8 ms. Fixed the test, and `added_ms` is now clamped at 0.
3. **Model status stuck on "Downloading… 23 of 23 MB"** while the model initialised. It now shows "Downloaded. Starting…", and late progress events can't overwrite "ready".
4. **Dashboard "Safe" note was wrong** (it counted all chat messages). Fixed.
5. **Benchmarks tagged non-significant changes as "better"** (13 → 12 false alarms, p = 1.0). Tags now need p < 0.05; otherwise the row says "within noise".
6. **A made-up CI.** My first exporter derived a confidence interval for the served model's F1 from the paired-difference CI. Removed before commit: only the difference CI exists.
7. **The e2e first "passed" against a leftover dev server,** not the build (stopping the background task left Vite running, and Playwright reused it). Stray processes stopped, re-run against the real build: 3/3.
8. **ONNX Runtime loaded from jsDelivr** even in the production build. Now served from `/ort/`.

## Honest limits and open items (also on the How it works page)
- Lowercase names are usually missed; addresses, dates of birth and bank details are not detected.
- The names sample is AI-drafted and not reviewed by the owner, so its recall figures are indicative only.
- The name model is still downloaded from the Hugging Face Hub (it sees the visitor's IP address, never message text).
- `frontend/public/models/maskwall-minilm/` (the 23 MB pre-check model) is gitignored and made by `node scripts/prepare_browser_model.mjs`. **Day 4 must host it** (HF Hub, or copy it during the build). Otherwise the deployed early warning shows "did not load", and the server check still works.
- `dist/` contains the ONNX Runtime WASM twice (21.6 MB each; visitors download only `/ort/`).
- The 12 ms server latency figure is from the previous MiniLM (same architecture); re-benchmark on the Space in Day 4.
- Deployed CORS: set `MASKWALL_CORS_ORIGINS` to the Vercel URL, and `VITE_API_URL` to the Space URL at build time.

## How to run (Windows, from the repo root)
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --app-dir backend --port 8000   # backend
cd frontend
npm install
node scripts/prepare_browser_model.mjs     # once: copy the served model for the in-browser pre-check
npm run dev                                # http://localhost:5173
npm test; npm run lint                     # unit tests + lint/type check
npm run e2e                                # needs the backend running; builds and serves the production build
npm run export-results                     # after any change in ml/results
```
