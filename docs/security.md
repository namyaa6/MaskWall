# Security and hygiene review (Day 4, 2026-11-18)

## 1. Secrets in the repository and its history
- **Working tree and all reachable commits:** searched for key-shaped strings (`gsk_`, `AIza`, `hf_`, `sk-`, PEM private-key headers, `KEY/SECRET/TOKEN = <24+ chars>`). No real credential found. The only hit is a test fixture string `-----BEGIN RSA PRIVATE KEY-----` in `backend/tests/test_output_scanner.py` (a header, used to test the output scanner).
- **`.env` was committed once.** The very first commit (`43764c4`, 2026-11-12) contained `.env` with a `GROQ_API_KEY` line. It was removed by amending and force-pushing, so the file is not in the current history. The old commit still exists locally as an unreachable object and, because the repository is public, **must be treated as exposed** (a clone, a fork or GitHub's cache taken in that window could still hold it).
  - What the value was: a 13-character string starting with "you…" and without the `gsk_` prefix of real Groq keys, so it looks like a placeholder (for example `your_key_here`). I looked at its length and prefix only; the value is not reproduced anywhere.
  - **Action: the owner confirms that the Groq key in use today was created after that date or has been rotated** (rotate at console.groq.com, then update the local `.env` and the Space secret). Rotation is the only real remedy; history rewriting cannot recall copies.
  - Status: *awaiting the owner's confirmation.*
- `.env`, model binaries and `backend/data/` are gitignored. Keys are read from the environment on the server and are excluded from `repr()` of the settings object; deployed secrets live only in the Space settings.

## 2. `npm audit` (frontend), 21 findings at the start, 17 after patch updates; no `--force`
Patch updates applied (exact-pinned, tests, lint and build re-run green): `react-router-dom` 6.30.1 to 6.30.6, `typescript-eslint` 8.46.4 to 8.73.0, `vite` 6.4.1 to 6.4.6, `vitest` 3.2.4 to 3.2.7; `sharp` overridden to 0.35.5 (patched).

The audit database still reports the items below, because their fix is a **major** version change (or the advisory range has no in-range fix). Disposition for each:

| Finding | In production bundle? | Disposition |
|---|---|---|
| `@huggingface/transformers` and `sharp` (libvips, libheif and librsvg CVEs); audit says fix = Transformers.js 4.x (major) | **No.** `sharp` is a Node-only image library; the browser bundle never contains it, and the site never decodes images. | The patched `sharp` 0.35.5 is forced by `overrides`, so what is installed is patched; the audit still flags it because it compares against the parent's declared range. Not upgrading Transformers.js to 4.x: a major change to the in-browser model layer, to be evaluated separately with the parity tests. |
| `react-router` / `react-router-dom` (moderate: open redirect through a backslash in `<Link>`/`navigate` with untrusted input); fix = v7 (major) | Yes (client routing). | Not exploitable here: every `<Link>` and `navigate()` target is a hard-coded internal path; no user-supplied or URL-derived string is ever passed to them. v7 migration is not worth the risk for this. |
| `vitest`, `@vitest/mocker`, `tinypool` (critical/moderate: prototype pollution / file read when the Vitest UI server is running, or via mock registration) | No (dev dependency). | Not exploitable: tests run with `vitest run` in CI and locally, the Vitest UI/API server is never started, and nothing from the test runner is deployed. Fix needs vitest 5 (major). |
| `vite`, `fdir`, `picomatch`, `tinyglobby` (ReDoS; Vite dev-server file read) | No: build-time only; the dev server is never exposed (Vercel serves the built `dist/`). | The 6.4.6 patch was applied. The remaining `picomatch` ReDoS has no in-range fix; it needs attacker-controlled glob patterns, and we pass only our own. Re-check when Vite 7 is adopted. |
| `typescript-eslint` family, `fast-glob`, `micromatch`, `minimatch` (ReDoS / globparts) | No (lint tooling). | Same reasoning: only our own config globs are matched. |

**Standing rule:** `npm audit --omit=dev` is reviewed before each release; findings in the shipped bundle are fixed, others are explained here. CI runs `npm audit --omit=dev --audit-level=high` as an informational step (it does not block, because the two production-side findings above are explained, not fixable without a major upgrade).

## 3. Other items (tracked in later Day 4 steps)
- Deployed CORS limited to the production origin; proxy-hop count measured on the real Space before trusting `X-Forwarded-For`.
- A global daily cap on LLM calls so public traffic cannot exhaust the free provider quotas.
- Final independent review (security and claims-versus-evidence) recorded in `docs/decisions.md`.
