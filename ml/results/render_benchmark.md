# Hosting benchmark: Render free tier (0.1 vCPU nominal, 512 MB) vs developer laptop

**Hardware label for every number below: "Render free tier (0.1 vCPU)".** Measured on 2026-11-19 on the throwaway service `maskwall-bench` (same Dockerfile base, same lock file, same model). What the container reported about itself: AMD EPYC 7R13, `cgroup cpu.max = 15000 100000`, which is **0.15 vCPU of quota**, not 0.1 (Render documents 0.1; the container's own limit says 0.15; both are "a small fraction of one core"), memory limit 512 MB.
Protocol (fixed before any number was seen, `backend/bench/bench_service.py`): the real FastAPI app loaded in-process, 2 ORT threads, batch 1, scanner as served (sliding windows, max 6), timed unit = `InjectionScanner.scan()` (tokenize + inference). Raw files: `bench_render.json` (second run), `bench_laptop.json`.

## Memory (the owner's gate: "if 512 MB is not enough, stop")
| | Render free tier (0.1 vCPU) | laptop |
|---|---|---|
| whole API process after start (RSS / working set) | 141.8 MB | 129.4 MB |
| peak after a 4-window 7,000-character document (RSS) | 222.1 to 223.5 MB (two runs) | 208.3 MB |
| container (cgroup) peak, whole run | 190.6 MB (run 1), 226.8 MB (run 2) | n/a |
| limit | 512 MB | n/a |
**Verdict: MiniLM as served fits with about 285 MB spare (peak is 37 to 44% of the limit). No feature has to be removed for memory.**
DeBERTa (not served): loading it into the same container made the request fail with a 502 and the service restarted, consistent with an out-of-memory kill (alone on the laptop it peaks at 408 MB; next to the 190 MB process it exceeds 512 MB). Render's Events tab is the authority on the cause. It does not matter for the product (DeBERTa is benchmark-only), but it means **DeBERTa latency on Render could not be measured**; only the laptop figures exist for it.

## Latency, MiniLM, `scan()` = tokenize + inference
| case | Render free tier (0.1 vCPU), run 1 / run 2 | laptop |
|---|---|---|
| short prompt, 300 back-to-back calls: p50 / p95 | 89.1 / 98.5 ms; 91.9 / 98.6 ms | 2.8 / 4.3 ms |
| inference only, 40 calls: p50 / p95 | 3.1 / 97.8 ms; 49.3 / 98.2 ms | 2.8 / 4.1 ms |
| one 480-token window, n=56 to 57: p50 / p95 | 1,594.8 / 1,801.2 ms; 1,597.7 / 1,898.3 ms | 127.8 / 172.7 ms |
| 7,000 characters = 4 windows, n=14: p50 / p95 | 6,749.9 / 7,036.5 ms; 6,905.2 / 7,165.9 ms | 551.8 / 588.3 ms |
| `POST /scan` in-process, 50 calls: p50 / p95 | 107.3 / 200.2 ms; 191.4 / 200.3 ms | 10.7 / 17.4 ms |

**How to read it.** (1) The distribution is **bimodal**, the signature of CPU-quota throttling: a task that needs about 3 ms either runs inside the quota (about 3 ms) or waits for the next 100 ms accounting period (about 90 to 100 ms). That is why the inference-only p50 moved between 3.1 and 49.3 ms across two runs while p95 stayed at about 98 ms; **the p50 of tiny calls is not a stable number on this hardware**, and back-to-back calls (this protocol) are the throttling worst case. Sporadic requests (a demo visitor typing a message) can be much faster; that has **not** been measured here and is measured separately on the real API (`render_api_latency.*`, Day 4 step 6). (2) Throughput-bound work shows the real compute ratio: a full window is about **12x** slower than on the laptop (1.6 s vs 0.13 s), a 4-window document about **12.5x** (6.9 s vs 0.55 s).

## The 50 ms rule (Day 1.11: "p50 CPU latency at batch size 1 under 50 ms"), reported without changing it
- The rule was defined for `session.run` on natural-length validation prompts (private, not available on the host). The closest thing measured here: **back-to-back `scan()` on short prompts has p50 about 90 ms on Render's free tier: above 50 ms. By that measurement MiniLM does NOT meet the rule on this hardware.** The inference-only p50 (3.1 or 49.3 ms across two runs) is unstable, so it supports neither a pass nor a fail.
- The rule's purpose was to choose between models on development hardware; it is **not** re-applied to change the choice. MiniLM stays the served model (DeBERTa is 4 to 5x slower on the laptop, so it would not pass either, and does not fit next to the app in 512 MB).
- Consequence for users, stated plainly: a short message costs roughly 0.1 s of scanning when requests are back-to-back, a message of about 500 tokens about 1.6 s, and a long document (4+ windows) about 7 s, plus the LLM time. The Attack playground and Chat show server-measured timings, so users can see it.
