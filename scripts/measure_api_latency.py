"""Sporadic-traffic latency of the DEPLOYED firewall API (what a visitor typically experiences), via POST /scan.

    python scripts/measure_api_latency.py https://maskwall-api.onrender.com --out ml/results/render_api_latency.json

Requests are SPACED (default 3 s for short prompts, more for long ones) so the free tier's CPU quota is not exhausted by back-to-back
calls; that is the opposite of backend/bench/bench_service.py, which measures the back-to-back (throttled) worst case. Spacing also keeps
the run under the per-client rate limit (30/minute on /scan). No LLM is called. Reports, for each case, the server-measured numbers
returned by the API (`latency_ms`: total, inference) and the client-measured round trip (network included: from the developer machine to
the host's region).
Hardware label: Render free tier (0.1 vCPU nominal; cgroup quota measured 0.15 vCPU), region Singapore.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "bench"))
from bench_service import FILLER, SHORT  # noqa: E402  (the same fixed prompts as the back-to-back benchmark)


def post(base: str, text: str) -> tuple[dict, float]:
    req = urllib.request.Request(f"{base}/scan", data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read())
    return body, (time.perf_counter() - t) * 1000


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run_case(base: str, name: str, texts: list[str], gap_s: float) -> dict:
    rtt, total, infer, tokens, windows = [], [], [], [], []
    for t in texts:
        body, ms = post(base, t)
        rtt.append(ms)
        total.append(body["latency_ms"]["total_ms"])
        infer.append(body["latency_ms"]["inference_ms"])
        tokens.append(body["input"]["n_tokens"])
        windows.append(body["input"]["n_windows"])
        time.sleep(gap_s)
    f = lambda xs: {"p50_ms": round(pct(xs, 0.5), 1), "p95_ms": round(pct(xs, 0.95), 1), "max_ms": round(max(xs), 1)}      # noqa: E731
    out = {"case": name, "n": len(texts), "gap_s": gap_s, "server_total": f(total), "server_inference": f(infer), "client_round_trip": f(rtt),
           "n_tokens_median": statistics.median(tokens), "n_windows_median": statistics.median(windows)}
    print(name, json.dumps({k: out[k] for k in ("server_total", "server_inference", "client_round_trip", "n_tokens_median", "n_windows_median")}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--out", default="render_api_latency.json")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    health = json.loads(urllib.request.urlopen(f"{base}/health", timeout=120).read())
    assert health["model_loaded"], health
    # calibrate a text that is one full window (<= 505 tokens)
    n_chars = 2600
    while True:
        text = (FILLER * 20)[:n_chars]
        body, _ = post(base, text)
        if body["input"]["n_tokens"] <= 505 or n_chars < 400:
            break
        n_chars = int(n_chars * 0.9)
        time.sleep(2)
    one_window = text
    long_doc = (FILLER * 40)[:7000]
    results = [
        run_case(base, "short prompt (20 fixed prompts)", SHORT, 3.0),
        run_case(base, "one window (~500 tokens)", [one_window] * 6, 5.0),
        run_case(base, "long document (7,000 characters)", [long_doc] * 4, 8.0),
    ]
    out = {"measured": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "host": base, "health": health,
           "hardware": "Render free tier (0.1 vCPU nominal; cgroup quota 0.15 vCPU), Singapore", "client": "developer machine (India), plain HTTPS",
           "note": "spaced requests (sporadic traffic), POST /scan, no LLM; server_* are measured inside the API, client_round_trip includes the network",
           "cases": results}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("written", a.out)


if __name__ == "__main__":
    main()
