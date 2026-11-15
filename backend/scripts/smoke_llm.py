"""Real smoke test: one benign /chat call through each LLM provider (Groq, then Gemini), through the full pipeline
(PII mask -> injection scan -> policy -> LLM -> output scan). Not part of the pytest suite (it needs network and keys).

    python backend/scripts/smoke_llm.py [--model-dir backend/model]

API keys are read from the environment / repo-root .env by Settings and are NEVER printed or logged: the script prints only the
provider, verdict, status, latency, a short slice of the model's reply and error kinds, and at the end verifies that none of
its own output contains a key. If a provider rejects the configured model id (HTTP 404), it lists that provider's model ids
(ids only) and retries once with a plausible one.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

PROMPT = "Reply with just the single word: pong"
PREFER = {
    "groq": ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
    "gemini": ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite"],
}


def list_model_ids(provider: str, key: str) -> list[str]:
    if provider == "groq":
        r = httpx.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
        return [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
    r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models", headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=20)
    if r.status_code != 200:
        return []
    return [m["name"].split("/", 1)[1] for m in r.json().get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]


def call(provider: str, base: Settings, model_dir: Path, model: str | None = None):
    kw = dict(llm_order=(provider,), db_path=Path(ROOT / "backend" / "data" / "smoke.sqlite"), model_dir=model_dir, rate_limit="100/minute",
              groq_api_key=base.groq_api_key, gemini_api_key=base.gemini_api_key, groq_model=base.groq_model, gemini_model=base.gemini_model)
    if model:
        kw["groq_model" if provider == "groq" else "gemini_model"] = model
    with TestClient(create_app(Settings(**kw))) as c:
        r = c.post("/chat", json={"message": PROMPT})
    return r, kw["groq_model" if provider == "groq" else "gemini_model"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ROOT / "backend" / "model")
    args = ap.parse_args()
    base = Settings.from_env()
    out = io.StringIO()
    ok_all = True
    with redirect_stdout(out):
        for provider in ("groq", "gemini"):
            key = base.provider_key(provider)
            if not key:
                print(f"[{provider}] SKIPPED: no API key configured")
                ok_all = False
                continue
            r, model = call(provider, base, args.model_dir)
            if r.status_code == 502 and any(a.get("kind") == "client_error" and a.get("status") == 404 for a in r.json().get("detail", {}).get("attempts", [])):
                ids = list_model_ids(provider, key)
                pick = next((m for m in PREFER[provider] if m in ids), None)
                print(f"[{provider}] model id {model!r} rejected (404); {len(ids)} models listed; retrying with {pick!r}")
                if pick:
                    r, model = call(provider, base, args.model_dir, pick)
            if r.status_code != 200:
                print(f"[{provider}] FAILED: HTTP {r.status_code} {str(r.json().get('detail'))[:300]}")
                ok_all = False
                continue
            b = r.json()
            print(f"[{provider}] OK  model={b['llm']['model']}  action={b['action']}  injection_score={b['input']['score']}  "
                  f"failed_over={b['llm']['failed_over']}  llm_ms={b['latency_ms']['llm_ms']}  total_ms={b['latency_ms']['total_ms']}")
            print(f"[{provider}] reply ({len(b['response'])} chars): {b['response'][:80]!r}")
            ok_all &= b["action"] == "allow" and b["llm"]["provider"] == provider and not b["llm"]["failed_over"] and bool(b["response"].strip())
    text = out.getvalue()
    leaked = [n for n, k in (("GROQ_API_KEY", base.groq_api_key), ("GEMINI_API_KEY", base.gemini_api_key)) if k and k in text]
    print(text, end="")
    print("key-leak check on this output:", "FAILED (" + ", ".join(leaked) + ")" if leaked else "passed (no key appears in the output above)")
    return 0 if ok_all and not leaked else 1


if __name__ == "__main__":
    sys.exit(main())
