import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.abuse import client_key
from app.config import Settings
from app.logging_db import EventLog
from app.main import create_app
from tests.fakes import FakeScanner, StubLLM


def make(tmp_path, **kw):
    s = Settings(db_path=tmp_path / "e.sqlite", model_dir=tmp_path / "no_model", groq_api_key="k1", gemini_api_key="k2", **kw)
    db = EventLog(":memory:")
    scanner = FakeScanner()
    return create_app(s, scanner=scanner, llm=StubLLM(), db=db), scanner, db


# ------------------------------------------------------------------ body size cap (before any parsing)
def test_declared_content_length_over_the_cap_is_rejected_before_parsing(tmp_path):
    app, scanner, _ = make(tmp_path, max_chars=100)               # cap = 100 * 12 + 4096 = 5,296 bytes
    with TestClient(app) as c:
        r = c.post("/chat", content=b"x" * 6000, headers={"content-type": "application/json"})
        assert r.status_code == 413 and r.json()["detail"] == {"error": "body_too_large", "max_bytes": 5296}
        assert c.get("/stats").json()["since_start"]["too_large"] == 1
    assert scanner.calls == []


def test_a_chunked_body_without_content_length_is_capped_too(tmp_path):
    app, scanner, _ = make(tmp_path, max_chars=100)

    def chunks():
        for _ in range(100):
            yield b"y" * 100                                       # 10,000 bytes streamed, no Content-Length

    with TestClient(app) as c:
        r = c.post("/chat", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413 and scanner.calls == []


def test_the_worst_case_legitimate_body_still_passes_the_cap(tmp_path):
    app, scanner, _ = make(tmp_path, max_chars=8000, rate_limit="1000/minute")
    text = "\U0001F600" * 8000                                      # 8,000 emoji: JSON-escaped as surrogate pairs = 12 bytes each
    body = json.dumps({"message": text})                           # ensure_ascii escaping (what many clients send)
    assert len(body.encode()) > 8000 * 11
    with TestClient(app) as c:
        r = c.post("/chat", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 200 and len(scanner.calls) == 1


def test_normal_requests_are_unaffected_and_get_replayed_intact(tmp_path):
    app, scanner, _ = make(tmp_path)
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "hello there"})
    assert r.status_code == 200 and scanner.calls == ["hello there"]


# ------------------------------------------------------------------ global limit applies to invalid requests too (review finding)
def test_a_flood_of_invalid_requests_is_rate_limited(tmp_path):
    app, scanner, _ = make(tmp_path, global_rate_limit="5/minute", rate_limit="1000/minute")
    with TestClient(app) as c:
        codes = [c.post("/chat", json={"message": ""}).status_code for _ in range(8)]       # all fail validation (422)
        r = c.post("/chat", json={"message": "valid"})
    assert codes == [422] * 5 + [429] * 3 and r.status_code == 429
    assert r.headers["retry-after"].isdigit() and r.json()["detail"]["scope"] == "global"
    assert scanner.calls == []
    assert app.state.counters["rate_limited"] == 4


def test_health_is_exempt_from_the_global_limit(tmp_path):
    app, *_ = make(tmp_path, global_rate_limit="2/minute")
    with TestClient(app) as c:
        assert all(c.get("/health").status_code == 200 for _ in range(30))
        assert [c.get("/stats").status_code for _ in range(3)] == [200, 200, 429]


def test_429_responses_still_carry_cors_headers(tmp_path):
    app, *_ = make(tmp_path, global_rate_limit="1/minute", cors_origins=("https://ui.example",))
    with TestClient(app) as c:
        c.get("/stats", headers={"Origin": "https://ui.example"})
        r = c.get("/stats", headers={"Origin": "https://ui.example"})
    assert r.status_code == 429 and r.headers["access-control-allow-origin"] == "https://ui.example"


def test_flooding_never_writes_to_the_event_log(tmp_path):
    app, _, db = make(tmp_path, global_rate_limit="3/minute", rate_limit="1000/minute")
    with TestClient(app) as c:
        for _ in range(50):
            c.post("/scan", json={"text": "hello"})
    assert db.stats()["total_requests"] == 3 and app.state.counters["rate_limited"] == 47


# ------------------------------------------------------------------ client key
def test_client_key_logic():
    h = {"x-forwarded-for": "1.1.1.1, 2.2.2.2"}
    assert client_key(h, ("9.9.9.9", 1), trust_proxy=False) == "9.9.9.9"          # header ignored by default
    assert client_key(h, ("9.9.9.9", 1), trust_proxy=True) == "2.2.2.2"           # last entry, never the forgeable first
    assert client_key({}, ("9.9.9.9", 1), trust_proxy=True) == "9.9.9.9"          # no header: the connection address
    assert client_key({}, None, trust_proxy=False) == "unknown"


def test_trust_proxy_and_missing_fingerprint_secret_log_warnings(tmp_path, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="maskwall")
    s = Settings(db_path=tmp_path / "w.sqlite", model_dir=tmp_path / "no_model", trust_proxy=True)
    with TestClient(create_app(s, scanner=FakeScanner(), llm=StubLLM())) as c:                  # the app builds (and owns) its event log
        c.get("/health")
    text = caplog.text
    assert "MASKWALL_TRUST_PROXY is on" in text and "MASKWALL_FINGERPRINT_SECRET is not set" in text


def test_with_a_fingerprint_secret_there_is_no_warning_about_it(tmp_path, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="maskwall")
    s = Settings(db_path=tmp_path / "w.sqlite", model_dir=tmp_path / "no_model", fingerprint_secret="s3cret")
    with TestClient(create_app(s, scanner=FakeScanner(), llm=StubLLM())) as c:
        c.get("/health")
        assert c.app.state.db.keyed_externally is True
    assert "FINGERPRINT_SECRET is not set" not in caplog.text
