import sqlite3
import threading
import time

import pytest

from app.logging_db import COLUMNS, EventLog


@pytest.fixture
def log(tmp_path):
    db = EventLog(tmp_path / "events.sqlite")
    yield db
    db.close()


def ev(**kw):
    base = {"request_id": "r1", "endpoint": "/chat", "action": "allow", "injection_score": 0.1, "coverage": "full", "n_tokens": 12,
            "n_windows": 1, "input_chars": 40, "pii": {}, "latency_ms": {"inference_ms": 10.0, "total_ms": 20.0}}
    base.update(kw)
    return base


def test_empty_database_stats(log):
    s = log.stats()
    assert s["total_requests"] == 0 and s["attacks_blocked"] == 0 and s["pii_total"] == 0 and s["hourly"] == []
    assert s["latency_ms"]["total_ms"] == {"n": 0, "mean": None, "p50": None, "p95": None}


def test_record_and_count_by_action(log):
    for action in ["allow"] * 5 + ["block"] * 3 + ["warn"] * 2 + ["output_blocked"]:
        log.record(ev(action=action))
    s = log.stats()
    assert s["total_requests"] == 11 and s["attacks_blocked"] == 3 and s["warnings"] == 2 and s["outputs_blocked"] == 1
    assert s["by_action"] == {"allow": 5, "block": 3, "warn": 2, "output_blocked": 1}


def test_pii_counts_are_summed_by_type(log):
    log.record(ev(pii={"AADHAAR": 1, "EMAIL": 2}))
    log.record(ev(pii={"EMAIL": 1, "PHONE": 3}))
    log.record(ev(pii={}))
    s = log.stats()
    assert s["pii_caught_server_side"] == {"AADHAAR": 1, "EMAIL": 3, "PHONE": 3} and s["pii_total"] == 7


def test_latency_percentiles_per_layer(log):
    for i in range(1, 101):
        log.record(ev(latency_ms={"inference_ms": float(i), "total_ms": float(i * 2)}))
    s = log.stats()["latency_ms"]
    assert s["inference_ms"]["n"] == 100 and s["inference_ms"]["mean"] == 50.5
    assert s["inference_ms"]["p50"] == 50.5 and s["inference_ms"]["p95"] == 95.05
    assert s["total_ms"]["p50"] == 101.0
    assert s["llm_ms"]["n"] == 0                                    # layers never recorded stay empty


def test_providers_failovers_and_output_flags(log):
    log.record(ev(provider="groq", failed_over=0))
    log.record(ev(provider="gemini", failed_over=1))
    log.record(ev(provider="gemini", failed_over=1, output_flagged=1, output_kinds=["canary"]))
    s = log.stats()
    assert s["llm"] == {"providers": {"groq": 1, "gemini": 2}, "failovers": 2} and s["output_flagged"] == 1


def test_score_histogram_and_hourly_series(log):
    now = time.time()
    for score in (0.05, 0.15, 0.95, 1.0):
        log.record(ev(injection_score=score), ts=now)
    log.record(ev(injection_score=None, action="block"), ts=now - 2 * 3600)
    log.record(ev(action="allow"), ts=now - 400 * 3600)               # older than the 24h window
    s = log.stats()
    counts = s["injection_score_histogram"]["counts"]                      # 0.05 -> bin 0; 0.15 and the fixture default 0.1 -> bin 1; 0.95 and 1.0 -> bin 9
    assert counts[0] == 1 and counts[1] == 2 and counts[9] == 2 and sum(counts) == 5
    assert sum(sum(v for k, v in h.items() if k != "hour_start") for h in s["hourly"]) == 5
    assert s["hourly"] == sorted(s["hourly"], key=lambda h: h["hour_start"])


# ------------------------------------------------------------------ privacy
def test_only_whitelisted_fields_can_be_logged(log):
    for bad in ["text", "message", "prompt", "response", "masked_text", "ip", "user_agent"]:
        with pytest.raises(ValueError):
            log.record(ev(**{bad: "some raw text"}))
    assert log.stats()["total_requests"] == 0


def test_schema_has_no_text_columns(log):
    cols = {r[1] for r in sqlite3.connect(log.path).execute("PRAGMA table_info(events)")}
    assert cols == {"id", "ts", *COLUMNS}
    assert not cols & {"text", "message", "prompt", "response", "content", "ip"}


def test_fingerprint_is_stable_salted_and_not_a_plain_hash(tmp_path):
    a, b = EventLog(tmp_path / "a.sqlite"), EventLog(tmp_path / "b.sqlite")
    assert a.fingerprint("hello") == a.fingerprint("hello") and a.fingerprint("hello") != a.fingerprint("hellp")
    assert a.fingerprint("hello") != b.fingerprint("hello")                 # different salt per database
    import hashlib
    assert a.fingerprint("hello") != hashlib.sha256(b"hello").hexdigest()[:16]
    a.close(); b.close()


def test_salt_persists_across_reopen(tmp_path):
    p = tmp_path / "e.sqlite"
    first = EventLog(p)
    fp = first.fingerprint("same input")
    first.record(ev())
    first.close()
    second = EventLog(p)
    assert second.fingerprint("same input") == fp and second.stats()["total_requests"] == 1
    second.close()


def test_raw_text_does_not_appear_in_the_database_file(tmp_path):
    p = tmp_path / "e.sqlite"
    db = EventLog(p)
    secret = "my-very-secret-prompt-text-12345"
    db.record(ev(input_fingerprint=db.fingerprint(secret), input_chars=len(secret)))
    db.close()
    assert secret.encode() not in p.read_bytes()


# ------------------------------------------------------------------ robustness
def test_concurrent_writers(log):
    def worker(n):
        for _ in range(50):
            log.record(ev(request_id=f"t{n}"))
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert log.stats()["total_requests"] == 400


def test_in_memory_database_works():
    db = EventLog(":memory:")
    db.record(ev())
    assert db.stats()["total_requests"] == 1
    db.close()
