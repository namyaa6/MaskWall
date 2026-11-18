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
    a.close()
    b.close()


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


def test_old_databases_are_migrated_with_new_columns(tmp_path):
    p = tmp_path / "old.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT INTO meta VALUES ('salt', '00112233445566778899aabbccddeeff')")
    con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, request_id TEXT, endpoint TEXT, action TEXT)")
    con.execute("INSERT INTO events (ts, request_id, endpoint, action) VALUES (?, 'old', '/chat', 'allow')", (time.time(),))
    con.commit()
    con.close()
    db = EventLog(p)                                    # must add every missing column, keep the old row
    db.record(ev(truncated=1))
    s = db.stats()
    assert s["total_requests"] == 2 and s["truncated_replies"] == 1
    db.close()


def test_truncated_count(log):
    log.record(ev(truncated=1))
    log.record(ev(truncated=0))
    log.record(ev())
    assert log.stats()["truncated_replies"] == 1


# ------------------------------------------------------------------ retention, fingerprint key, stats cost (review findings)
def test_events_older_than_retention_are_pruned_at_startup_and_periodically(tmp_path):
    p = tmp_path / "r.sqlite"
    db = EventLog(p, retention_days=30)
    now = time.time()
    db.record(ev(request_id="old"), ts=now - 40 * 86400)
    db.record(ev(request_id="new"), ts=now - 1 * 86400)
    assert db.prune() == 1 and db.stats()["total_requests"] == 1
    db.record(ev(request_id="old2"), ts=now - 99 * 86400)
    db.close()
    assert EventLog(p, retention_days=30).stats()["total_requests"] == 1        # pruned again when the app starts


def test_row_cap_keeps_only_the_newest_rows(tmp_path):
    db = EventLog(tmp_path / "c.sqlite", max_rows=100, retention_days=None)
    for i in range(250):
        db.record(ev(request_id=f"r{i}"))
    db.prune()
    ids = [r["request_id"] for r in db._db.execute("SELECT request_id FROM events ORDER BY id")]
    assert len(ids) == 100 and ids[0] == "r150" and ids[-1] == "r249"


def test_pruning_runs_by_itself_every_500_inserts(tmp_path):
    db = EventLog(tmp_path / "a.sqlite", max_rows=300, retention_days=None)
    for i in range(1000):
        db.record(ev())
    assert db.stats()["total_requests"] <= 300 + 500                  # bounded without an explicit prune() call


def test_an_external_secret_keeps_the_key_out_of_the_database_file(tmp_path):
    p = tmp_path / "k.sqlite"
    secret = "super-secret-fingerprint-key-9f3a"
    db = EventLog(p, secret=secret)
    fp = db.fingerprint("my password is hunter2")
    db.record(ev(input_fingerprint=fp))
    assert db.keyed_externally is True
    db.close()
    assert secret.encode() not in p.read_bytes()
    # someone who only has the file recomputes with the stored salt: it does NOT match
    con = sqlite3.connect(p)
    salt = bytes.fromhex(con.execute("SELECT value FROM meta WHERE key='salt'").fetchone()[0])
    import hashlib
    import hmac
    assert hmac.new(salt, b"my password is hunter2", hashlib.sha256).hexdigest()[:16] != fp
    con.close()
    assert EventLog(p, secret=secret).fingerprint("my password is hunter2") == fp                # same secret: stable across restarts
    assert EventLog(p, secret="another secret").fingerprint("my password is hunter2") != fp


def test_without_a_secret_the_salt_in_the_file_is_enough_to_test_guesses_and_the_log_says_so(tmp_path):
    """Documented limitation: this is why MASKWALL_FINGERPRINT_SECRET exists."""
    p = tmp_path / "s.sqlite"
    db = EventLog(p)
    fp = db.fingerprint("my password is hunter2")
    assert db.keyed_externally is False
    db.close()
    con = sqlite3.connect(p)
    salt = bytes.fromhex(con.execute("SELECT value FROM meta WHERE key='salt'").fetchone()[0])
    import hashlib
    import hmac
    assert hmac.new(salt, b"my password is hunter2", hashlib.sha256).hexdigest()[:16] == fp
    import app.logging_db as module
    assert "anyone holding the file can recompute fingerprints" in " ".join(module.__doc__.split())               # the limitation is stated in the module docs


def test_stats_result_is_cached_for_ttl_seconds_and_refreshed_after(tmp_path):
    db = EventLog(tmp_path / "t.sqlite")
    db.record(ev())
    first = db.stats(ttl=0.3)
    db.record(ev())
    assert db.stats(ttl=0.3) is first and first["total_requests"] == 1       # served from cache
    assert db.stats()["total_requests"] == 2                                   # ttl=0 always recomputes
    time.sleep(0.35)
    assert db.stats(ttl=0.3)["total_requests"] == 2


def test_stats_on_a_large_table_is_fast_and_does_not_block_writers(tmp_path):
    db = EventLog(tmp_path / "big.sqlite", retention_days=None, max_rows=None)
    now = time.time()
    rows = [(now - i, "r", "/chat", "allow" if i % 5 else "block", 0.1, '{"EMAIL":1}', '{"inference_ms":3.0,"total_ms":5.0}') for i in range(100_000)]
    with db._lock:
        db._db.executemany("INSERT INTO events (ts, request_id, endpoint, action, injection_score, pii, latency_ms) VALUES (?,?,?,?,?,?,?)", rows)
        db._db.commit()
    t = time.perf_counter()
    s = db.stats()
    elapsed = time.perf_counter() - t
    assert s["total_requests"] == 100_000 and s["pii_caught_server_side"] == {"EMAIL": 100_000} and elapsed < 3.0, elapsed
    # a write during a long read is not blocked by the statistics lock (separate read connection, WAL)
    done = []
    reader = threading.Thread(target=lambda: [db.stats() for _ in range(3)])
    reader.start()
    t = time.perf_counter()
    for _ in range(20):
        db.record(ev())
    done.append(time.perf_counter() - t)
    reader.join()
    assert done[0] < 2.0, done
