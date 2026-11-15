"""SQLite event log + aggregate statistics for the dashboard.

Privacy by construction: **no prompt, reply or placeholder text is ever stored.** An event holds the verdict, scores, counts,
per-layer latencies and an input fingerprint. `record()` accepts only whitelisted fields, so raw text cannot be logged by accident.

Fingerprint (HMAC-SHA256, truncated to 64 bits): lets you count repeated inputs. Its strength depends on where the key lives:
  * with `secret` (MASKWALL_FINGERPRINT_SECRET): the key is NOT in the database file, so someone who only has the file cannot
    test guesses against it;
  * without it: a random salt is stored in the database itself. That only stops accidental recovery: anyone holding the file can
    recompute fingerprints for guessed prompts (a low-entropy prompt such as "my password is hunter2" can be confirmed offline).
Fingerprints are also deterministic, so equal inputs are linkable.

Operational limits: events older than `retention_days` and anything beyond `max_rows` are pruned; /stats runs aggregate queries in SQL on
a separate read connection (WAL mode) and can cache its result for a few seconds, so dashboard polling cannot block writers.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path

COLUMNS = {
    "request_id": "TEXT", "endpoint": "TEXT", "action": "TEXT", "reasons": "TEXT", "injection_score": "REAL", "coverage": "TEXT",
    "scan_mode": "TEXT", "n_tokens": "INTEGER", "n_windows": "INTEGER", "input_chars": "INTEGER", "input_fingerprint": "TEXT",
    "pii": "TEXT", "output_flagged": "INTEGER", "output_kinds": "TEXT", "provider": "TEXT", "model": "TEXT", "failed_over": "INTEGER",
    "http_status": "INTEGER", "latency_ms": "TEXT", "window_ms": "TEXT", "truncated": "INTEGER",
}
JSON_COLUMNS = {"reasons", "pii", "output_kinds", "latency_ms", "window_ms"}
LAYERS = ("pii_ms", "tokenize_ms", "inference_ms", "llm_ms", "output_ms", "total_ms")
PRUNE_EVERY = 500


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 2)


class EventLog:
    def __init__(self, path: str | Path, *, secret: str | None = None, retention_days: float | None = 90, max_rows: int | None = 500_000):
        self.path = str(path)
        self.retention_days, self.max_rows = retention_days, max_rows
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            cols = ", ".join(f"{k} {v}" for k, v in COLUMNS.items())
            self._db.execute(f"CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, {cols})")
            have = {r["name"] for r in self._db.execute("PRAGMA table_info(events)")}
            for name, typ in COLUMNS.items():          # migrate databases created before a column existed
                if name not in have:
                    self._db.execute(f"ALTER TABLE events ADD COLUMN {name} {typ}")
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            row = self._db.execute("SELECT value FROM meta WHERE key='salt'").fetchone()
            if row is None:
                self._db.execute("INSERT INTO meta(key, value) VALUES ('salt', ?)", (secrets.token_hex(16),))
                row = self._db.execute("SELECT value FROM meta WHERE key='salt'").fetchone()
            self._db.commit()
        # the HMAC key: an external secret if given (not stored anywhere in the file), otherwise the salt kept in the database
        self.keyed_externally = bool(secret)
        self._key = hashlib.sha256(secret.encode("utf-8")).digest() if secret else bytes.fromhex(row["value"])
        # a separate read-only connection for statistics (WAL lets readers run while a writer writes)
        if self.path == ":memory:":
            self._read, self._read_lock = self._db, self._lock
        else:
            self._read = sqlite3.connect(self.path, check_same_thread=False)
            self._read.row_factory = sqlite3.Row
            self._read.execute("PRAGMA query_only=ON")
            self._read_lock = threading.Lock()
        self._inserts = 0
        self._cache: tuple[float, dict] | None = None
        self.prune()

    def fingerprint(self, text: str) -> str:
        return hmac.new(self._key, text.encode("utf-8"), hashlib.sha256).hexdigest()[:16]

    # ---------------------------------------------------------------- writing and retention
    def record(self, event: dict, *, ts: float | None = None) -> int:
        unknown = set(event) - set(COLUMNS)
        if unknown:
            raise ValueError(f"unknown event fields {sorted(unknown)}: only {sorted(COLUMNS)} may be logged (never raw text)")
        values = {k: (json.dumps(v, separators=(",", ":")) if k in JSON_COLUMNS and v is not None else v) for k, v in event.items()}
        names = ["ts", *values]
        with self._lock:
            cur = self._db.execute(f"INSERT INTO events ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})", [ts or time.time(), *values.values()])
            self._db.commit()
            self._inserts += 1
            rowid = int(cur.lastrowid)
        if self._inserts % PRUNE_EVERY == 0:
            self.prune()
        return rowid

    def prune(self) -> int:
        """Delete events older than retention_days and, if still above max_rows, the oldest ones. Returns rows deleted."""
        deleted = 0
        with self._lock:
            if self.retention_days:
                deleted += self._db.execute("DELETE FROM events WHERE ts < ?", (time.time() - self.retention_days * 86400,)).rowcount
            if self.max_rows:
                total = self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                if total > self.max_rows:
                    deleted += self._db.execute("DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY id LIMIT ?)", (total - self.max_rows,)).rowcount
            self._db.commit()
        return deleted

    # ---------------------------------------------------------------- statistics
    def stats(self, *, latency_sample: int = 1000, hours: int = 24, ttl: float = 0.0) -> dict:
        """Aggregates computed in SQL. With ttl > 0 a result younger than ttl seconds is returned from cache (dashboard polling)."""
        if ttl > 0 and self._cache and time.time() - self._cache[0] < ttl:
            return self._cache[1]
        with self._read_lock:
            q = self._read.execute
            total = q("SELECT COUNT(*) FROM events").fetchone()[0]
            by_action = {r["action"]: r["n"] for r in q("SELECT action, COUNT(*) AS n FROM events GROUP BY action")}
            by_endpoint = {r["endpoint"]: r["n"] for r in q("SELECT endpoint, COUNT(*) AS n FROM events GROUP BY endpoint")}
            providers = {r["provider"]: r["n"] for r in q("SELECT provider, COUNT(*) AS n FROM events WHERE provider IS NOT NULL GROUP BY provider")}
            failovers = q("SELECT COUNT(*) FROM events WHERE failed_over = 1").fetchone()[0]
            out_flagged = q("SELECT COUNT(*) FROM events WHERE output_flagged = 1").fetchone()[0]
            truncated = q("SELECT COUNT(*) FROM events WHERE truncated = 1").fetchone()[0]
            pii = {r["k"]: int(r["n"]) for r in q(
                "SELECT j.key AS k, SUM(j.value) AS n FROM events, json_each(events.pii) AS j WHERE events.pii IS NOT NULL AND events.pii != '{}' GROUP BY j.key")}
            lat_rows = [json.loads(r["latency_ms"]) for r in q("SELECT latency_ms FROM events WHERE latency_ms IS NOT NULL ORDER BY id DESC LIMIT ?", (latency_sample,))]
            bins = [0] * 10
            for r in q("SELECT MIN(CAST(injection_score * 10 AS INTEGER), 9) AS b, COUNT(*) AS n FROM events WHERE injection_score IS NOT NULL GROUP BY b"):
                bins[r["b"]] = r["n"]
            since = time.time() - hours * 3600
            hourly: dict[int, dict] = {}
            for r in q("SELECT CAST(ts / 3600 AS INTEGER) AS h, action, COUNT(*) AS n FROM events WHERE ts >= ? GROUP BY h, action", (since,)):
                hourly.setdefault(r["h"] * 3600, {})[r["action"]] = r["n"]
            first = q("SELECT MIN(ts) FROM events").fetchone()[0]
        latency = {}
        for layer in LAYERS:
            vals = [float(r[layer]) for r in lat_rows if r.get(layer) is not None]
            latency[layer] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 2) if vals else None, "p50": _pct(vals, 0.5), "p95": _pct(vals, 0.95)}
        result = {
            "total_requests": total,
            "attacks_blocked": by_action.get("block", 0),
            "outputs_blocked": by_action.get("output_blocked", 0),
            "warnings": by_action.get("warn", 0),
            "by_action": by_action, "by_endpoint": by_endpoint,
            "pii_caught_server_side": pii, "pii_total": sum(pii.values()),
            "output_flagged": out_flagged, "truncated_replies": truncated,
            "llm": {"providers": providers, "failovers": failovers},
            "latency_ms": latency,
            "injection_score_histogram": {"bins": [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)], "counts": bins},
            "hourly": [{"hour_start": h, **a} for h, a in sorted(hourly.items())],
            "first_event_ts": first,
        }
        self._cache = (time.time(), result)
        return result

    def close(self) -> None:
        with self._lock:
            self._db.close()
        if self._read is not self._db:
            with self._read_lock:
                self._read.close()
