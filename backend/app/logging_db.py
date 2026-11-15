"""SQLite event log + aggregate statistics for the dashboard.

Privacy by construction: **no prompt, reply or placeholder text is ever stored.** An event holds the verdict, scores, counts,
per-layer latencies and an input fingerprint (HMAC-SHA256 with a random per-database salt, truncated: it lets you count repeated
inputs without being reversible by guessing prompts). `record()` accepts only whitelisted fields, so raw text cannot be logged
by accident.
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


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 2)


class EventLog:
    def __init__(self, path: str | Path):
        self.path = str(path)
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
            self._salt = bytes.fromhex(row["value"])
            self._db.commit()

    def fingerprint(self, text: str) -> str:
        return hmac.new(self._salt, text.encode("utf-8"), hashlib.sha256).hexdigest()[:16]

    def record(self, event: dict, *, ts: float | None = None) -> int:
        unknown = set(event) - set(COLUMNS)
        if unknown:
            raise ValueError(f"unknown event fields {sorted(unknown)}: only {sorted(COLUMNS)} may be logged (never raw text)")
        values = {k: (json.dumps(v, separators=(",", ":")) if k in JSON_COLUMNS and v is not None else v) for k, v in event.items()}
        names = ["ts", *values]
        with self._lock:
            cur = self._db.execute(f"INSERT INTO events ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})", [ts or time.time(), *values.values()])
            self._db.commit()
            return int(cur.lastrowid)

    # ---------------------------------------------------------------- statistics
    def stats(self, *, latency_sample: int = 1000, hours: int = 24) -> dict:
        with self._lock:
            q = self._db.execute
            total = q("SELECT COUNT(*) FROM events").fetchone()[0]
            by_action = {r["action"]: r["n"] for r in q("SELECT action, COUNT(*) AS n FROM events GROUP BY action")}
            by_endpoint = {r["endpoint"]: r["n"] for r in q("SELECT endpoint, COUNT(*) AS n FROM events GROUP BY endpoint")}
            providers = {r["provider"]: r["n"] for r in q("SELECT provider, COUNT(*) AS n FROM events WHERE provider IS NOT NULL GROUP BY provider")}
            failovers = q("SELECT COUNT(*) FROM events WHERE failed_over = 1").fetchone()[0]
            out_flagged = q("SELECT COUNT(*) FROM events WHERE output_flagged = 1").fetchone()[0]
            truncated = q("SELECT COUNT(*) FROM events WHERE truncated = 1").fetchone()[0]
            pii_rows = [json.loads(r["pii"]) for r in q("SELECT pii FROM events WHERE pii IS NOT NULL AND pii != '{}'")]
            lat_rows = [json.loads(r["latency_ms"]) for r in q("SELECT latency_ms FROM events WHERE latency_ms IS NOT NULL ORDER BY id DESC LIMIT ?", (latency_sample,))]
            scores = [r["injection_score"] for r in q("SELECT injection_score FROM events WHERE injection_score IS NOT NULL")]
            since = time.time() - hours * 3600
            hourly = {}
            for r in q("SELECT CAST(ts / 3600 AS INTEGER) AS h, action, COUNT(*) AS n FROM events WHERE ts >= ? GROUP BY h, action", (since,)):
                hourly.setdefault(r["h"] * 3600, {})[r["action"]] = r["n"]
            first = q("SELECT MIN(ts) FROM events").fetchone()[0]
        pii: dict[str, int] = {}
        for row in pii_rows:
            for k, v in row.items():
                pii[k] = pii.get(k, 0) + int(v)
        latency = {}
        for layer in LAYERS:
            vals = [float(r[layer]) for r in lat_rows if r.get(layer) is not None]
            latency[layer] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 2) if vals else None, "p50": _pct(vals, 0.5), "p95": _pct(vals, 0.95)}
        bins = [0] * 10
        for s in scores:
            bins[min(int(s * 10), 9)] += 1
        return {
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

    def close(self) -> None:
        with self._lock:
            self._db.close()
