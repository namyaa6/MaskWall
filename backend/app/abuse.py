"""Abuse guard: an ASGI middleware that runs BEFORE routing and request parsing.

  1. A global per-client rate limit on every request, including ones that later fail validation (a flood of invalid bodies is otherwise
     free: slowapi's per-route limits only run once a request has already been parsed and validated). /health is exempt.
  2. A request-body size cap, checked before any JSON parsing: via Content-Length when present, and by buffering at most
     `max_body_bytes` when the body is chunked. Over the cap: HTTP 413.

Both rejections are counted in memory (`counters`), not written to the event log: a flood must not be able to grow the database.
The client key is shared with the per-route limiter (see client_key).
"""
from __future__ import annotations

import json
import time

from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter


def client_key(headers, client, trust_proxy: bool) -> str:
    """Rate-limit key. X-Forwarded-For is client-controlled unless a trusted proxy sets it, so it is ignored by default;
    with trust_proxy the LAST entry (appended by the one trusted proxy) is used, never the first (forgeable).
    trust_proxy is only safe when the app is reachable ONLY through that proxy: if clients can connect directly, they can send any value."""
    if trust_proxy:
        xff = headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[-1].strip()
    return client[0] if client else "unknown"


async def _respond(send, status: int, body: dict, headers: list[tuple[bytes, bytes]] | None = None) -> None:
    payload = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode()), *(headers or [])]})
    await send({"type": "http.response.body", "body": payload})


class AbuseGuard:
    def __init__(self, app, *, rate_limit: str, max_body_bytes: int, trust_proxy: bool, counters: dict, exempt: tuple[str, ...] = ("/health",)):
        self.app, self.max_body_bytes, self.trust_proxy, self.counters, self.exempt = app, max_body_bytes, trust_proxy, counters, exempt
        self.item = parse(rate_limit)
        self.limiter = FixedWindowRateLimiter(MemoryStorage())

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        if scope["path"] not in self.exempt:
            key = client_key(headers, scope.get("client"), self.trust_proxy)
            if not self.limiter.hit(self.item, key):
                self.counters["rate_limited"] += 1
                retry = max(1, int(self.limiter.get_window_stats(self.item, key).reset_time - time.time()) + 1)
                return await _respond(send, 429, {"detail": {"error": "rate_limited", "scope": "global"}}, [(b"retry-after", str(retry).encode())])
        declared = headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            self.counters["too_large"] += 1
            return await _respond(send, 413, {"detail": {"error": "body_too_large", "max_bytes": self.max_body_bytes}})
        if scope["method"] in ("POST", "PUT", "PATCH"):
            chunks, total = [], 0
            while True:                                           # bounded buffering: never more than max_body_bytes + one chunk
                msg = await receive()
                if msg["type"] != "http.request":
                    return await self.app(scope, _replay([msg]), send)
                chunks.append(msg.get("body", b""))
                total += len(chunks[-1])
                if total > self.max_body_bytes:
                    self.counters["too_large"] += 1
                    return await _respond(send, 413, {"detail": {"error": "body_too_large", "max_bytes": self.max_body_bytes}})
                if not msg.get("more_body", False):
                    break
            body = b"".join(chunks)
            return await self.app(scope, _replay([{"type": "http.request", "body": body, "more_body": False}], receive), send)
        return await self.app(scope, receive, send)


def _replay(messages: list[dict], fallback=None):
    queue = list(messages)

    async def receive():
        if queue:
            return queue.pop(0)
        if fallback is not None:
            return await fallback()
        return {"type": "http.disconnect"}

    return receive
