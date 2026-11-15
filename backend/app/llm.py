"""LLM layer: Groq (primary) and Gemini (fallback) behind one interface, with failover.

Both providers are called with plain `httpx` (no vendor SDKs: smaller image, and the transport can be mocked in tests).
Failover happens only on: timeouts, network errors, rate limits (HTTP 429), server errors (HTTP 5xx) and malformed
responses. Other client errors (400, 401, 403, 404, ...) stop the chain: they indicate a bad request or bad credentials,
which a different provider would not fix.

Errors never carry API keys, request headers or prompt text: only the provider, a kind, the HTTP status and, where the provider
sends one, a short error type string.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import httpx

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class ErrorKind(str, Enum):
    TIMEOUT = "timeout"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    BAD_RESPONSE = "bad_response"
    CLIENT_ERROR = "client_error"


FAILOVER_KINDS = frozenset({ErrorKind.TIMEOUT, ErrorKind.NETWORK, ErrorKind.RATE_LIMIT, ErrorKind.SERVER_ERROR, ErrorKind.BAD_RESPONSE})


class LLMError(Exception):
    def __init__(self, provider: str, kind: ErrorKind, status: int | None = None, detail: str = "", retry_after: float | None = None):
        self.provider, self.kind, self.status, self.detail, self.retry_after = provider, kind, status, detail, retry_after
        super().__init__(f"{provider}: {kind.value}" + (f" (HTTP {status})" if status else "") + (f": {detail}" if detail else ""))

    @property
    def failover_worthy(self) -> bool:
        return self.kind in FAILOVER_KINDS


@dataclass(frozen=True)
class Completion:
    """What one provider returned: the text and why generation stopped. `truncated` means the reply hit the token limit."""
    text: str
    finish_reason: str | None = None
    truncated: bool = False


@dataclass(frozen=True)
class Attempt:
    provider: str
    ok: bool
    ms: float
    kind: str | None = None
    status: int | None = None


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: float
    attempts: list[Attempt] = field(default_factory=list)
    finish_reason: str | None = None
    truncated: bool = False

    @property
    def failed_over(self) -> bool:
        return len(self.attempts) > 1


class LLMFailure(Exception):
    """No provider produced an answer (all failed, or a non-failover error stopped the chain)."""

    def __init__(self, attempts: list[Attempt], stopped_early: bool, last: LLMError):
        self.attempts, self.stopped_early, self.last = attempts, stopped_early, last
        super().__init__("; ".join(f"{a.provider}:{a.kind}" + (f"({a.status})" if a.status else "") for a in attempts if not a.ok) or "no providers")


class LLMClient(Protocol):
    name: str
    model: str

    async def complete(self, messages: list[dict], *, max_tokens: int, temperature: float) -> Completion: ...
    async def aclose(self) -> None: ...


def _short_error_type(resp: httpx.Response) -> str:
    """The provider's own error type/status string if there is one (e.g. 'invalid_request_error'); never the body."""
    try:
        err = resp.json().get("error")
    except Exception:
        return ""
    if isinstance(err, dict):
        for k in ("type", "status", "code"):
            v = err.get(k)
            if isinstance(v, str) and v.replace("_", "").replace("-", "").isalnum():
                return v[:60]
    return ""


def _check_status(provider: str, resp: httpx.Response) -> None:
    s = resp.status_code
    if 200 <= s < 300:
        return
    ra = None
    try:
        ra = float(resp.headers.get("retry-after", ""))
    except ValueError:
        pass
    detail = _short_error_type(resp)
    if s == 429:
        raise LLMError(provider, ErrorKind.RATE_LIMIT, s, detail, ra)
    if s == 408:
        raise LLMError(provider, ErrorKind.TIMEOUT, s, detail)
    if s >= 500:
        raise LLMError(provider, ErrorKind.SERVER_ERROR, s, detail)
    raise LLMError(provider, ErrorKind.CLIENT_ERROR, s, detail)


class _HTTPClient:
    name = ""

    def __init__(self, api_key: str, model: str, *, timeout_s: float = 20.0, transport: httpx.AsyncBaseTransport | None = None, base_url: str = ""):
        self.model = model
        self._key = api_key
        self._http = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_s, connect=min(5.0, timeout_s)), transport=transport)

    async def _post(self, url: str, headers: dict, payload: dict) -> httpx.Response:
        try:
            return await self._http.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as e:
            raise LLMError(self.name, ErrorKind.TIMEOUT, detail=type(e).__name__) from None
        except httpx.TransportError as e:
            raise LLMError(self.name, ErrorKind.NETWORK, detail=type(e).__name__) from None

    async def aclose(self) -> None:
        await self._http.aclose()


class GroqClient(_HTTPClient):
    name = "groq"

    def __init__(self, api_key: str, model: str, **kw):
        super().__init__(api_key, model, base_url=kw.pop("base_url", GROQ_BASE_URL), **kw)

    async def complete(self, messages: list[dict], *, max_tokens: int = 2048, temperature: float = 0.2) -> Completion:
        resp = await self._post("/chat/completions", {"Authorization": f"Bearer {self._key}"},
                                {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
        _check_status(self.name, resp)
        try:
            choice = resp.json()["choices"][0]
            text = choice["message"]["content"]
        except Exception:
            raise LLMError(self.name, ErrorKind.BAD_RESPONSE, resp.status_code, "unparseable response") from None
        finish = choice.get("finish_reason")
        finish = finish if isinstance(finish, str) else None
        if not isinstance(text, str) or not text.strip():
            # reasoning models spend max_tokens on hidden reasoning first: "length" with no text = budget gone before any output
            raise LLMError(self.name, ErrorKind.BAD_RESPONSE, resp.status_code,
                           "empty response (token limit reached)" if finish == "length" else "empty response")
        return Completion(text, finish, truncated=finish == "length")


class GeminiClient(_HTTPClient):
    name = "gemini"

    def __init__(self, api_key: str, model: str, **kw):
        super().__init__(api_key, model, base_url=kw.pop("base_url", GEMINI_BASE_URL), **kw)

    @staticmethod
    def _convert(messages: list[dict]) -> tuple[str | None, list[dict]]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
        return system, contents

    async def complete(self, messages: list[dict], *, max_tokens: int = 2048, temperature: float = 0.2) -> Completion:
        system, contents = self._convert(messages)
        payload: dict = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}}
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}
        # the key travels in a header, never in the URL (URLs end up in logs and exception messages)
        resp = await self._post(f"/models/{self.model}:generateContent", {"x-goog-api-key": self._key}, payload)
        _check_status(self.name, resp)
        try:
            cand = resp.json()["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"]["parts"])
        except Exception:
            raise LLMError(self.name, ErrorKind.BAD_RESPONSE, resp.status_code, "no candidate (blocked or empty)") from None
        finish = cand.get("finishReason")
        finish = finish if isinstance(finish, str) else None
        if not text.strip():
            raise LLMError(self.name, ErrorKind.BAD_RESPONSE, resp.status_code,
                           "empty response (token limit reached)" if finish == "MAX_TOKENS" else "empty response")
        return Completion(text, finish, truncated=finish == "MAX_TOKENS")


class FailoverLLM:
    """Try the providers in order; fail over on timeouts, network errors, rate limits, 5xx and malformed responses."""

    def __init__(self, clients: list[LLMClient]):
        if not clients:
            raise ValueError("FailoverLLM needs at least one provider")
        self.clients = clients

    @property
    def providers(self) -> list[str]:
        return [c.name for c in self.clients]

    async def complete(self, messages: list[dict], *, max_tokens: int = 2048, temperature: float = 0.2) -> LLMResult:
        attempts: list[Attempt] = []
        t_all = time.perf_counter()
        last: LLMError | None = None
        for client in self.clients:
            t0 = time.perf_counter()
            try:
                comp = await client.complete(messages, max_tokens=max_tokens, temperature=temperature)
            except LLMError as e:
                attempts.append(Attempt(client.name, False, (time.perf_counter() - t0) * 1000, e.kind.value, e.status))
                last = e
                if not e.failover_worthy:
                    raise LLMFailure(attempts, True, e) from None
                continue
            attempts.append(Attempt(client.name, True, (time.perf_counter() - t0) * 1000))
            return LLMResult(comp.text, client.name, client.model, (time.perf_counter() - t_all) * 1000, attempts,
                             comp.finish_reason, comp.truncated)
        assert last is not None
        raise LLMFailure(attempts, False, last)

    async def aclose(self) -> None:
        for c in self.clients:
            await c.aclose()


def build_llm(settings, *, transports: dict | None = None) -> FailoverLLM | None:
    """FailoverLLM from the settings: providers in `llm_order` that have an API key; None if none is configured."""
    transports = transports or {}
    clients: list[LLMClient] = []
    for name in settings.llm_order:
        key = settings.provider_key(name)
        if not key:
            continue
        cls, model = (GroqClient, settings.groq_model) if name == "groq" else (GeminiClient, settings.gemini_model)
        clients.append(cls(key, model, timeout_s=settings.llm_timeout_s, transport=transports.get(name)))
    return FailoverLLM(clients) if clients else None
