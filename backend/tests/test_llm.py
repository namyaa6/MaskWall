import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.llm import ErrorKind, FailoverLLM, GeminiClient, GroqClient, LLMError, LLMFailure, build_llm

GROQ_KEY, GEMINI_KEY = "test-groq-key-ABC123", "test-gemini-key-XYZ789"
MSGS = [{"role": "system", "content": "SYSTEM PROMPT TEXT"}, {"role": "user", "content": "SECRET USER QUESTION"}]


def run(coro):
    return asyncio.run(coro)


def groq_ok(text="hello from groq"):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def gemini_ok(text="hello from gemini"):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


class Recorder:
    """A mock transport that records requests and answers with a scripted response or exception."""

    def __init__(self, action):
        self.action, self.requests = action, []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request):
        self.requests.append(request)
        a = self.action(request) if callable(self.action) else self.action
        if isinstance(a, Exception):
            raise a
        return a

    @property
    def calls(self):
        return len(self.requests)


def resp(status=200, body=None, headers=None):
    return httpx.Response(status, json=body if body is not None else {}, headers=headers)


def groq(rec, **kw):
    return GroqClient(GROQ_KEY, "groq-model", transport=rec.transport, **kw)


def gemini(rec, **kw):
    return GeminiClient(GEMINI_KEY, "gemini-model", transport=rec.transport, **kw)


# ------------------------------------------------------------------ Groq client
def test_groq_request_and_response():
    rec = Recorder(resp(200, groq_ok("hi there")))
    assert run(groq(rec).complete(MSGS, max_tokens=77, temperature=0.3)) == "hi there"
    req = rec.requests[0]
    assert req.url.path.endswith("/chat/completions") and req.headers["authorization"] == f"Bearer {GROQ_KEY}"
    body = json.loads(req.content)
    assert body["model"] == "groq-model" and body["max_tokens"] == 77 and body["temperature"] == 0.3 and body["messages"] == MSGS
    assert GROQ_KEY not in str(req.url)


@pytest.mark.parametrize("status,kind", [
    (429, ErrorKind.RATE_LIMIT), (408, ErrorKind.TIMEOUT), (500, ErrorKind.SERVER_ERROR), (502, ErrorKind.SERVER_ERROR),
    (503, ErrorKind.SERVER_ERROR), (529, ErrorKind.SERVER_ERROR), (400, ErrorKind.CLIENT_ERROR), (401, ErrorKind.CLIENT_ERROR),
    (403, ErrorKind.CLIENT_ERROR), (404, ErrorKind.CLIENT_ERROR),
])
def test_groq_status_codes_map_to_error_kinds(status, kind):
    with pytest.raises(LLMError) as e:
        run(groq(Recorder(resp(status, {"error": {"type": "some_error"}}))).complete(MSGS))
    assert e.value.kind is kind and e.value.status == status and e.value.provider == "groq"
    assert e.value.failover_worthy == (kind is not ErrorKind.CLIENT_ERROR)


@pytest.mark.parametrize("exc,kind", [
    (httpx.ReadTimeout("t"), ErrorKind.TIMEOUT), (httpx.ConnectTimeout("t"), ErrorKind.TIMEOUT), (httpx.PoolTimeout("t"), ErrorKind.TIMEOUT),
    (httpx.ConnectError("c"), ErrorKind.NETWORK), (httpx.ReadError("r"), ErrorKind.NETWORK),
])
def test_groq_transport_exceptions_map_to_error_kinds(exc, kind):
    with pytest.raises(LLMError) as e:
        run(groq(Recorder(exc)).complete(MSGS))
    assert e.value.kind is kind and e.value.failover_worthy


@pytest.mark.parametrize("body", [{"nope": 1}, {"choices": []}, {"choices": [{"message": {"content": ""}}]}, {"choices": [{"message": {"content": "   "}}]}])
def test_groq_malformed_or_empty_200_is_a_bad_response(body):
    with pytest.raises(LLMError) as e:
        run(groq(Recorder(resp(200, body))).complete(MSGS))
    assert e.value.kind is ErrorKind.BAD_RESPONSE and e.value.failover_worthy


def test_groq_retry_after_is_parsed():
    with pytest.raises(LLMError) as e:
        run(groq(Recorder(resp(429, {}, {"retry-after": "7"}))).complete(MSGS))
    assert e.value.retry_after == 7.0


def test_errors_never_contain_keys_or_prompt_text_even_if_the_provider_echoes_them():
    echo = {"error": {"type": "invalid_request_error", "message": f"bad request for SECRET USER QUESTION using {GROQ_KEY}"}}
    with pytest.raises(LLMError) as e:
        run(groq(Recorder(resp(400, echo))).complete(MSGS))
    text = str(e.value) + repr(e.value.__dict__)
    assert GROQ_KEY not in text and "SECRET USER QUESTION" not in text and "invalid_request_error" in text


# ------------------------------------------------------------------ Gemini client
def test_gemini_request_format_and_key_in_header_not_url():
    rec = Recorder(resp(200, gemini_ok("gem answer")))
    msgs = [*MSGS, {"role": "assistant", "content": "earlier reply"}, {"role": "user", "content": "follow up"}]
    assert run(gemini(rec).complete(msgs, max_tokens=55, temperature=0.1)) == "gem answer"
    req = rec.requests[0]
    assert req.url.path.endswith("/models/gemini-model:generateContent")
    assert req.headers["x-goog-api-key"] == GEMINI_KEY and GEMINI_KEY not in str(req.url)
    body = json.loads(req.content)
    assert body["system_instruction"]["parts"][0]["text"] == "SYSTEM PROMPT TEXT"
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["generationConfig"] == {"maxOutputTokens": 55, "temperature": 0.1}


def test_gemini_joins_parts_and_handles_blocked_or_empty_candidates():
    parts = {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}
    assert run(gemini(Recorder(resp(200, parts))).complete(MSGS)) == "ab"
    for body in [{"promptFeedback": {"blockReason": "SAFETY"}}, {"candidates": []}, {"candidates": [{"finishReason": "SAFETY"}]},
                 {"candidates": [{"content": {"parts": [{"text": " "}]}}]}]:
        with pytest.raises(LLMError) as e:
            run(gemini(Recorder(resp(200, body))).complete(MSGS))
        assert e.value.kind is ErrorKind.BAD_RESPONSE


@pytest.mark.parametrize("status,kind", [(429, ErrorKind.RATE_LIMIT), (503, ErrorKind.SERVER_ERROR), (400, ErrorKind.CLIENT_ERROR), (403, ErrorKind.CLIENT_ERROR)])
def test_gemini_status_mapping(status, kind):
    with pytest.raises(LLMError) as e:
        run(gemini(Recorder(resp(status, {"error": {"status": "RESOURCE_EXHAUSTED"}}))).complete(MSGS))
    assert e.value.kind is kind and e.value.provider == "gemini"


# ------------------------------------------------------------------ failover: one test per path
def pair(groq_action, gemini_action=None):
    g, m = Recorder(groq_action), Recorder(gemini_action if gemini_action is not None else resp(200, gemini_ok()))
    return FailoverLLM([groq(g), gemini(m)]), g, m


def test_primary_success_never_calls_the_fallback():
    llm, g, m = pair(resp(200, groq_ok()))
    r = run(llm.complete(MSGS))
    assert (r.provider, r.model, r.text, g.calls, m.calls) == ("groq", "groq-model", "hello from groq", 1, 0)
    assert len(r.attempts) == 1 and r.attempts[0].ok and not r.failed_over and r.latency_ms >= 0


@pytest.mark.parametrize("failure,kind", [
    (httpx.ReadTimeout("slow"), "timeout"), (httpx.ConnectTimeout("slow"), "timeout"),
    (resp(500), "server_error"), (resp(502), "server_error"), (resp(503), "server_error"),
    (resp(429, {}, {"retry-after": "30"}), "rate_limit"),
    (httpx.ConnectError("down"), "network"),
    (resp(200, {"garbage": True}), "bad_response"),
])
def test_failover_on_timeout_5xx_rate_limit_network_and_bad_response(failure, kind):
    llm, g, m = pair(failure)
    r = run(llm.complete(MSGS))
    assert r.provider == "gemini" and r.text == "hello from gemini" and r.failed_over
    assert g.calls == 1 and m.calls == 1
    assert [(a.provider, a.ok, a.kind) for a in r.attempts] == [("groq", False, kind), ("gemini", True, None)]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_do_not_fail_over(status):
    llm, g, m = pair(resp(status, {"error": {"type": "invalid_request_error"}}))
    with pytest.raises(LLMFailure) as e:
        run(llm.complete(MSGS))
    assert e.value.stopped_early and e.value.last.kind is ErrorKind.CLIENT_ERROR and g.calls == 1 and m.calls == 0


def test_all_providers_failing_raises_with_every_attempt_recorded():
    llm, g, m = pair(httpx.ReadTimeout("slow"), resp(503))
    with pytest.raises(LLMFailure) as e:
        run(llm.complete(MSGS))
    f = e.value
    assert not f.stopped_early and g.calls == 1 and m.calls == 1
    assert [(a.provider, a.kind, a.status) for a in f.attempts] == [("groq", "timeout", None), ("gemini", "server_error", 503)]
    assert GROQ_KEY not in str(f) and GEMINI_KEY not in str(f) and "SECRET USER QUESTION" not in str(f)


def test_fallback_client_error_after_primary_failure_is_reported_not_retried():
    llm, g, m = pair(resp(503), resp(401, {"error": {"status": "UNAUTHENTICATED"}}))
    with pytest.raises(LLMFailure) as e:
        run(llm.complete(MSGS))
    assert e.value.stopped_early and e.value.attempts[-1].provider == "gemini" and e.value.attempts[-1].status == 401


def test_provider_order_is_configurable():
    g, m = Recorder(resp(200, groq_ok())), Recorder(resp(200, gemini_ok()))
    r = run(FailoverLLM([gemini(m), groq(g)]).complete(MSGS))
    assert r.provider == "gemini" and g.calls == 0 and m.calls == 1


def test_failover_llm_requires_a_provider():
    with pytest.raises(ValueError):
        FailoverLLM([])


# ------------------------------------------------------------------ building from settings
def test_build_llm_uses_only_configured_providers_in_order(tmp_path):
    s = Settings(db_path=tmp_path / "e", groq_api_key=GROQ_KEY, gemini_api_key=GEMINI_KEY, llm_order=("gemini", "groq"))
    assert build_llm(s).providers == ["gemini", "groq"]
    assert build_llm(Settings(db_path=tmp_path / "e", groq_api_key=None, gemini_api_key=GEMINI_KEY)).providers == ["gemini"]
    assert build_llm(Settings(db_path=tmp_path / "e", groq_api_key=None, gemini_api_key=None)) is None


def test_build_llm_accepts_mock_transports(tmp_path):
    rec = Recorder(resp(200, groq_ok("via mock")))
    s = Settings(db_path=tmp_path / "e", groq_api_key=GROQ_KEY, gemini_api_key=None)
    assert run(build_llm(s, transports={"groq": rec.transport}).complete(MSGS)).text == "via mock"
