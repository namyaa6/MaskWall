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
    assert run(groq(rec).complete(MSGS, max_tokens=77, temperature=0.3)).text == "hi there"
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
    assert run(gemini(rec).complete(msgs, max_tokens=55, temperature=0.1)).text == "gem answer"
    req = rec.requests[0]
    assert req.url.path.endswith("/models/gemini-model:generateContent")
    assert req.headers["x-goog-api-key"] == GEMINI_KEY and GEMINI_KEY not in str(req.url)
    body = json.loads(req.content)
    assert body["system_instruction"]["parts"][0]["text"] == "SYSTEM PROMPT TEXT"
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["generationConfig"] == {"maxOutputTokens": 55, "temperature": 0.1}


def test_gemini_joins_parts_and_handles_blocked_or_empty_candidates():
    parts = {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}
    assert run(gemini(Recorder(resp(200, parts))).complete(MSGS)).text == "ab"
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


# ------------------------------------------------------------------ finish reasons and truncation
def groq_body(text, finish):
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}]}


def gemini_body(text, finish):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}]}


@pytest.mark.parametrize("finish,truncated", [("stop", False), ("length", True), ("content_filter", False), ("tool_calls", False), (None, False)])
def test_groq_finish_reason_and_truncation(finish, truncated):
    c = run(groq(Recorder(resp(200, groq_body("a reply", finish)))).complete(MSGS))
    assert c.text == "a reply" and c.finish_reason == finish and c.truncated is truncated


@pytest.mark.parametrize("finish,truncated", [("STOP", False), ("MAX_TOKENS", True), ("SAFETY", False), ("OTHER", False), (None, False)])
def test_gemini_finish_reason_and_truncation(finish, truncated):
    c = run(gemini(Recorder(resp(200, gemini_body("a reply", finish)))).complete(MSGS))
    assert c.text == "a reply" and c.finish_reason == finish and c.truncated is truncated


def test_a_missing_or_non_string_finish_reason_is_treated_as_unknown_not_truncated():
    body = {"choices": [{"message": {"content": "x"}, "finish_reason": 7}]}
    c = run(groq(Recorder(resp(200, body))).complete(MSGS))
    assert c.finish_reason is None and c.truncated is False


def test_reasoning_budget_exhausted_before_any_output_is_a_bad_response_that_fails_over():
    # reasoning models count hidden reasoning against max_tokens: finish_reason "length" with empty content
    with pytest.raises(LLMError) as e:
        run(groq(Recorder(resp(200, groq_body("", "length")))).complete(MSGS))
    assert e.value.kind is ErrorKind.BAD_RESPONSE and "token limit" in e.value.detail and e.value.failover_worthy
    with pytest.raises(LLMError) as e2:
        run(gemini(Recorder(resp(200, {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}))).complete(MSGS))
    assert e2.value.kind is ErrorKind.BAD_RESPONSE and "token limit" in e2.value.detail


def test_failover_result_carries_the_serving_providers_finish_reason():
    g, m = Recorder(httpx.ReadTimeout("slow")), Recorder(resp(200, gemini_body("cut off mid-sent", "MAX_TOKENS")))
    r = run(FailoverLLM([groq(g), gemini(m)]).complete(MSGS))
    assert r.provider == "gemini" and r.truncated is True and r.finish_reason == "MAX_TOKENS" and r.failed_over
    r2 = run(FailoverLLM([groq(Recorder(resp(200, groq_body("full answer", "stop"))))]).complete(MSGS))
    assert r2.truncated is False and r2.finish_reason == "stop"


def test_the_default_token_limit_is_2048_and_is_sent_to_the_provider():
    rec = Recorder(resp(200, groq_body("ok", "stop")))
    run(groq(rec).complete(MSGS))
    assert json.loads(rec.requests[0].content)["max_tokens"] == 2048
    rec2 = Recorder(resp(200, gemini_body("ok", "STOP")))
    run(gemini(rec2).complete(MSGS))
    assert json.loads(rec2.requests[0].content)["generationConfig"]["maxOutputTokens"] == 2048


# ------------------------------------------------------------------ exceptions the review found escaping failover
@pytest.mark.parametrize("exc", [
    httpx.TooManyRedirects("loop"), httpx.DecodingError("bad gzip"), httpx.UnsupportedProtocol("ftp"), httpx.RemoteProtocolError("bad"),
    httpx.ProxyError("proxy"), httpx.ReadError("r"), httpx.WriteError("w"), httpx.LocalProtocolError("l"),
])
def test_any_httpx_request_error_becomes_a_network_error_and_fails_over(exc):
    with pytest.raises(LLMError) as e:
        run(groq(Recorder(exc)).complete(MSGS))
    assert e.value.kind is ErrorKind.NETWORK and e.value.failover_worthy
    llm, g, m = pair(exc)
    r = run(llm.complete(MSGS))
    assert r.provider == "gemini" and r.attempts[0].kind == "network"


class ExplodingClient:
    name, model = "exploding", "m"

    async def complete(self, messages, *, max_tokens, temperature):
        raise RuntimeError("bug inside a client with SECRET USER QUESTION and " + GROQ_KEY)

    async def aclose(self):
        pass


def test_an_unexpected_exception_from_one_client_moves_on_to_the_next_without_leaking_it():
    m = Recorder(resp(200, gemini_ok()))
    r = run(FailoverLLM([ExplodingClient(), gemini(m)]).complete(MSGS))
    assert r.provider == "gemini" and [(a.provider, a.ok, a.kind) for a in r.attempts] == [("exploding", False, "bad_response"), ("gemini", True, None)]
    with pytest.raises(LLMFailure) as e:
        run(FailoverLLM([ExplodingClient()]).complete(MSGS))
    blob = str(e.value) + repr(e.value.attempts) + repr(e.value.last.__dict__)
    assert GROQ_KEY not in blob and "SECRET USER QUESTION" not in blob and "bug inside" not in blob       # the exception text is never kept


def test_cancellation_is_not_swallowed_by_failover():
    class Cancelling:
        name, model = "c", "m"

        async def complete(self, messages, *, max_tokens, temperature):
            raise asyncio.CancelledError()

        async def aclose(self):
            pass

    with pytest.raises(asyncio.CancelledError):
        run(FailoverLLM([Cancelling(), gemini(Recorder(resp(200, gemini_ok())))]).complete(MSGS))


# ------------------------------------------------------------------ no key in ANY error, for every kind and both providers (mutation survivor)
ERROR_SCENARIOS = [
    httpx.ReadTimeout("t"), httpx.ConnectTimeout("t"), httpx.ConnectError("c"), httpx.TooManyRedirects("r"), httpx.DecodingError("d"),
    resp(429, {"error": {"type": "rate_limit_exceeded", "message": "key " + GROQ_KEY + " and " + GEMINI_KEY + " SECRET USER QUESTION"}}),
    resp(500, {"error": {"message": GROQ_KEY}}), resp(503), resp(400, {"error": {"type": "invalid_request_error", "message": GEMINI_KEY}}),
    resp(401, {"error": {"status": "UNAUTHENTICATED", "message": GROQ_KEY}}), resp(404), resp(200, {"garbage": GROQ_KEY}),
    resp(200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}),
]


@pytest.mark.parametrize("scenario", ERROR_SCENARIOS, ids=lambda s: type(s).__name__ + (str(getattr(s, "status_code", "")) if hasattr(s, "status_code") else ""))
@pytest.mark.parametrize("factory", [groq, gemini], ids=["groq", "gemini"])
def test_no_error_ever_carries_a_key_or_prompt_text(factory, scenario):
    secrets = (GROQ_KEY, GEMINI_KEY, "SECRET USER QUESTION", "SYSTEM PROMPT TEXT")
    with pytest.raises(LLMError) as e:
        run(factory(Recorder(scenario)).complete(MSGS))
    blob = str(e.value) + repr(e.value.__dict__) + repr(e.value.args)
    assert not any(s in blob for s in secrets), blob
    g, m = Recorder(scenario), Recorder(scenario)
    with pytest.raises(LLMFailure) as f:
        run(FailoverLLM([groq(g), gemini(m)]).complete(MSGS))
    blob = str(f.value) + repr(f.value.attempts) + repr(f.value.last.__dict__)
    assert not any(s in blob for s in secrets), blob
