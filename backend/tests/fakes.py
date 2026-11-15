"""Test doubles for the API tests: a scanner with a scripted score and an LLM that records what it was sent."""
import threading
import time

from app.llm import Attempt, ErrorKind, LLMError, LLMFailure, LLMResult
from app.scanners.injection import ScanResult, WindowResult


class FakeScanner:
    """Scripted scanner: block threshold 0.9, warn threshold 0.7 (thr_f1=0.9, thr_fpr1pct=0.7). Records the texts it saw."""
    thr_f1, thr_fpr1pct = 0.9, 0.7

    def __init__(self, score=0.01, coverage="full", mode="head_tail", raises=None, delay=0.0):
        self.score, self.coverage, self.mode, self.raises, self.delay = score, coverage, mode, raises, delay
        self.calls: list[str] = []
        self._active = self.max_active = 0
        self._lock = threading.Lock()

    def scan(self, text: str) -> ScanResult:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            self.calls.append(text)
            if self.raises:
                raise self.raises
            if self.delay:
                time.sleep(self.delay)
            n = max(len(text.split()), 1)
            return ScanResult(self.score, self.coverage, self.mode, n, 1, [WindowResult(0, n, self.score, 1.5)], 0.2, 1.5, n)
        finally:
            with self._lock:
                self._active -= 1


class StubLLM:
    """Scripted LLM: returns `text` (or raises `fail`). Records the messages it was sent."""

    def __init__(self, text="Here is a helpful answer.", provider="groq", failed_over=False, fail: LLMFailure | None = None,
                 truncated=False, finish_reason="stop"):
        self.text, self.provider, self.failed_over, self.fail = text, provider, failed_over, fail
        self.truncated, self.finish_reason = truncated, ("length" if truncated and finish_reason == "stop" else finish_reason)
        self.calls: list[list[dict]] = []
        self.max_tokens_seen: list[int] = []

    async def complete(self, messages, *, max_tokens=512, temperature=0.2):
        self.calls.append(messages)
        self.max_tokens_seen.append(max_tokens)
        if self.fail:
            raise self.fail
        attempts = [Attempt("groq", False, 3.0, "timeout")] if self.failed_over else []
        attempts.append(Attempt(self.provider, True, 5.0))
        return LLMResult(self.text, self.provider, "stub-model", 8.0, attempts, self.finish_reason, self.truncated)

    async def aclose(self):
        pass


def llm_failure(*kinds):
    attempts = [Attempt(p, False, 2.0, k.value, 503 if k is ErrorKind.SERVER_ERROR else None) for p, k in zip(("groq", "gemini"), kinds)]
    return LLMFailure(attempts, False, LLMError(attempts[-1].provider, kinds[-1]))
