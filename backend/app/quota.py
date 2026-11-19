"""Daily caps on LLM calls, so public traffic cannot use up the owner's free Groq/Gemini quota.

Two caps, both counting only messages that actually reach the LLM (blocked attacks, rejected inputs and /scan never count, so the Attack
playground keeps working when a cap is reached):
  * DailyLimit: ONE global counter for the whole app (MASKWALL_DAILY_LLM_LIMIT).
  * PerClientDailyLimit: a smaller cap per client key (MASKWALL_DAILY_LLM_LIMIT_PER_CLIENT), so a single visitor cannot use the whole global
    budget (at the 30/minute rate limit one client could otherwise drain 500 slots in about 17 minutes and lock everyone out for the day).

The day is the UTC calendar day. Counters live in MEMORY, so any restart resets them: on Render's free tier that means every sleep/wake
(after about 15 minutes idle) and every deploy, so "500 per day" really means "500 per awake period, at most per UTC day". Persisting the
counters would need an external store (the free instance's disk is ephemeral); that is a known limit, stated on the site. A slot is taken
BEFORE the provider call and is not refunded if the call fails, because a failed call may still have used provider quota.
`limit <= 0` disables a cap.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

DAY_S = 86400

CLIENT_LIMIT_MESSAGE = (
    "You have used your free AI answers for today (the demo gives each visitor a small daily allowance so the free AI accounts last). "
    "The firewall itself still works for you: try the Attack playground, which scans a message and shows the verdict without calling the AI. "
    "The allowance resets at midnight UTC."
)

LIMIT_REACHED_MESSAGE = (
    "The demo's free daily quota for AI answers has been used up, so this message was not sent to the AI. "
    "The firewall itself still works: try the Attack playground, which scans a message and shows the verdict without calling the AI. "
    "The quota resets at midnight UTC."
)


class DailyLimit:
    def __init__(self, limit: int, clock: Callable[[], float] = time.time):
        self.limit, self._clock = int(limit), clock
        self._lock = threading.Lock()
        self._day = int(clock() // DAY_S)
        self._used = 0

    def _roll(self) -> None:
        day = int(self._clock() // DAY_S)
        if day != self._day:
            self._day, self._used = day, 0

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def try_acquire(self) -> bool:
        """Take one slot. True if the call may go ahead."""
        if not self.enabled:
            return True
        with self._lock:
            self._roll()
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    @property
    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._used

    @property
    def remaining(self) -> int | None:
        return None if not self.enabled else max(self.limit - self.used, 0)

    def seconds_until_reset(self) -> int:
        return int(DAY_S - (self._clock() % DAY_S)) + 1


class PerClientDailyLimit:
    """At most `limit` LLM calls per client key per UTC day. Memory is bounded: when more than `max_keys` clients are tracked, the table is
    cleared (an attacker spraying addresses can reset it, which is why the GLOBAL cap exists as well)."""

    def __init__(self, limit: int, clock: Callable[[], float] = time.time, max_keys: int = 20_000):
        self.limit, self._clock, self.max_keys = int(limit), clock, max_keys
        self._lock = threading.Lock()
        self._day = int(clock() // DAY_S)
        self._used: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def _roll(self) -> None:
        day = int(self._clock() // DAY_S)
        if day != self._day:
            self._day, self._used = day, {}

    def try_acquire(self, key: str) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            self._roll()
            n = self._used.get(key, 0)
            if n >= self.limit:
                return False
            if key not in self._used and len(self._used) >= self.max_keys:
                self._used.clear()
            self._used[key] = n + 1
            return True

    def used(self, key: str) -> int:
        with self._lock:
            self._roll()
            return self._used.get(key, 0)

    def seconds_until_reset(self) -> int:
        return int(DAY_S - (self._clock() % DAY_S)) + 1
