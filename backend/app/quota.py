"""Global daily cap on LLM calls, so public traffic cannot use up the owner's free Groq/Gemini quota.

One counter for the whole app (not per client: per-client limits already exist in abuse.py). The day is the UTC calendar day;
the counter lives in memory, so a restart resets it (on the free Space a restart only follows 48 h of idleness or a redeploy, and an
attacker cannot trigger one). Only messages that actually reach the LLM count: blocked attacks, rejected inputs and /scan never do,
so the Attack playground keeps working when the cap is reached. A slot is taken BEFORE the provider call and is not refunded if the
call fails, because a failed call may still have used provider quota. `limit <= 0` disables the cap.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

DAY_S = 86400

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
