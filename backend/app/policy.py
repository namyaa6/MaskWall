"""Policy engine: turn scanner results into allow / warn / block. Pure functions, no I/O, easy to test.

Input side (injection score from the model, coverage from the scan):
  score >= block threshold            -> BLOCK  (injection_detected)
  warn <= score < block               -> WARN   (injection_suspected)
  coverage == "partial"               -> at least WARN (partial_scan): part of the input was never scanned
  STRICT mode                         -> WARN is treated as BLOCK
Thresholds come from the model's own validation tuning (backend/model/<name>/model_meta.json):
  block = max(thr_f1, thr_fpr1pct)   the more conservative of the F1-optimal and the FPR <= 1% thresholds
  warn  = min(thr_f1, thr_fpr1pct)   between the two the model is suspicious but not validated as precise
Output side: any leakage finding (canary, system prompt, secrets) -> BLOCK the reply.
Hidden Unicode smuggling characters (tag block, supplementary variation selectors) make the policy at least WARN.
Server-side PII masking never changes the action by itself; it is reported as a note (the browser layer should have caught it).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


BLOCKED_INPUT_MESSAGE = "This message was blocked by the MaskWall firewall because it looks like a prompt-injection or jailbreak attempt."
BLOCKED_OUTPUT_MESSAGE = "The model's reply was withheld by the MaskWall firewall because it appeared to leak protected information."


@dataclass(frozen=True)
class Thresholds:
    warn: float
    block: float

    def __post_init__(self):
        if not 0.0 <= self.warn <= self.block <= 1.0:
            raise ValueError(f"need 0 <= warn <= block <= 1, got warn={self.warn}, block={self.block}")


def thresholds_from_meta(thr_f1: float, thr_fpr1pct: float, *, block: float | None = None, warn: float | None = None) -> Thresholds:
    """Model-derived thresholds, optionally overridden (MASKWALL_BLOCK_THRESHOLD / MASKWALL_WARN_THRESHOLD)."""
    b = max(thr_f1, thr_fpr1pct) if block is None else block
    w = min(thr_f1, thr_fpr1pct) if warn is None else warn
    return Thresholds(warn=min(w, b), block=b)


@dataclass(frozen=True)
class InputDecision:
    action: Action
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def allowed_to_reach_llm(self) -> bool:
        return self.action is not Action.BLOCK


@dataclass(frozen=True)
class OutputDecision:
    action: Action
    reasons: tuple[str, ...] = ()


def decide_input(score: float, coverage: str, thresholds: Thresholds, *, pii_masked: int = 0, strict: bool = False,
                 hidden_unicode: int = 0) -> InputDecision:
    reasons: list[str] = []
    if score >= thresholds.block:
        action = Action.BLOCK
        reasons.append("injection_detected")
    elif score >= thresholds.warn:
        action = Action.WARN
        reasons.append("injection_suspected")
    else:
        action = Action.ALLOW
    if coverage != "full":
        reasons.append("partial_scan")
        if action is Action.ALLOW:
            action = Action.WARN
    if hidden_unicode > 0:                       # tag characters / variation selectors have no legitimate use in a prompt
        reasons.append("hidden_characters")
        if action is Action.ALLOW:
            action = Action.WARN
    if strict and action is Action.WARN:
        action = Action.BLOCK
        reasons.append("strict_mode")
    notes = ("pii_masked_server_side",) if pii_masked > 0 else ()
    return InputDecision(action, tuple(reasons), notes)


def decide_output(flagged: bool, kinds: list[str] | tuple[str, ...] = ()) -> OutputDecision:
    if flagged:
        return OutputDecision(Action.BLOCK, tuple(f"output_leak:{k}" for k in kinds) or ("output_leak",))
    return OutputDecision(Action.ALLOW)
