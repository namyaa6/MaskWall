"""Token windowing for inputs longer than the model's 512-token position limit.

The models were fine-tuned on `[CLS] + head 256 + tail 254 + [SEP]` (ml/headtail.py), so a single window is always at most
510 content tokens. Two ways to handle longer inputs:

  head_tail  one window = first 256 + last 254 tokens. Cheap, but the middle of a long document is never seen.
  sliding    overlapping windows of 510 content tokens with a stride of 255 (50% overlap), score = max over windows.
             Any span of up to 255 tokens (an injected sentence) lies fully inside at least one window.
             A cap on the number of windows bounds latency; if more are needed, the first (cap-1) windows and the LAST
             window are scanned (trailing injections are common) and coverage is reported as "partial".
"""
from __future__ import annotations

CONTENT = 510   # tokens per window, excluding [CLS] and [SEP]
HEAD = 256
TAIL = 254
STRIDE = 255


def head_tail_ids(ids: list[int], head: int = HEAD, tail: int = TAIL) -> list[int]:
    return ids[:head] + ids[-tail:] if len(ids) > head + tail else list(ids)


def window_spans(n: int, content: int = CONTENT, stride: int = STRIDE) -> list[tuple[int, int]]:
    """All overlapping windows [start, end) over n tokens. The last window is aligned to the end of the text."""
    if n <= content:
        return [(0, n)]
    spans, s = [], 0
    while s + content < n:
        spans.append((s, s + content))
        s += stride
    spans.append((n - content, n))
    return spans


def select_windows(spans: list[tuple[int, int]], max_windows: int) -> tuple[list[tuple[int, int]], bool]:
    """Apply the cap. Returns (windows to scan, partial). Over the cap: first (max-1) windows + the last one."""
    if max_windows < 1:
        raise ValueError("max_windows must be >= 1")
    if len(spans) <= max_windows:
        return spans, False
    if max_windows == 1:
        return [spans[-1]], True
    return spans[: max_windows - 1] + [spans[-1]], True


def covered_tokens(spans: list[tuple[int, int]]) -> int:
    """Number of distinct token positions covered by the spans."""
    covered, end = 0, 0
    for s, e in sorted(spans):
        s = max(s, end)
        if e > s:
            covered += e - s
            end = e
    return covered
