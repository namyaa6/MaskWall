import random

import pytest

from app.scanners.windowing import CONTENT, HEAD, STRIDE, TAIL, covered_tokens, head_tail_ids, select_windows, window_spans


def test_head_tail_short_input_unchanged():
    ids = list(range(100))
    assert head_tail_ids(ids) == ids
    assert head_tail_ids(list(range(CONTENT))) == list(range(CONTENT))


def test_head_tail_long_input_keeps_head_and_tail():
    ids = list(range(2000))
    out = head_tail_ids(ids)
    assert len(out) == HEAD + TAIL == CONTENT
    assert out[:HEAD] == ids[:HEAD] and out[-TAIL:] == ids[-TAIL:]
    assert 1000 not in out                                   # the middle is invisible: the reason sliding windows exist


@pytest.mark.parametrize("n,expected", [
    (0, [(0, 0)]), (1, [(0, 1)]), (510, [(0, 510)]), (511, [(0, 510), (1, 511)]),
    (1000, [(0, 510), (255, 765), (490, 1000)]),
])
def test_window_spans_examples(n, expected):
    assert window_spans(n) == expected


def test_window_spans_invariants():
    for n in list(range(1, 1200)) + [1785, 2500, 4000, 10000]:
        spans = window_spans(n)
        assert spans[0][0] == 0 and spans[-1][1] == n                       # whole text covered start to end
        assert all(0 <= e - s <= CONTENT for s, e in spans)
        assert all(b[0] - a[0] <= STRIDE for a, b in zip(spans, spans[1:]))  # consecutive starts at most one stride apart
        assert covered_tokens(spans) == n


def test_any_span_up_to_one_stride_fits_in_one_window():
    """An injected sentence of <= 255 tokens must lie fully inside at least one window, wherever it sits."""
    rng = random.Random(0)
    for n in [511, 700, 1000, 1785, 3000]:
        spans = window_spans(n)
        for _ in range(400):
            length = rng.randint(1, STRIDE)
            a = rng.randint(0, n - length)
            assert any(s <= a and a + length <= e for s, e in spans), (n, a, length)


def test_select_windows_under_cap_is_untouched():
    spans = window_spans(1000)
    assert select_windows(spans, 6) == (spans, False)
    assert select_windows(spans, 3) == (spans, False)


def test_select_windows_over_cap_keeps_first_and_last():
    spans = window_spans(4000)
    assert len(spans) > 6
    chosen, partial = select_windows(spans, 6)
    assert partial and len(chosen) == 6
    assert chosen[:5] == spans[:5] and chosen[-1] == spans[-1]
    assert select_windows(spans, 1) == ([spans[-1]], True)
    with pytest.raises(ValueError):
        select_windows(spans, 0)


def test_covered_tokens_counts_overlap_once():
    assert covered_tokens([(0, 10), (5, 15), (20, 25)]) == 20
    assert covered_tokens([]) == 0
