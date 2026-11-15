import pytest

from app.policy import Action, Thresholds, decide_input, decide_output, thresholds_from_meta

T = Thresholds(warn=0.5, block=0.9)


# ------------------------------------------------------------------ thresholds from the model's validation tuning
def test_thresholds_block_is_max_and_warn_is_min_whatever_the_order():
    a = thresholds_from_meta(thr_f1=0.924, thr_fpr1pct=0.779)      # MiniLM int8: F1-optimal threshold above the FPR<=1% one
    b = thresholds_from_meta(thr_f1=0.277, thr_fpr1pct=0.60)       # reversed order
    assert (a.warn, a.block) == (0.779, 0.924) and (b.warn, b.block) == (0.277, 0.60)


def test_threshold_overrides_and_validation():
    t = thresholds_from_meta(0.9, 0.8, block=0.95, warn=0.7)
    assert (t.warn, t.block) == (0.7, 0.95)
    assert thresholds_from_meta(0.9, 0.8, block=0.5).warn == 0.5      # warn can never exceed block
    for bad in [dict(warn=0.9, block=0.5), dict(warn=-0.1, block=0.5), dict(warn=0.1, block=1.1)]:
        with pytest.raises(ValueError):
            Thresholds(**bad)


# ------------------------------------------------------------------ the input truth table
@pytest.mark.parametrize("score,coverage,strict,action,reasons", [
    (0.00, "full", False, Action.ALLOW, ()),
    (0.49, "full", False, Action.ALLOW, ()),
    (0.50, "full", False, Action.WARN, ("injection_suspected",)),          # exactly at the warn threshold
    (0.89, "full", False, Action.WARN, ("injection_suspected",)),
    (0.90, "full", False, Action.BLOCK, ("injection_detected",)),          # exactly at the block threshold
    (1.00, "full", False, Action.BLOCK, ("injection_detected",)),
    (0.10, "partial", False, Action.WARN, ("partial_scan",)),              # unscanned text never silently allowed
    (0.70, "partial", False, Action.WARN, ("injection_suspected", "partial_scan")),
    (0.95, "partial", False, Action.BLOCK, ("injection_detected", "partial_scan")),
    (0.10, "full", True, Action.ALLOW, ()),                                # strict mode only upgrades warnings
    (0.70, "full", True, Action.BLOCK, ("injection_suspected", "strict_mode")),
    (0.10, "partial", True, Action.BLOCK, ("partial_scan", "strict_mode")),
    (0.95, "full", True, Action.BLOCK, ("injection_detected",)),
])
def test_input_truth_table(score, coverage, strict, action, reasons):
    d = decide_input(score, coverage, T, strict=strict)
    assert d.action is action and d.reasons == reasons


def test_blocked_input_never_reaches_the_llm():
    assert decide_input(0.95, "full", T).allowed_to_reach_llm is False
    assert decide_input(0.60, "full", T).allowed_to_reach_llm is True
    assert decide_input(0.01, "full", T).allowed_to_reach_llm is True


def test_pii_masking_is_a_note_not_a_verdict():
    d = decide_input(0.01, "full", T, pii_masked=2)
    assert d.action is Action.ALLOW and d.notes == ("pii_masked_server_side",)
    assert decide_input(0.01, "full", T, pii_masked=0).notes == ()


def test_zero_width_band_when_thresholds_coincide():
    t = Thresholds(warn=0.8, block=0.8)
    assert decide_input(0.79, "full", t).action is Action.ALLOW
    assert decide_input(0.80, "full", t).action is Action.BLOCK            # no warn band at all


# ------------------------------------------------------------------ output
def test_output_decisions():
    assert decide_output(False).action is Action.ALLOW
    d = decide_output(True, ["canary", "secret"])
    assert d.action is Action.BLOCK and d.reasons == ("output_leak:canary", "output_leak:secret")
    assert decide_output(True).reasons == ("output_leak",)


# ------------------------------------------------------------------ hidden Unicode smuggling characters
def test_hidden_characters_make_an_otherwise_clean_input_a_warning():
    d = decide_input(0.01, "full", T, hidden_unicode=12)
    assert d.action is Action.WARN and d.reasons == ("hidden_characters",)


def test_hidden_characters_never_downgrade_a_block_and_combine_with_other_reasons():
    assert decide_input(0.95, "full", T, hidden_unicode=3).action is Action.BLOCK
    assert decide_input(0.95, "full", T, hidden_unicode=3).reasons == ("injection_detected", "hidden_characters")
    d = decide_input(0.01, "partial", T, hidden_unicode=1)
    assert d.action is Action.WARN and d.reasons == ("partial_scan", "hidden_characters")


def test_strict_mode_blocks_hidden_characters():
    d = decide_input(0.01, "full", T, hidden_unicode=1, strict=True)
    assert d.action is Action.BLOCK and d.reasons == ("hidden_characters", "strict_mode")
    assert decide_input(0.01, "full", T, hidden_unicode=0).action is Action.ALLOW
