"""PII backup scanner tests.

The format cases live in tests/fixtures/pii_cases.json and are shared with the browser's TypeScript port
(frontend/src/pii/detectors.test.ts), so both implementations are held to the same expectations. Python-only
properties (exhaustive Verhoeff checks, spans, no raw values, server-vs-browser prefixes) stay here.
"""
import dataclasses
import json
from pathlib import Path

import pytest

from app.normalize import normalize_text
from app.scanners.pii_regex import aadhaar_verified, mask, pan_verified, scan, verhoeff_check_digit, verhoeff_valid

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "pii_cases.json").read_text(encoding="utf-8"))
SERVER_PREFIX = "S_"


def aadhaar(payload="23456789012"):
    """A valid 12-digit Aadhaar-style number: 11 digits + Verhoeff check digit."""
    return payload + verhoeff_check_digit(payload)


# ------------------------------------------------------------------ shared fixture (same cases as the browser)
@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["id"] for c in FIXTURE["cases"]])
def test_shared_pii_case(case):
    r = mask(case["text"])
    assert r.counts == case["counts"]
    assert r.unverified == case.get("unverified", {})          # hidden by shape only ("looks like"), strict check failed
    if "masked" in case:
        assert r.masked_text == case["masked"].replace("{P}", SERVER_PREFIX)


@pytest.mark.parametrize("case", FIXTURE["normalize_cases"], ids=[c["id"] for c in FIXTURE["normalize_cases"]])
def test_shared_normalize_case(case):
    n = normalize_text(case["text"])
    assert (n.text, n.smuggling_chars, n.invisible_removed) == (case["normalized"], case["smuggling"], case["invisible"])


def test_fixture_aadhaar_numbers_are_the_documented_ones():
    assert aadhaar("23456789012") == "234567890124" and aadhaar("34567890123") == "345678901238"


# ------------------------------------------------------------------ Verhoeff
def test_verhoeff_known_vector():
    assert verhoeff_check_digit("236") == "3"       # classic example: 2363 is valid
    assert verhoeff_valid("2363") and not verhoeff_valid("2364")


def test_verhoeff_detects_single_digit_errors_and_adjacent_swaps():
    n = aadhaar()
    assert verhoeff_valid(n)
    for i in range(len(n)):
        for d in "0123456789":
            if d != n[i]:
                assert not verhoeff_valid(n[:i] + d + n[i + 1:]), (i, d)
    for i in range(len(n) - 1):
        if n[i] != n[i + 1]:
            assert not verhoeff_valid(n[:i] + n[i + 1] + n[i] + n[i + 2:])


def test_verhoeff_rejects_non_digits():
    assert not verhoeff_valid("23a3") and not verhoeff_valid("")


# ------------------------------------------------------------------ masking behaviour (Python-only properties)
def test_aadhaar_phone_overlap_never_double_masked():
    n = aadhaar("91987654321")              # 12 digits that also look like +91 plus a mobile number
    r = mask(f"num {n}")
    assert r.total == 1 and "[S_" in r.masked_text


def test_masking_is_idempotent():
    once = mask("hi [PHONE_1] and [EMAIL_2], my pan is ABCPE1234F")
    assert mask(once.masked_text).total == 0


def test_server_placeholders_cannot_collide_with_browser_placeholders():
    r = mask("mail a@b.com")
    assert r.masked_text == "mail [S_EMAIL_1]" and not r.masked_text.startswith("mail [EMAIL_")


def test_no_raw_values_in_findings():
    n = aadhaar()
    r = mask(f"{n} ABCPE1234F a@b.com 9876543210")
    blob = repr(r.findings) + str([dataclasses.asdict(f) for f in r.findings])
    for raw in (n, "ABCPE1234F", "a@b.com", "9876543210"):
        assert raw not in blob


def test_large_input():
    big = ("lorem ipsum 123456 dolor " * 2000) + " mail x@y.com"
    r = mask(big)
    assert r.counts == {"EMAIL": 1} and r.masked_text.endswith("mail [S_EMAIL_1]")


def test_spans_point_at_original_text():
    n = aadhaar()
    text = f"x {n} y"
    f = scan(text)[0]
    assert text[f.start:f.end] == n and f.type == "AADHAAR"


# ------------------------------------------------------------------ shape-based Aadhaar/PAN (Day 4.21): strict checks are labels, not gates
def test_strict_checks_are_only_labels():
    n = aadhaar()
    assert aadhaar_verified(n) and not aadhaar_verified(n[:-1] + str((int(n[-1]) + 1) % 10))   # bad checksum
    assert not aadhaar_verified("0" + n[1:]) and not aadhaar_verified("1" + n[1:])             # first digit 0 / 1
    assert pan_verified("ABCPE1234F") and pan_verified("abcpe1234f") and not pan_verified("ABCDE1234F")


def test_findings_carry_the_verified_flag():
    n = aadhaar()
    good = scan(f"{n} ABCPE1234F")
    assert [f.verified for f in good] == [True, True]
    bad = scan("1234 5768 9881 ABCDE1234F")
    assert [f.type for f in bad] == ["AADHAAR", "PAN"] and [f.verified for f in bad] == [False, False]


def test_plus91_phone_is_still_a_phone_and_not_an_unverified_aadhaar():
    r = mask("call +919876543210 or 91-9876543210")
    assert r.counts == {"PHONE": 2} and r.unverified == {}


def test_unverified_values_are_still_numbered_and_never_leaked_in_findings():
    r = mask("1234 5768 9881 and 123457689881 and 2222 3333 4444")
    assert r.masked_text == "[S_AADHAAR_1] and [S_AADHAAR_1] and [S_AADHAAR_2]"
    assert "1234" not in repr(r.findings)
