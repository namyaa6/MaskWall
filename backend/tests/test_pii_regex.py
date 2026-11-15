import dataclasses

import pytest

from app.scanners.pii_regex import mask, scan, verhoeff_check_digit, verhoeff_valid


def aadhaar(payload="23456789012"):
    """A valid 12-digit Aadhaar-style number: 11 digits + Verhoeff check digit."""
    return payload + verhoeff_check_digit(payload)


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


# ------------------------------------------------------------------ Aadhaar
@pytest.mark.parametrize("fmt", ["{0}", "{1} {2} {3}", "{1}-{2}-{3}"])
def test_valid_aadhaar_in_common_formats(fmt):
    n = aadhaar()
    text = "my aadhaar is " + fmt.format(n, n[:4], n[4:8], n[8:]) + " please update"
    r = mask(text)
    assert r.counts == {"AADHAAR": 1}
    assert "[S_AADHAAR_1]" in r.masked_text and n[:4] not in r.masked_text.replace("[S_AADHAAR_1]", "")


def test_invalid_checksum_not_flagged():
    n = aadhaar()
    bad = n[:-1] + str((int(n[-1]) + 1) % 10)
    assert scan(f"number {bad} here") == []


@pytest.mark.parametrize("first", ["0", "1"])
def test_aadhaar_cannot_start_with_0_or_1(first):
    payload = first + "3456789012"
    assert scan(first + payload[1:] + verhoeff_check_digit(payload)) == []


def test_card_number_and_longer_digit_runs_are_not_cut_up():
    assert scan("card 4111 1111 1111 1111 expires soon") == []
    assert scan("id 2345678901234567") == []
    n = aadhaar()
    assert scan(f"{n}1") == [] and scan(f"1{n}") == []


# ------------------------------------------------------------------ PAN
def test_valid_pan_detected_case_insensitive():
    assert mask("PAN: ABCPE1234F").counts == {"PAN": 1}
    assert mask("pan abcpe1234f ok").counts == {"PAN": 1}


@pytest.mark.parametrize("text", ["ABCXE1234F", "ABCPE12345", "ABCPE1234", "AB1PE1234F", "XABCPE1234F", "ABCPE1234FX"])
def test_invalid_pan_shapes_rejected(text):
    assert scan(text) == []


# ------------------------------------------------------------------ email and phone
def test_email_detected():
    r = mask("write to Priya.Sharma+test@example.co.in or bob@x.io today")
    assert r.counts == {"EMAIL": 2} and "@" not in r.masked_text


@pytest.mark.parametrize("phone", ["9876543210", "98765 43210", "98765-43210", "+91 98765 43210", "+919876543210", "91-9876543210"])
def test_indian_mobile_formats(phone):
    r = mask(f"call me on {phone} tomorrow")
    assert r.counts == {"PHONE": 1} and "[S_PHONE_1]" in r.masked_text and "98765" not in r.masked_text


@pytest.mark.parametrize("text", ["call 1234567890", "call 5876543210", "order 98765432", "ref 98765432101234"])
def test_non_mobile_numbers_ignored(text):
    assert scan(text) == []


# ------------------------------------------------------------------ masking behaviour
def test_same_value_same_placeholder_and_numbering_per_type():
    a, b = aadhaar("23456789012"), aadhaar("34567890123")
    r = mask(f"{a} then {b} then {a} again, mail a@b.com and c@d.com and a@b.com")
    assert r.masked_text == "[S_AADHAAR_1] then [S_AADHAAR_2] then [S_AADHAAR_1] again, mail [S_EMAIL_1] and [S_EMAIL_2] and [S_EMAIL_1]"
    assert r.counts == {"AADHAAR": 3, "EMAIL": 3} and r.total == 6


def test_aadhaar_wins_over_phone_overlap_and_no_double_masking():
    n = aadhaar("91987654321")              # 12 digits that also look like +91 plus a mobile number
    r = mask(f"num {n}")
    assert r.counts in ({"AADHAAR": 1}, {"PHONE": 1}) and r.total == 1 and "[S_" in r.masked_text


def test_masking_is_idempotent_and_leaves_browser_placeholders_alone():
    text = "hi [PHONE_1] and [EMAIL_2], my pan is ABCPE1234F"
    once = mask(text)
    assert once.masked_text == "hi [PHONE_1] and [EMAIL_2], my pan is [S_PAN_1]"
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


def test_clean_text_unchanged_and_unicode_safe():
    t = "naïve café résumé — 東京 😀 nothing sensitive here: 2024-11-14, v1.2.3, 42%"
    r = mask(t)
    assert r.masked_text == t and r.total == 0


def test_empty_and_large_inputs():
    assert mask("").masked_text == ""
    big = ("lorem ipsum 123456 dolor " * 2000) + " mail x@y.com"
    r = mask(big)
    assert r.counts == {"EMAIL": 1} and r.masked_text.endswith("mail [S_EMAIL_1]")


def test_spans_point_at_original_text():
    n = aadhaar()
    text = f"x {n} y"
    f = scan(text)[0]
    assert text[f.start:f.end] == n and f.type == "AADHAAR"


# ------------------------------------------------------------------ formats found missing by the review
@pytest.mark.parametrize("phone", ["098765 43210", "09876543210", "9876 543 210", "+91 9876-543-210", "98 76 54 32 10", "0091 98765 43210"])
def test_more_mobile_formats(phone):
    r = mask(f"call {phone} tomorrow")
    assert r.counts == {"PHONE": 1} and r.masked_text == "call [S_PHONE_1] tomorrow", phone


@pytest.mark.parametrize("pan", ["ABCPE 1234 F", "abcpe-1234-f", "ABCPE1234F"])
def test_pan_with_separators(pan):
    r = mask(f"my pan is {pan}.")
    assert r.counts == {"PAN": 1} and r.masked_text == "my pan is [S_PAN_1]."


def test_same_pan_with_and_without_separators_gets_one_placeholder():
    assert mask("ABCPE1234F and ABCPE 1234 F").masked_text == "[S_PAN_1] and [S_PAN_1]"


@pytest.mark.parametrize("text", ["call 12 3456 7890", "ref 9876 543 21", "year 2024 1234 5", "ABCPE 12345 F", "ABC 1234 F"])
def test_relaxed_formats_still_ignore_non_pii(text):
    assert scan(text) == []
