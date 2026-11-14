"""Server-side PII backup scanner: regex + checksum, for data the browser layer (Layer 1) may have missed.

Types: Aadhaar (12 digits, Verhoeff checksum, first digit 2-9), PAN (5 letters + 4 digits + 1 letter, valid entity letter),
email, Indian mobile numbers. Names and free-text PII are the browser NER model's job; this is a cheap safety net.

`mask(text)` replaces findings with placeholders such as [S_AADHAAR_1]. The "S_" prefix keeps them distinct from the
browser's placeholders ([PHONE_1] ...), whose numbering restarts per message: a server placeholder that collided with a client
key would be un-masked to the wrong value on the client. The same value gets the same placeholder within one call.
Raw values are never returned or stored; only types, spans and counts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- Verhoeff checksum (used by Aadhaar)
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1], [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1], [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(number: str) -> bool:
    """True if the digit string (check digit included) passes the Verhoeff check."""
    if not number.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(payload: str) -> str:
    """Check digit to append to a digit string (handy for tests and fixtures)."""
    c = 0
    for i, ch in enumerate(reversed(payload)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


# ---------------------------------------------------------------- patterns
# Aadhaar: not part of a longer digit run (also across single space/hyphen separators, so card numbers are not cut up)
_AADHAAR = re.compile(r"(?<!\d)(?<!\d[ -])([2-9]\d{3}[ -]?\d{4}[ -]?\d{4})(?![ -]?\d)")
_PAN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{3}[PCHFATBLJGpchfatbljg][A-Za-z]\d{4}[A-Za-z])(?![A-Za-z0-9])")
_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
_PHONE = re.compile(r"(?<![\d+])(?:(?:\+|00)?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)")

TYPES = ("AADHAAR", "PAN", "EMAIL", "PHONE")


@dataclass(frozen=True)
class PIIFinding:
    type: str
    start: int
    end: int
    placeholder: str = ""


@dataclass
class MaskResult:
    masked_text: str
    findings: list[PIIFinding] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.type] = out.get(f.type, 0) + 1
        return out

    @property
    def total(self) -> int:
        return len(self.findings)


def _candidates(text: str):
    """(start, end, type, normalised value) for every valid hit; earlier types win overlaps."""
    for m in _AADHAAR.finditer(text):
        digits = re.sub(r"\D", "", m.group(1))
        if len(digits) == 12 and verhoeff_valid(digits):
            yield m.start(1), m.end(1), "AADHAAR", digits
    for m in _PAN.finditer(text):
        yield m.start(1), m.end(1), "PAN", m.group(1).upper()
    for m in _EMAIL.finditer(text):
        yield m.start(), m.end(), "EMAIL", m.group().lower()
    for m in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        yield m.start(), m.end(), "PHONE", digits[-10:]


def mask(text: str) -> MaskResult:
    taken: list[tuple[int, int]] = []
    hits = []
    for s, e, t, v in _candidates(text):          # generator order = priority: Aadhaar, PAN, email, phone
        if any(s < b and a < e for a, b in taken):
            continue
        taken.append((s, e))
        hits.append((s, e, t, v))
    hits.sort()
    numbering: dict[tuple[str, str], int] = {}
    per_type: dict[str, int] = {}
    findings, out, pos = [], [], 0
    for s, e, t, v in hits:
        key = (t, v)
        if key not in numbering:
            per_type[t] = per_type.get(t, 0) + 1
            numbering[key] = per_type[t]
        ph = f"[S_{t}_{numbering[key]}]"
        out.append(text[pos:s])
        out.append(ph)
        pos = e
        findings.append(PIIFinding(t, s, e, ph))
    out.append(text[pos:])
    return MaskResult("".join(out), findings)


def scan(text: str) -> list[PIIFinding]:
    return mask(text).findings
