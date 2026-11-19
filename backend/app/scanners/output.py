"""Output scanner (Layer 3): stop the LLM's reply from leaking the system prompt or secrets before it reaches the user.

Checks, in order:
  canary         a random token planted in the system prompt; seeing it (also reversed, hex, rot13, as decimal code points, base64 or
                 base32 at any alignment inside a longer encoded text, fullwidth letters, or split by spaces / punctuation) in the
                 output proves the prompt leaked
  system_prompt  verbatim (or near-verbatim) reproduction of the system prompt: shingles of K normalised words from the prompt
                 found in the output (>= MIN_SHINGLES of them, or >= MIN_FRACTION of all prompt shingles)
  known_secret   the application's own configured API keys appearing in the output (also with separators inserted)
  secret         well-known credential formats (Groq, Google, OpenAI, Anthropic, AWS, GitHub, Slack, Stripe, JWT, private keys)
                 and "key = value" assignments with a high-entropy value

Findings carry only a kind and a name, never the matched text. Limits: paraphrased prompts, translated prompts and
credentials in unknown formats are not detected; neither are encodings that are not listed above (for example a cipher of the
model's own invention). The reply is NFKC-normalised first, so fullwidth and other compatibility forms are folded.
"""
from __future__ import annotations

import base64
import codecs
import math
import re
import secrets
import unicodedata
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")

_SECRET_PATTERNS: dict[str, re.Pattern] = {
    "groq_api_key": re.compile(r"(?<![A-Za-z0-9])gsk_[A-Za-z0-9]{20,}"),
    "google_api_key": re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_\-]{35}"),
    "anthropic_api_key": re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai_api_key": re.compile(r"(?<![A-Za-z0-9])sk-(?!ant-)(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    "aws_access_key_id": re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])"),
    "github_token": re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{36,}"),
    "slack_token": re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"),
    "stripe_key": re.compile(r"(?<![A-Za-z0-9])[sr]k_(?:live|test)_[0-9a-zA-Z]{16,}"),
    "jwt": re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    "private_key_block": re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
}
_ASSIGNMENT = re.compile(r"(?i)\b(?:api[_-]?key|secret[_-]?key|secret|access[_-]?token|auth[_-]?token|token|passwd|password)\b\s*[:=]\s*[\"']?([A-Za-z0-9_\-/+=.]{20,})")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {c: s.count(c) for c in set(s)}
    return -sum(n / len(s) * math.log2(n / len(s)) for n in counts.values())


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _squash(text: str) -> str:
    """Lowercase alphanumerics only: defeats spaces, hyphens and punctuation inserted inside a secret."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _aligned_encodings(raw: bytes) -> list[str]:
    """base64 and base32 of `raw` as they appear INSIDE a longer encoded text. Encoded alone, the canary is only found when it starts at a
    group boundary (3 bytes for base64, 5 for base32); inside an encoded system prompt it can start at any offset, which changes the
    characters. For every offset, encode with padding bytes in front and drop the first (partly padding-dependent) characters and the last
    few (which depend on whatever follows). Compared lowercased, as everything else here."""
    out: list[str] = []
    for enc, group, bits_per_char in ((base64.b64encode, 3, 6), (base64.b32encode, 5, 5)):
        for shift in range(group):
            text = enc(b"\x00" * shift + raw).decode().rstrip("=")
            drop = -(-shift * 8 // bits_per_char)          # ceil: characters holding the padding bytes (the last of them partly)
            text = text[drop: len(text) - 3]
            if len(text) >= 10:
                out.append(text.lower())
    return out


def make_canary() -> str:
    return "MWC-" + secrets.token_hex(6)


@dataclass(frozen=True)
class OutputFinding:
    kind: str       # canary | system_prompt | known_secret | secret
    name: str       # e.g. "groq_api_key", "verbatim_prompt", "configured_key"
    count: int = 1


@dataclass
class OutputScan:
    findings: list[OutputFinding] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.findings)

    @property
    def kinds(self) -> list[str]:
        return sorted({f.kind for f in self.findings})

    def to_dict(self) -> dict:
        return {"flagged": self.flagged, "kinds": self.kinds, "findings": [{"kind": f.kind, "name": f.name, "count": f.count} for f in self.findings]}


class OutputScanner:
    def __init__(self, system_prompt: str, canary: str | None = None, known_secrets: tuple[str, ...] | list[str] = (),
                 shingle_k: int = 5, min_shingles: int = 4, min_fraction: float = 0.25):
        self.system_prompt, self.canary = system_prompt, canary
        self.k, self.min_shingles, self.min_fraction = shingle_k, min_shingles, min_fraction
        w = _words(system_prompt)
        self._prompt_shingles = {tuple(w[i:i + self.k]) for i in range(len(w) - self.k + 1)} if len(w) >= self.k else set()
        self._known = [s for s in known_secrets if s and len(s) >= 8]
        self._known_squashed = [_squash(s) for s in self._known]
        self._canary_forms: list[str] = []
        if canary:
            raw = canary.encode()
            self._canary_forms = [canary.lower(), canary[::-1].lower(), raw.hex(), codecs.encode(canary, "rot13").lower(),
                                  "".join(str(ord(c)) for c in canary), *_aligned_encodings(raw)]

    # ---------------------------------------------------------------- individual checks
    def _canary(self, text: str):
        if not self.canary:
            return
        low, sq = text.lower(), _squash(text)
        forms_sq = [_squash(f) for f in self._canary_forms]
        if any(f in low for f in self._canary_forms) or any(f and f in sq for f in forms_sq):
            yield OutputFinding("canary", "system_prompt_canary")

    def _prompt_leak(self, text: str):
        if not self._prompt_shingles:
            return
        w = _words(text)
        out_sh = {tuple(w[i:i + self.k]) for i in range(len(w) - self.k + 1)}
        hit = len(out_sh & self._prompt_shingles)
        if hit >= self.min_shingles or hit / len(self._prompt_shingles) >= self.min_fraction:
            yield OutputFinding("system_prompt", "verbatim_prompt", hit)

    def _known_secrets(self, text: str):
        low, sq = text.lower(), _squash(text)
        for s, ss in zip(self._known, self._known_squashed):
            if s.lower() in low or (len(ss) >= 8 and ss in sq):
                yield OutputFinding("known_secret", "configured_key")

    def _patterns(self, text: str):
        for name, pat in _SECRET_PATTERNS.items():
            n = len(pat.findall(text))
            if n:
                yield OutputFinding("secret", name, n)
        n = 0
        for m in _ASSIGNMENT.finditer(text):
            v = m.group(1)
            if re.search(r"\d", v) and re.search(r"[A-Za-z]", v) and _entropy(v) >= 3.5:
                n += 1
        if n:
            yield OutputFinding("secret", "key_value_assignment", n)

    # ---------------------------------------------------------------- public
    def scan(self, text: str) -> OutputScan:
        if not text:
            return OutputScan()
        text = unicodedata.normalize("NFKC", text)
        findings = [*self._canary(text), *self._prompt_leak(text), *self._known_secrets(text), *self._patterns(text)]
        return OutputScan(findings)
