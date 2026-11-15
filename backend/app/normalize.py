"""Input normalisation, applied BEFORE PII masking, injection scanning and the LLM call, so all three see the same text.

Why: the classifier tokenises raw text. A review showed that the same attack sentence written in fullwidth letters scored 0.006 (blocked
when written normally: 0.999), and that text encoded in Unicode "tag" characters (U+E0000 block) is invisible to the tokenizer (0 tokens:
allowed) while some LLMs still read it. Fix: normalise first, and send the normalised text to the LLM.

Steps:
  1. NFKC (fullwidth/compatibility forms become ordinary characters)
  2. non-ASCII decimal digits become ASCII digits (so Devanagari or fullwidth numbers are visible to the PII scanner and the model)
  3. remove the Unicode tag block (U+E0000-E007F) and supplementary variation selectors (U+E0100-E01EF): no legitimate use in prompts,
     the standard channels for hidden-text smuggling; they are counted separately and make the policy warn
  4. remove other invisible format characters (zero-width spaces, word joiners, BOM, bidi controls, soft hyphen) and control characters
     except newline, tab and carriage return
  5. keep ZWJ/ZWNJ (needed by Devanagari and emoji sequences) except between two ASCII letters, where they only split a word

Not handled (model weaknesses, not normalisation): base64 or other encodings, letter-spacing, homoglyph scripts the tokenizer maps unevenly.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_LATIN_JOINER = re.compile(r"(?<=[A-Za-z])[‌‍](?=[A-Za-z])")
_KEEP_CONTROLS = {"\n", "\t", "\r"}


@dataclass(frozen=True)
class Normalized:
    text: str
    changed: bool
    smuggling_chars: int      # tag-block and supplementary variation-selector characters removed
    invisible_removed: int    # other invisible format/control characters removed

    @property
    def hidden_total(self) -> int:
        return self.smuggling_chars + self.invisible_removed


def normalize_text(raw: str) -> Normalized:
    t = unicodedata.normalize("NFKC", raw)
    out: list[str] = []
    smuggling = invisible = 0
    for ch in t:
        cp = ord(ch)
        if 0xE0000 <= cp <= 0xE007F or 0xE0100 <= cp <= 0xE01EF:
            smuggling += 1
            continue
        cat = unicodedata.category(ch)
        if cat == "Cc" and ch not in _KEEP_CONTROLS:
            invisible += 1
            continue
        if cat == "Cf" and ch not in ("‌", "‍"):
            invisible += 1
            continue
        if cat == "Nd" and not ch.isascii():
            ch = str(unicodedata.decimal(ch))
        out.append(ch)
    t = "".join(out)
    no_join = _LATIN_JOINER.sub("", t)
    invisible += len(t) - len(no_join)
    return Normalized(no_join, no_join != raw, smuggling, invisible)
