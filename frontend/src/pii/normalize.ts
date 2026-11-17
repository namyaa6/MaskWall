/**
 * Port of backend/app/normalize.py. Applied in the browser BEFORE masking and the attack pre-check, so the browser
 * masks and scores the same text the server will see (the server normalises again; on already-normal text it is a no-op).
 *
 *  1. NFKC (fullwidth/compatibility forms become ordinary characters)
 *  2. non-ASCII decimal digits become ASCII digits
 *  3. remove the Unicode tag block (U+E0000-E007F) and supplementary variation selectors (U+E0100-E01EF): counted as smuggling
 *  4. remove other invisible format characters (Cf) and control characters (Cc) except newline, tab and carriage return
 *  5. keep ZWJ/ZWNJ except between two ASCII letters
 * Shared cases: backend/tests/fixtures/pii_cases.json ("normalize_cases").
 */
export interface Normalized {
  text: string;
  changed: boolean;
  smuggling: number;
  invisible: number;
}

const CC = /^\p{Cc}$/u;
const CF = /^\p{Cf}$/u;
const ND = /^\p{Nd}$/u;
const KEEP_CONTROLS = new Set(["\n", "\t", "\r"]);
const JOINERS = new Set(["‌", "‍"]);
const LATIN_JOINER = /(?<=[A-Za-z])[‌‍](?=[A-Za-z])/g;

/** Value of a Unicode decimal digit. Unicode encodes every Nd digit set as a contiguous run 0..9, so value = offset mod 10 within its run. */
function digitValue(cp: number): number {
  let start = cp;
  while (start > 0 && ND.test(String.fromCodePoint(start - 1))) start--;
  return (cp - start) % 10;
}

export function normalizeText(raw: string): Normalized {
  const t = raw.normalize("NFKC");
  let out = "";
  let smuggling = 0;
  let invisible = 0;
  for (const ch of t) {
    const cp = ch.codePointAt(0)!;
    if ((cp >= 0xe0000 && cp <= 0xe007f) || (cp >= 0xe0100 && cp <= 0xe01ef)) {
      smuggling++;
      continue;
    }
    if (CC.test(ch) && !KEEP_CONTROLS.has(ch)) {
      invisible++;
      continue;
    }
    if (CF.test(ch) && !JOINERS.has(ch)) {
      invisible++;
      continue;
    }
    if (cp > 0x7f && ND.test(ch)) {
      out += String(digitValue(cp));
      continue;
    }
    out += ch;
  }
  const noJoin = out.replace(LATIN_JOINER, "");
  invisible += out.length - noJoin.length;
  return { text: noJoin, changed: noJoin !== raw, smuggling, invisible };
}
