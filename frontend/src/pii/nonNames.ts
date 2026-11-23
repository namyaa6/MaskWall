/**
 * Terms that can never be a person's name: the names of Indian ID documents and payment/KYC terms (Day 4.24).
 *
 * Found on the live site: in "Aadhaar 1234 5768 9881 and PAN ABCDE1234F" the name model tagged the word "Aadhaar" as a person
 * ([NAME_1]), so the AI received "[NAME_1] [AADHAAR_1] ..." and the panel counted 3 hidden details instead of 2. The model reads a
 * capitalised word in front of a number as a name; it has no idea what an Aadhaar is.
 *
 * `dropNonNames` is applied to the NAME spans the model (or Strict mode) returns, never to emails, phones, Aadhaar or PAN, which
 * come from rules. The list was written from the owner's report and the document names a Chat user is likely to type, BEFORE it was
 * measured on the name evaluation sets, and was not tuned on them (docs/decisions.md Day 4.24). Matching is case-insensitive and by
 * whole word; a multi-word term ("voter id") must match all its words.
 *
 * Known trade-off: a real person whose whole name is one of these words ("Pan") is no longer hidden. A name that merely
 * CONTAINS one ("Rahul Pan Sharma") is left alone: only a term at the start or the end of a span is removed.
 */
import type { PiiSpan } from "./detectors";

export const NON_NAME_TERMS: readonly string[] = [
  "aadhaar", "aadhar", "adhaar", "aadhaar card", "aadhar card",
  "pan", "pan card",
  "upi", "gst", "gstin", "ifsc", "otp", "kyc", "cvv", "uidai",
  "passport", "voter id", "voter card", "driving licence", "driving license", "ration card",
];

// longest term first, so "pan card" is removed as a whole before "pan" could leave a stray "card"
const TERMS = NON_NAME_TERMS.map((t) => t.split(" ")).sort((a, b) => b.length - a.length);

interface Word {
  lw: string;
  start: number;
  end: number;
}

const wordsOf = (text: string, from: number, to: number): Word[] =>
  [...text.slice(from, to).matchAll(/[\p{L}\p{N}]+/gu)].map((m) => ({ lw: m[0].toLowerCase(), start: from + m.index!, end: from + m.index! + m[0].length }));

const termAt = (words: Word[], i: number): number => {
  for (const t of TERMS) if (i + t.length <= words.length && t.every((w, k) => words[i + k].lw === w)) return t.length;
  return 0;
};

const termEndingAt = (words: Word[], j: number): number => {
  for (const t of TERMS) if (j - t.length >= 0 && t.every((w, k) => words[j - t.length + k].lw === w)) return t.length;
  return 0;
};

/** Remove document/payment terms from the start and end of each NAME span; drop a span that is nothing but such terms. */
export function dropNonNames(text: string, spans: PiiSpan[]): PiiSpan[] {
  const out: PiiSpan[] = [];
  for (const s of spans) {
    if (s.type !== "NAME") {
      out.push(s);
      continue;
    }
    const words = wordsOf(text, s.start, s.end);
    let i = 0;
    let j = words.length;
    for (let n = termAt(words.slice(0, j), i); n; n = termAt(words.slice(0, j), i)) i += n;
    for (let n = termEndingAt(words.slice(i, j), j - i); n; n = termEndingAt(words.slice(i, j), j - i)) j -= n;
    if (words.length === 0 || (i === 0 && j === words.length)) out.push(s);   // nothing to remove: the span is returned exactly as the model gave it
    else if (i < j) {
      const start = words[i].start;
      const end = words[j - 1].end;
      out.push({ ...s, start, end, key: text.slice(start, end).toLowerCase().replace(/\s+/g, " ") });
    }
  }
  return out;
}
