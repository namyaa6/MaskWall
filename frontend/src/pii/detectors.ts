/**
 * Port of backend/app/scanners/pii_regex.py: Aadhaar (Verhoeff checksum), PAN, email, Indian mobile numbers.
 * Same patterns, same priority (Aadhaar > PAN > email > phone) and the same overlap rule; held to the same
 * shared cases (backend/tests/fixtures/pii_cases.json). Input is expected to be normalised first (normalize.ts),
 * so digits are ASCII; Python's Unicode-aware \w is matched with \p{L}\p{N}_ here.
 */
export type RegexPiiType = "AADHAAR" | "PAN" | "EMAIL" | "PHONE";
export type PiiType = RegexPiiType | "NAME";
export const PII_TYPES: readonly PiiType[] = ["NAME", "EMAIL", "PHONE", "AADHAAR", "PAN"];

export interface PiiSpan {
  start: number;
  end: number;
  type: PiiType;
  /** normalised value: same value => same placeholder (e.g. "ABCPE 1234 F" and "abcpe1234f") */
  key: string;
}

// ---------------------------------------------------------------- Verhoeff
const D = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
  [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1], [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
  [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];
const P = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
  [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1], [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];
const INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9];

export function verhoeffValid(num: string): boolean {
  if (!/^[0-9]+$/.test(num)) return false;
  let c = 0;
  const digits = [...num].reverse();
  for (let i = 0; i < digits.length; i++) c = D[c][P[i % 8][Number(digits[i])]];
  return c === 0;
}

export function verhoeffCheckDigit(payload: string): string {
  let c = 0;
  const digits = [...payload].reverse();
  for (let i = 0; i < digits.length; i++) c = D[c][P[(i + 1) % 8][Number(digits[i])]];
  return String(INV[c]);
}

// ---------------------------------------------------------------- patterns (kept textually close to the Python ones)
const AADHAAR = /(?<!\d)(?<!\d[ -])([2-9]\d{3}[ -]?\d{4}[ -]?\d{4})(?![ -]?\d)/g;
const PAN = /(?<![A-Za-z0-9])([A-Za-z]{3}[PCHFATBLJGpchfatbljg][A-Za-z][ -]?\d{4}[ -]?[A-Za-z])(?![A-Za-z0-9])/g;
const EMAIL = /(?<![\p{L}\p{N}_.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\p{L}\p{N}_-])/gu;
const PHONE = /(?<![\d+])(?:(?:(?:\+|00)?91[ -]?)|0)?[6-9](?:[ -]?\d){9}(?![ -]?\d)/g;

/** Every valid hit, in priority order (overlaps not yet resolved). */
function* candidates(text: string): Generator<PiiSpan> {
  for (const m of text.matchAll(AADHAAR)) {
    const digits = m[1].replace(/\D/g, "");
    if (digits.length === 12 && verhoeffValid(digits)) yield { start: m.index!, end: m.index! + m[0].length, type: "AADHAAR", key: digits };
  }
  for (const m of text.matchAll(PAN)) yield { start: m.index!, end: m.index! + m[0].length, type: "PAN", key: m[1].replace(/[ -]/g, "").toUpperCase() };
  for (const m of text.matchAll(EMAIL)) yield { start: m.index!, end: m.index! + m[0].length, type: "EMAIL", key: m[0].toLowerCase() };
  for (const m of text.matchAll(PHONE)) yield { start: m.index!, end: m.index! + m[0].length, type: "PHONE", key: m[0].replace(/\D/g, "").slice(-10) };
}

const overlaps = (a: { start: number; end: number }, b: { start: number; end: number }) => a.start < b.end && b.start < a.end;

/**
 * Regex findings plus optional extra spans (names from the NER model). Earlier sources win overlaps:
 * Aadhaar, PAN, email, phone, then the extra spans in the order given. Returned sorted by position.
 */
export function detect(text: string, extra: PiiSpan[] = []): PiiSpan[] {
  const taken: PiiSpan[] = [];
  for (const c of [...candidates(text), ...extra]) {
    if (taken.some((t) => overlaps(t, c))) continue;
    taken.push(c);
  }
  return taken.sort((a, b) => a.start - b.start);
}
