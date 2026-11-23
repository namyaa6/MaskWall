/**
 * Placeholder vault: masks personal data before a message leaves the device and puts it back into the reply.
 *
 * - One vault per conversation, held in memory only (a React ref). It is never written to localStorage,
 *   sessionStorage, IndexedDB, a log or the network; reloading the page forgets it.
 * - Browser placeholders are [NAME_1], [EMAIL_1], [PHONE_1], [AADHAAR_1], [PAN_1]. The server's backup masking uses
 *   [S_EMAIL_1] etc., which this vault never unmasks (it has no value for them): they stay visible as "hidden by server".
 * - Numbering is per conversation, so a placeholder is never reused for a different value.
 * - Numbering skips any [TYPE_n] the user typed literally, so a typed "[EMAIL_1]" can never be confused with a real
 *   placeholder when the reply is unmasked.
 */
import { detect, PII_TYPES, type PiiSpan, type PiiType } from "./detectors";

export interface MaskOutcome {
  masked: string;
  /** one entry per masked occurrence (no raw values) */
  findings: { type: PiiType; placeholder: string }[];
  counts: Partial<Record<PiiType, number>>;
  /** of `counts`: Aadhaar/PAN-shaped values that failed the strict check (hidden anyway, shown as "looks like") */
  unverified: Partial<Record<PiiType, number>>;
}

const PLACEHOLDER = new RegExp(`\\[(${PII_TYPES.join("|")})_(\\d+)\\]`, "g");

export class PlaceholderVault {
  private byKey = new Map<string, string>();         // "TYPE\u0000key" -> placeholder
  private original = new Map<string, string>();      // placeholder -> text as the user wrote it (first occurrence)
  private counters = new Map<PiiType, number>();
  private reserved = new Set<string>();              // literal placeholders seen in the user's own text

  /** Mask `text` (already normalised). `extra` = name spans from the NER model. */
  mask(text: string, extra: PiiSpan[] = []): MaskOutcome {
    for (const m of text.matchAll(PLACEHOLDER)) if (!this.original.has(m[0])) this.reserved.add(m[0]);
    const spans = detect(text, extra);
    const findings: MaskOutcome["findings"] = [];
    const counts: MaskOutcome["counts"] = {};
    const unverified: MaskOutcome["unverified"] = {};
    let out = "";
    let pos = 0;
    for (const s of spans) {
      const ph = this.placeholderFor(s.type, s.key, text.slice(s.start, s.end));
      out += text.slice(pos, s.start) + ph;
      pos = s.end;
      findings.push({ type: s.type, placeholder: ph });
      counts[s.type] = (counts[s.type] ?? 0) + 1;
      if (s.verified === false) unverified[s.type] = (unverified[s.type] ?? 0) + 1;
    }
    out += text.slice(pos);
    return { masked: out, findings, counts, unverified };
  }

  /** Replace this vault's placeholders with the original values. Unknown placeholders (including [S_...]) are left as they are. */
  unmask(text: string): string {
    return text.replace(PLACEHOLDER, (ph) => this.original.get(ph) ?? ph);
  }

  /** The user's real values held by this vault (memory only). Used to keep them out of links in the AI's reply (components/linkSafety.ts). */
  values(): string[] {
    return Array.from(new Set(this.original.values()));
  }

  /** True if `placeholder` was created by this vault (used to style pills). */
  has(placeholder: string): boolean {
    return this.original.has(placeholder);
  }

  get size(): number {
    return this.original.size;
  }

  clear(): void {
    this.byKey.clear();
    this.original.clear();
    this.counters.clear();
    this.reserved.clear();
  }

  private placeholderFor(type: PiiType, key: string, surface: string): string {
    const k = `${type}\u0000${key}`;
    const existing = this.byKey.get(k);
    if (existing) return existing;
    let n = this.counters.get(type) ?? 0;
    let ph: string;
    do {
      n++;
      ph = `[${type}_${n}]`;
    } while (this.reserved.has(ph));
    this.counters.set(type, n);
    this.byKey.set(k, ph);
    this.original.set(ph, surface);
    return ph;
  }
}
