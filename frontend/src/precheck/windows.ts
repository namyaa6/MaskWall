/**
 * Port of backend/app/scanners/windowing.py (sliding mode, the server's default). Windows of 510 content tokens with a
 * stride of 255; the last window is aligned to the end; over the cap, the first (cap-1) windows plus the last one are
 * scanned and coverage is "partial". Same numbers as the server so the browser score matches the server score.
 */
export const CONTENT = 510;
export const STRIDE = 255;
export const MAX_WINDOWS = 6;

export type Span = [number, number];

export const HEAD = 256;
export const TAIL = 254;

/** head_tail mode (the training-time truncation): first 256 + last 254 tokens when longer than 510. */
export function headTailIds(ids: number[], head = HEAD, tail = TAIL): number[] {
  return ids.length > head + tail ? [...ids.slice(0, head), ...ids.slice(-tail)] : [...ids];
}

export function windowSpans(n: number, content = CONTENT, stride = STRIDE): Span[] {
  if (n <= content) return [[0, n]];
  const spans: Span[] = [];
  let s = 0;
  while (s + content < n) {
    spans.push([s, s + content]);
    s += stride;
  }
  spans.push([n - content, n]);
  return spans;
}

export function selectWindows(spans: Span[], maxWindows = MAX_WINDOWS): { spans: Span[]; partial: boolean } {
  if (maxWindows < 1) throw new Error("maxWindows must be >= 1");
  if (spans.length <= maxWindows) return { spans, partial: false };
  if (maxWindows === 1) return { spans: [spans[spans.length - 1]], partial: true };
  return { spans: [...spans.slice(0, maxWindows - 1), spans[spans.length - 1]], partial: true };
}

export function coveredTokens(spans: Span[]): number {
  let covered = 0;
  let end = 0;
  for (const [s0, e] of [...spans].sort((a, b) => a[0] - b[0])) {
    const s = Math.max(s0, end);
    if (e > s) {
      covered += e - s;
      end = e;
    }
  }
  return covered;
}

export interface Thresholds {
  warn: number;
  block: number;
}

/** Same rule as backend/app/policy.py thresholds_from_meta: block = max, warn = min of the two validation thresholds. */
export function thresholdsFromMeta(meta: { thr_f1: number; thr_fpr1pct: number }): Thresholds {
  return { block: Math.max(meta.thr_f1, meta.thr_fpr1pct), warn: Math.min(meta.thr_f1, meta.thr_fpr1pct) };
}

export type Verdict = "allow" | "warn" | "block";

/** Score-and-coverage part of backend/app/policy.py decide_input (the server adds Unicode-smuggling and strict-mode rules). */
export function verdictFor(score: number, coverage: "full" | "partial", t: Thresholds): Verdict {
  if (score >= t.block) return "block";
  if (score >= t.warn || coverage === "partial") return "warn";
  return "allow";
}
