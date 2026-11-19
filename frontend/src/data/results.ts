// Typed access to the generated results (src/data/results.json, made by scripts/exportResults.ts from ml/results).
import data from "./results.json";

export const R = data;

export const pct = (x: number | null | undefined, digits = 1) => (x == null ? "–" : `${(x * 100).toFixed(digits)}%`);
export const f3 = (x: number | null | undefined) => (x == null ? "–" : x.toFixed(3));
export const kn = (o: { k: number; n: number }) => `${o.k} of ${o.n}`;
/** Duration for humans: 3.8 ms, 91 ms, 1.2 s, 6.8 s. */
export const dur = (x: number | null | undefined) =>
  x == null ? "–" : x < 10 ? `${x.toFixed(1)} ms` : x < 1000 ? `${Math.round(x)} ms` : `${(x / 1000).toFixed(1)} s`;
export const ms = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x)} ms`);
