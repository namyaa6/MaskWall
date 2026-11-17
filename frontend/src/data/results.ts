// Typed access to the generated results (src/data/results.json, made by scripts/exportResults.ts from ml/results).
import data from "./results.json";

export const R = data;

export const pct = (x: number | null | undefined, digits = 1) => (x == null ? "–" : `${(x * 100).toFixed(digits)}%`);
export const f3 = (x: number | null | undefined) => (x == null ? "–" : x.toFixed(3));
export const kn = (o: { k: number; n: number }) => `${o.k} of ${o.n}`;
export const ms = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x)} ms`);
