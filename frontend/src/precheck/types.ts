// Types shared by the pre-check worker and the UI (kept free of Transformers.js imports so the UI bundle stays small).
export type { Thresholds, Verdict } from "./windows";
import type { Verdict } from "./windows";

export interface PrecheckResult {
  score: number;
  verdict: Verdict;
  coverage: "full" | "partial";
  nTokens: number;
  nWindows: number;
  scannedTokens: number;
  ms: number;
}
