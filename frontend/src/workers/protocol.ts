import type { PiiSpan } from "../pii/detectors";
import type { PrecheckResult, Thresholds } from "../precheck/types";

// Messages between the UI thread and the two model workers. Each worker owns its own Transformers.js `env`
// (the name model comes from the Hugging Face Hub, the attack model from this site), so they never share settings.

export type ToWorker = { type: "load" } | { type: "run"; id: number; text: string };

export type FromWorker<R> =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; info: ModelInfo }
  | { type: "result"; id: number; result: R }
  | { type: "error"; id?: number; message: string };

export interface ModelInfo {
  model: string;
  approxMB: number;
  thresholds?: Thresholds;
}

export type NerResult = PiiSpan[];
export type { PrecheckResult };
