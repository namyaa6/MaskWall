/**
 * In-browser copy of the server's injection scanner (backend/app/scanners/injection.py, sliding mode): the SAME int8
 * MiniLM (hard-negative arm B), the same windows and the same thresholds. It is an early warning only: the server
 * re-scans every message and makes the decision.
 *
 * Loaded from <base>/maskwall-minilm/ (copied there by scripts/prepare_browser_model.mjs). Used by the pre-check worker
 * (browser, WASM) and by the parity test (Node, onnxruntime-node), which compares its scores with the server's golden scores.
 */
import { AutoModelForSequenceClassification, AutoTokenizer, Tensor, env } from "@huggingface/transformers";
import type { PrecheckResult } from "./types";
import { CONTENT, coveredTokens, headTailIds, selectWindows, thresholdsFromMeta, verdictFor, windowSpans, type Thresholds } from "./windows";

export const MODEL_DIR = "maskwall-minilm";

export interface ModelMeta {
  name: string;
  source: string;
  cls_id: number;
  sep_id: number;
  thr_f1: number;
  thr_fpr1pct: number;
}

export type { PrecheckResult };

export interface Precheck {
  meta: ModelMeta;
  thresholds: Thresholds;
  scan(text: string, mode?: "sliding" | "head_tail"): Promise<PrecheckResult>;
}

/** Load from the Hugging Face Hub instead of from `base` (production: the model is not part of the site). */
export interface RemoteModel {
  modelId: string;   // e.g. namya06/maskwall-injection-minilm (both layouts live at the repo root)
  revision: string;  // a commit: the same model version the server runs (backend/scripts/fetch_model.py)
}

export interface LoadOptions {
  /** URL path (browser) or directory (Node) that contains maskwall-minilm/ (ignored when `remote` is set) */
  base: string;
  remote?: RemoteModel;
  loadMeta: () => Promise<ModelMeta>;
  device?: "wasm" | "cpu";
  onProgress?: (loaded: number, total: number) => void;
}

const softmaxP1 = (a: number, b: number) => {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  return eb / (ea + eb);
};

export async function loadPrecheck(opts: LoadOptions): Promise<Precheck> {
  const modelId = opts.remote ? opts.remote.modelId : MODEL_DIR;
  env.allowLocalModels = !opts.remote;
  env.allowRemoteModels = !!opts.remote;
  env.localModelPath = opts.base.endsWith("/") ? opts.base : opts.base + "/";
  const revision = opts.remote?.revision;
  const progress_callback = (e: { status: string; loaded?: number; total?: number }) => {
    if (e.status === "progress" && opts.onProgress) opts.onProgress(e.loaded ?? 0, e.total ?? 0);
  };
  const [meta, tokenizer, model] = await Promise.all([
    opts.loadMeta(),
    AutoTokenizer.from_pretrained(modelId, revision ? { revision } : {}),
    AutoModelForSequenceClassification.from_pretrained(modelId, { dtype: "q8", device: opts.device ?? "wasm", progress_callback, ...(revision ? { revision } : {}) }),
  ]);
  const thresholds = thresholdsFromMeta(meta);

  async function scoreIds(content: number[]): Promise<number> {
    const ids = [meta.cls_id, ...content, meta.sep_id];
    const input_ids = new Tensor("int64", BigInt64Array.from(ids, (x) => BigInt(x)), [1, ids.length]);
    const attention_mask = new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]);
    const { logits } = (await model({ input_ids, attention_mask })) as { logits: Tensor };
    const d = logits.data as Float32Array;
    return softmaxP1(d[0], d[1]);
  }

  return {
    meta,
    thresholds,
    async scan(text: string, mode: "sliding" | "head_tail" = "sliding"): Promise<PrecheckResult> {
      const t0 = performance.now();
      const ids: number[] = text.trim() ? tokenizer.encode(text, { add_special_tokens: false }) : [];
      if (ids.length === 0) return { score: 0, verdict: "allow", coverage: "full", nTokens: 0, nWindows: 0, scannedTokens: 0, ms: performance.now() - t0 };
      if (mode === "head_tail") {
        // used by the parity test against the head_tail golden fixture; the app uses sliding (the server default)
        const content = headTailIds(ids);
        const score = await scoreIds(content);
        const coverage = ids.length > CONTENT ? "partial" : "full";
        return { score, verdict: verdictFor(score, coverage, thresholds), coverage, nTokens: ids.length, nWindows: 1, scannedTokens: content.length, ms: performance.now() - t0 };
      }
      const { spans, partial } = selectWindows(windowSpans(ids.length));
      let score = 0;
      for (const [s, e] of spans) score = Math.max(score, await scoreIds(ids.slice(s, e)));
      const coverage = partial ? "partial" : "full";
      return {
        score, verdict: verdictFor(score, coverage, thresholds), coverage, nTokens: ids.length, nWindows: spans.length,
        scannedTokens: coveredTokens(spans), ms: performance.now() - t0,
      };
    },
  };
}
