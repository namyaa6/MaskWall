/**
 * Loads a token-classification (NER) model with Transformers.js (WASM backend) and returns PER spans for a text.
 * Runs inside a Web Worker in the app (src/workers/privacy.worker.ts) and directly in the measurement page (bench/ner.html).
 */
import { AutoModelForTokenClassification, AutoTokenizer, env, type Tensor } from "@huggingface/transformers";
import type { PiiSpan } from "./detectors";
import { chunkText, personSpans } from "./nerCore";

env.allowLocalModels = false;  // name models come from the Hugging Face Hub (cached by the browser after the first download)

export interface LoadProgress {
  file: string;
  loaded: number;
  total: number;
}

export interface NerModel {
  id: string;
  detect(text: string): Promise<PiiSpan[]>;
  /** is this exact string a whole token of the model's vocabulary? (used to tell common words from names) */
  hasToken(word: string): boolean;
}

interface ProgressEvent {
  status: string;
  file?: string;
  loaded?: number;
  total?: number;
}

export async function loadNer(modelId: string, onProgress?: (p: LoadProgress) => void): Promise<NerModel> {
  const progress_callback = (e: ProgressEvent) => {
    if (e.status === "progress" && e.file && onProgress) onProgress({ file: e.file, loaded: e.loaded ?? 0, total: e.total ?? 0 });
  };
  const tokenizer = await AutoTokenizer.from_pretrained(modelId, { progress_callback });
  const model = await AutoModelForTokenClassification.from_pretrained(modelId, { dtype: "q8", device: "wasm", progress_callback });
  const id2label = (model.config as unknown as { id2label: Record<string, string> }).id2label;
  const toTokens = (ids: number[]): string[] =>
    (tokenizer as unknown as { model: { convert_ids_to_tokens(ids: number[]): string[] } }).model.convert_ids_to_tokens(ids);

  async function detectChunk(text: string, offset: number): Promise<PiiSpan[]> {
    if (!text.trim()) return [];
    const enc = tokenizer(text, { truncation: true, max_length: 512 }) as unknown as Record<string, Tensor>;
    const ids = Array.from(enc.input_ids.data as BigInt64Array, Number);
    const { logits } = (await model(enc)) as { logits: Tensor };
    const [, T, L] = logits.dims;
    const data = logits.data as Float32Array;
    const labels: string[] = [];
    for (let t = 0; t < T; t++) {
      let best = 0;
      for (let l = 1; l < L; l++) if (data[t * L + l] > data[t * L + best]) best = l;
      labels.push(id2label[String(best)]);
    }
    return personSpans(text, toTokens(ids), labels, offset);
  }

  const tokensToIds = (tokenizer as unknown as { model: { tokens_to_ids: Map<string, number> } }).model.tokens_to_ids;

  return {
    id: modelId,
    hasToken: (word: string) => tokensToIds.has(word),
    async detect(text: string) {
      const out: PiiSpan[] = [];
      for (const c of chunkText(text)) out.push(...(await detectChunk(c.text, c.offset)));
      return out;
    },
  };
}
