/// <reference lib="webworker" />
// Name-detection worker: the NER model chosen in docs/decisions.md Day 3.2, downloaded once from the Hugging Face Hub
// and cached by the browser. Only the model files are downloaded; the text never leaves this worker.
import { detectNamesForMode } from "../pii/nameMode";
import { NER_MODEL } from "../pii/nerConfig";
import { loadNer, type NerModel } from "../pii/nerModel";
import { serveOrtLocally } from "./localOrt";
import type { FromWorker, NerResult, ToWorker } from "./protocol";

serveOrtLocally();

let model: Promise<NerModel> | null = null;

// The word list for Strict name hiding (approach B2) is a separate chunk, fetched only the first time strict mode is used.
let strictWords: Promise<ReadonlySet<string>> | null = null;
const loadStrictWords = () => (strictWords ??= import("../data/commonWordsB2.json").then((m) => new Set<string>(m.default.words)));
const post = (m: FromWorker<NerResult>) => self.postMessage(m);

function load(): Promise<NerModel> {
  const files = new Map<string, { loaded: number; total: number }>();
  model ??= loadNer(NER_MODEL.id, (p) => {
    files.set(p.file, { loaded: p.loaded, total: p.total });
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    post({ type: "progress", loaded, total });
  }, NER_MODEL.revision).then((m) => {
    post({ type: "ready", info: { model: NER_MODEL.id, approxMB: NER_MODEL.approxMB } });
    return m;
  });
  model.catch((e) => {
    model = null;
    post({ type: "error", message: String(e?.message ?? e) });
  });
  return model;
}

self.onmessage = async (ev: MessageEvent<ToWorker>) => {
  const msg = ev.data;
  if (msg.type === "load") {
    load().catch(() => undefined);
    return;
  }
  try {
    const m = await load();
    const t0 = performance.now();
    const spans = await detectNamesForMode(msg.strict ? "strict" : "standard", msg.text, m, loadStrictWords);
    post({ type: "result", id: msg.id, result: { spans, ms: performance.now() - t0 } });
  } catch (e) {
    post({ type: "error", id: msg.id, message: String((e as Error)?.message ?? e) });
  }
};
