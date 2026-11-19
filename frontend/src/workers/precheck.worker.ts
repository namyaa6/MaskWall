/// <reference lib="webworker" />
// Attack pre-check worker: the served int8 MiniLM, cached by the browser. In production (VITE_PRECHECK_MODEL_ID set at build time) it is
// downloaded from the Hugging Face Hub, pinned to a commit; in development and in the e2e tests it comes from this site (/models/maskwall-minilm/).
import { loadPrecheck, MODEL_DIR, type ModelMeta, type Precheck } from "../precheck/precheckModel";
import { serveOrtLocally } from "./localOrt";
import type { FromWorker, PrecheckResult, ToWorker } from "./protocol";

serveOrtLocally();

const base = new URL(`${import.meta.env.BASE_URL}models/`, self.location.origin).href;
/** The commit of the Hub repo that matches the model the server runs (backend/scripts/fetch_model.py REVISION). */
const DEFAULT_REVISION = "78085d047839aef62e6bc4bad01dad4b4e9cf370";
const remoteId = (import.meta.env.VITE_PRECHECK_MODEL_ID as string | undefined)?.trim();
const remote = remoteId ? { modelId: remoteId, revision: (import.meta.env.VITE_PRECHECK_MODEL_REVISION as string | undefined)?.trim() || DEFAULT_REVISION } : undefined;
let model: Promise<Precheck> | null = null;
const post = (m: FromWorker<PrecheckResult>) => self.postMessage(m);

function load(): Promise<Precheck> {
  model ??= loadPrecheck({
    base,
    remote,
    loadMeta: async () => {
      const url = remote ? `https://huggingface.co/${remote.modelId}/resolve/${remote.revision}/model_meta.json` : `${base}${MODEL_DIR}/model_meta.json`;
      const r = await fetch(url);
      if (!r.ok) throw new Error(`model_meta.json: HTTP ${r.status}`);
      return (await r.json()) as ModelMeta;
    },
    onProgress: (loaded, total) => post({ type: "progress", loaded, total }),
  }).then((m) => {
    post({ type: "ready", info: { model: `${m.meta.name} (${m.meta.source})`, approxMB: 23, thresholds: m.thresholds } });
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
    post({ type: "result", id: msg.id, result: await m.scan(msg.text) });
  } catch (e) {
    post({ type: "error", id: msg.id, message: String((e as Error)?.message ?? e) });
  }
};
