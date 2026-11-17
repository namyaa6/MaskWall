/// <reference lib="webworker" />
// Attack pre-check worker: the served int8 MiniLM, loaded from this site (/models/maskwall-minilm/), cached by the browser.
import { loadPrecheck, MODEL_DIR, type ModelMeta, type Precheck } from "../precheck/precheckModel";
import type { FromWorker, PrecheckResult, ToWorker } from "./protocol";

const base = new URL(`${import.meta.env.BASE_URL}models/`, self.location.origin).href;
let model: Promise<Precheck> | null = null;
const post = (m: FromWorker<PrecheckResult>) => self.postMessage(m);

function load(): Promise<Precheck> {
  model ??= loadPrecheck({
    base,
    loadMeta: async () => {
      const r = await fetch(`${base}${MODEL_DIR}/model_meta.json`);
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
