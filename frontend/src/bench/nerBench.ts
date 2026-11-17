// Dev-only measurement page (bench/ner.html), driven by scripts/bench_ner_browser.ts in headless Chromium.
// Loads one NER model, times the load, then runs every sentence of the names sample and records spans and per-sentence time.
import raw from "../../../ml/data/pii_names_sample.jsonl?raw";
import { loadNer } from "../pii/nerModel";
import type { NameGold } from "../pii/nerCore";

declare global {
  interface Window {
    __result?: unknown;
    __error?: string;
  }
}

async function main() {
  const q = new URLSearchParams(location.search);
  const modelId = q.get("model")!;
  const rows = raw.trim().split("\n").map((l) => JSON.parse(l) as NameGold);
  let bytes = 0;
  const files = new Map<string, number>();
  const t0 = performance.now();
  const m = await loadNer(modelId, (p) => {
    files.set(p.file, p.total);
    bytes = [...files.values()].reduce((a, b) => a + b, 0);
  });
  const loadMs = performance.now() - t0;
  await m.detect("Warm-up sentence for Asha.");
  const ms: number[] = [];
  const predicted: { start: number; end: number }[][] = [];
  for (const r of rows) {
    const t = performance.now();
    const spans = await m.detect(r.text);
    ms.push(performance.now() - t);
    predicted.push(spans.map((s) => ({ start: s.start, end: s.end })));
  }
  window.__result = {
    model: modelId, loadMs, downloadedBytes: bytes, perSentenceMs: ms, predicted,
    crossOriginIsolated: self.crossOriginIsolated, hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent,
  };
  document.getElementById("status")!.textContent = "done";
}

main().catch((e) => {
  window.__error = String(e?.stack ?? e);
});
