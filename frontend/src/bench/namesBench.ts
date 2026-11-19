// Dev-only measurement page (bench/names.html), driven by scripts/bench_names_lowercase.ts in headless Chromium.
// Loads the NER model once, then runs every name approach (A, B, C, E) over the existing names sample and the lowercase set.
import existingRaw from "../../../ml/data/pii_names_sample.jsonl?raw";
import lowercaseRaw from "../../../ml/data/pii_names_lowercase.jsonl?raw";
import type { NameGold } from "../pii/nerCore";
import { NER_MODEL } from "../pii/nerConfig";
import { detectNames, type NameApproach } from "../pii/nameApproaches";
import { loadNer } from "../pii/nerModel";

declare global {
  interface Window {
    __result?: unknown;
    __error?: string;
  }
}

const parse = (raw: string) => raw.trim().split("\n").map((l) => JSON.parse(l) as NameGold & { set?: string });

async function main() {
  const datasets = { existing: parse(existingRaw), lowercase: parse(lowercaseRaw) };
  const model = await loadNer(NER_MODEL.id, undefined, NER_MODEL.revision);
  await model.detect("Warm-up sentence for Asha.");
  const out: Record<string, Record<string, { predicted: { start: number; end: number }[][]; ms: number[] }>> = {};
  for (const approach of ["A", "B", "C", "E"] as NameApproach[]) {
    out[approach] = {};
    for (const [name, rows] of Object.entries(datasets)) {
      const predicted: { start: number; end: number }[][] = [];
      const ms: number[] = [];
      for (const r of rows) {
        const t = performance.now();
        const spans = await detectNames(approach, r.text, model);
        ms.push(performance.now() - t);
        predicted.push(spans.map((s) => ({ start: s.start, end: s.end })));
      }
      out[approach][name] = { predicted, ms };
    }
  }
  window.__result = { results: out, crossOriginIsolated: self.crossOriginIsolated, hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent };
  document.getElementById("status")!.textContent = "done";
}

main().catch((e) => {
  window.__error = String(e?.stack ?? e);
});
