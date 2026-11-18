// Dev-only measurement page (bench/names_r2.html), driven by scripts/bench_names_r2.ts in headless Chromium.
// Loads the NER model once, then runs jobs {approach, config, dataset}. The sealed round-2 test set ("fresh") is imported lazily and
// ONLY when a job asks for it, so a development ("tune") run never touches it.
import devRaw from "../../../ml/data/pii_names_lowercase.jsonl?raw";
import existingRaw from "../../../ml/data/pii_names_sample.jsonl?raw";
import type { NameGold } from "../pii/nerCore";
import { NER_MODEL } from "../pii/nerConfig";
import { buildCommonWords, type CommonWordConfig, type CommonWordStats } from "../pii/commonWords";
import { detectNames, type NameApproach } from "../pii/nameApproaches";
import { loadNer } from "../pii/nerModel";

export interface Job {
  approach: NameApproach;
  config: CommonWordConfig | null;     // null: no word set (approach A ignores it; round-1 behaviour otherwise)
  dataset: "existing" | "dev" | "fresh";
}
export interface JobResult {
  predicted: { start: number; end: number }[][];
  ms: number[];
  nWords: number;
}

declare global {
  interface Window {
    __ready?: boolean;
    __error?: string;
    __run?: (jobs: Job[]) => Promise<JobResult[]>;
  }
}

const parse = (raw: string) => raw.trim().split("\n").map((l) => JSON.parse(l) as NameGold & { set?: string });

async function loadFresh(): Promise<string> {
  const mods = import.meta.glob("../../../ml/data/pii_names_lowercase_r2.jsonl", { query: "?raw", import: "default" });
  const loader = Object.values(mods)[0];
  if (!loader) throw new Error("fresh set not found");
  return (await loader()) as string;
}

async function loadStats(): Promise<CommonWordStats> {
  const mods = import.meta.glob("../../../ml/cache/common_words_stats.json", { import: "default" });
  const loader = Object.values(mods)[0];
  if (!loader) throw new Error("ml/cache/common_words_stats.json not found: run ml/data/build_common_words.py");
  return (await loader()) as CommonWordStats;
}

async function main() {
  const stats = await loadStats();
  const model = await loadNer(NER_MODEL.id);
  await model.detect("Warm-up sentence for Asha.");
  const cache = new Map<string, Set<string>>();
  const wordsFor = (c: CommonWordConfig | null) => {
    if (!c) return undefined;
    const k = `${c.tEn}/${c.kH}/${c.share}`;
    if (!cache.has(k)) cache.set(k, buildCommonWords(stats, c));
    return cache.get(k)!;
  };
  let freshRows: (NameGold & { set?: string })[] | null = null;
  window.__run = async (jobs: Job[]) => {
    const out: JobResult[] = [];
    for (const job of jobs) {
      let rows: (NameGold & { set?: string })[];
      if (job.dataset === "existing") rows = parse(existingRaw);
      else if (job.dataset === "dev") rows = parse(devRaw);
      else rows = freshRows ??= parse(await loadFresh());
      const words = wordsFor(job.config);
      const predicted: { start: number; end: number }[][] = [];
      const ms: number[] = [];
      for (const r of rows) {
        const t = performance.now();
        const spans = await detectNames(job.approach, r.text, model, words);
        ms.push(performance.now() - t);
        predicted.push(spans.map((s) => ({ start: s.start, end: s.end })));
      }
      out.push({ predicted, ms, nWords: words?.size ?? 0 });
    }
    return out;
  };
  window.__ready = true;
  document.getElementById("status")!.textContent = "ready";
}

main().catch((e) => {
  window.__error = String(e?.stack ?? e);
});
