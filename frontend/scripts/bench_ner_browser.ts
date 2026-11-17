/**
 * Browser NER model selection (protocol: docs/decisions.md Day 3.2, written before this ran).
 * Starts a Vite dev server, opens bench/ner.html in headless Chromium (persistent profile, so the second run per model
 * is a warm load from the browser cache), scores the names sample and applies the pre-registered rule.
 *
 *   cd frontend; npx vite-node scripts/bench_ner_browser.ts
 * Output: ml/results/ner_browser_eval.json and .md
 */
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "@playwright/test";
import { createServer } from "vite";
import { evaluateNames, type NameGold } from "../src/pii/nerCore";

const ROOT = resolve(__dirname, "..");
const REPO = resolve(ROOT, "..");
const CANDIDATES = [
  { id: "onnx-community/distilbert-NER-ONNX", q8MB: 65.8, license: "Apache-2.0", data: "CoNLL-2003 (English)" },
  { id: "Xenova/bert-base-NER", q8MB: 109.0, license: "MIT", data: "CoNLL-2003 (English)" },
  { id: "Xenova/distilbert-base-multilingual-cased-ner-hrl", q8MB: 135.4, license: "AFL-3.0", data: "10 high-resource languages (no Hindi)" },
];
const RULE = { maxMB: 120, maxMedianMs: 300, tieNames: 2 };

interface PageResult {
  model: string; loadMs: number; downloadedBytes: number; perSentenceMs: number[];
  predicted: { start: number; end: number }[][]; crossOriginIsolated: boolean; hardwareConcurrency: number; userAgent: string;
}

const pct = (xs: number[], p: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1) + 0.5))];
};
const r1 = (x: number) => Math.round(x * 10) / 10;

async function main() {
  const rows: NameGold[] = readFileSync(join(REPO, "ml/data/pii_names_sample.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const server = await createServer({ root: ROOT, configFile: join(ROOT, "vite.config.ts"), server: { port: 5199, strictPort: true }, logLevel: "warn" });
  await server.listen();
  const profile = mkdtempSync(join(tmpdir(), "maskwall-ner-"));
  const ctx = await chromium.launchPersistentContext(profile, { headless: true });
  const results = [];
  try {
    for (const c of CANDIDATES) {
      const runs: Record<string, PageResult> = {};
      for (const phase of ["cold", "warm"]) {
        const page = await ctx.newPage();
        page.on("pageerror", (e) => console.error(`[${c.id}] page error:`, e.message));
        await page.goto(`http://localhost:5199/bench/ner.html?model=${encodeURIComponent(c.id)}&phase=${phase}`);
        await page.waitForFunction(() => window.__result || window.__error, null, { timeout: 15 * 60_000, polling: 500 });
        const err = await page.evaluate(() => window.__error);
        if (err) throw new Error(`${c.id} ${phase}: ${err}`);
        runs[phase] = (await page.evaluate(() => window.__result)) as PageResult;
        await page.close();
        console.log(`${c.id} ${phase}: load ${Math.round(runs[phase].loadMs)} ms, median ${r1(pct(runs[phase].perSentenceMs, 0.5))} ms`);
      }
      const same = JSON.stringify(runs.cold.predicted) === JSON.stringify(runs.warm.predicted);
      const ev = evaluateNames(rows, runs.warm.predicted);
      results.push({
        ...c,
        coldLoadMs: Math.round(runs.cold.loadMs), warmLoadMs: Math.round(runs.warm.loadMs),
        downloadedMB: r1(runs.cold.downloadedBytes / 1e6),
        medianMs: r1(pct(runs.warm.perSentenceMs, 0.5)), p95Ms: r1(pct(runs.warm.perSentenceMs, 0.95)),
        coldMedianMs: r1(pct(runs.cold.perSentenceMs, 0.5)),
        coldWarmPredictionsIdentical: same, eval: ev,
        env: { crossOriginIsolated: runs.warm.crossOriginIsolated, hardwareConcurrency: runs.warm.hardwareConcurrency, userAgent: runs.warm.userAgent },
      });
    }
  } finally {
    await ctx.close();
    await server.close();
    rmSync(profile, { recursive: true, force: true });
  }

  // ---------------------------------------------------------------- pre-registered rule
  const eligible = results.filter((r) => r.q8MB <= RULE.maxMB && r.medianMs <= RULE.maxMedianMs);
  let selected: (typeof results)[number] | null = null;
  if (eligible.length) {
    const best = Math.max(...eligible.map((r) => r.eval.strict));
    const near = eligible.filter((r) => best - r.eval.strict <= RULE.tieNames).sort((a, b) => a.q8MB - b.q8MB);
    selected = near[0];
  }
  const out = {
    protocol: "docs/decisions.md Day 3.2 (written before measuring)", rule: RULE, sample: "ml/data/pii_names_sample.jsonl (AI-drafted, not reviewed by the owner)",
    n_sentences: rows.length, n_names: results[0]?.eval.gold, transformers_js: "3.8.1", measured: new Date().toISOString(),
    selected: selected?.id ?? null, results,
  };
  writeFileSync(join(REPO, "ml/results/ner_browser_eval.json"), JSON.stringify(out, null, 2) + "\n");

  const md: string[] = [
    "# Browser name (NER) model: measurement and selection",
    "",
    `Protocol and rule: docs/decisions.md Day 3.2 (written before this ran). Sample: ${out.sample}, ${rows.length} sentences, ${out.n_names} names.`,
    `Headless Chromium, Transformers.js 3.8.1, WASM, q8 weights; crossOriginIsolated=${results[0]?.env.crossOriginIsolated} (single-threaded WASM), ${results[0]?.env.hardwareConcurrency} logical cores. Latency = per sentence, warm run, after one warm-up.`,
    "",
    "| model | q8 MB | cold load s | warm load s | median ms | p95 ms | strict recall | any-overlap recall | false positives (on no-name sentences) | eligible |",
    "|---|---|---|---|---|---|---|---|---|---|",
    ...results.map((r) => `| \`${r.id}\` | ${r.q8MB} | ${(r.coldLoadMs / 1000).toFixed(1)} | ${(r.warmLoadMs / 1000).toFixed(1)} | ${r.medianMs} | ${r.p95Ms} | ${r.eval.strict}/${r.eval.gold} | ${r.eval.overlap}/${r.eval.gold} | ${r.eval.falsePositives} (${r.eval.fpOnNoNameSentences}) | ${eligible.includes(r) ? "yes" : "no"} |`),
    "",
    `**Selected by the rule: ${selected ? "`" + selected.id + "`" : "none (fallback rule applies)"}**`,
    "",
    "## Strict recall by category",
    "",
    "| model | " + Object.keys(results[0].eval.byCategory).join(" | ") + " |",
    "|---|" + Object.keys(results[0].eval.byCategory).map(() => "---").join("|") + "|",
    ...results.map((r) => `| \`${r.id}\` | ` + Object.values(r.eval.byCategory).map((c) => `${c.strict}/${c.gold}${c.fp ? ` (fp ${c.fp})` : ""}`).join(" | ") + " |"),
    "",
    "## Missed names and false positives",
    "",
    ...results.flatMap((r) => [
      `**${r.id}**`,
      "",
      `- missed (${r.eval.missed.length}): ${r.eval.missed.map((m) => `#${m.id} ${m.name}`).join(", ") || "none"}`,
      `- false positives (${r.eval.fps.length}): ${r.eval.fps.map((f) => `#${f.id} "${f.text}"`).join(", ") || "none"}`,
      "",
    ]),
  ];
  writeFileSync(join(REPO, "ml/results/ner_browser_eval.md"), md.join("\n"));
  console.log(md.slice(0, 12).join("\n"));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
