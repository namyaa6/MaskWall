/**
 * Lowercase-name approaches (A, B, C, E): browser measurement and selection by the rule pre-registered in
 * docs/decisions.md Day 3.12 (written before any approach, evaluation sentence or measurement existed).
 *
 *   cd frontend; npx vite-node scripts/bench_names_lowercase.ts
 * Output: ml/results/ner_lowercase_eval.json and .md
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { chromium } from "@playwright/test";
import { createServer } from "vite";
import { evaluateNames, type NameGold } from "../src/pii/nerCore";

const ROOT = resolve(__dirname, "..");
const REPO = resolve(ROOT, "..");
type Row = NameGold & { set?: string };
type Pred = { start: number; end: number }[][];
const APPROACHES = ["A", "B", "C", "E"] as const;
const LABEL: Record<string, string> = {
  A: "A. NER as typed (baseline)",
  B: "B. NER on a selectively re-cased copy",
  C: "C. NER + context cues",
  E: "E. B + C",
};
// pre-registered rule (docs/decisions.md Day 3.12)
const RULE = { fpNegSentencesExtra: 3, fpNegSentencesMax: 6, fpExistingExtra: 2, maxMedianMs: 300, minGainPoints: 15, tieNames: 2 };
const SIMPLICITY: Record<string, number> = { B: 1, C: 2, E: 3 }; // fewer moving parts first

const pct = (xs: number[], p: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1) + 0.5))];
};
const r1 = (x: number) => Math.round(x * 10) / 10;
const read = (f: string): Row[] => readFileSync(join(REPO, f), "utf8").trim().split("\n").map((l) => JSON.parse(l));

function subset(rows: Row[], pred: Pred, keep: (r: Row) => boolean): { rows: Row[]; pred: Pred } {
  const idx = rows.map((r, i) => (keep(r) ? i : -1)).filter((i) => i >= 0);
  return { rows: idx.map((i) => rows[i]), pred: idx.map((i) => pred[i]) };
}

async function main() {
  const existing = read("ml/data/pii_names_sample.jsonl");
  const lowercase = read("ml/data/pii_names_lowercase.jsonl");
  const server = await createServer({ root: ROOT, configFile: join(ROOT, "vite.config.ts"), server: { port: 5197, strictPort: true }, logLevel: "warn" });
  await server.listen();
  const browser = await chromium.launch({ headless: true });
  let page: { results: Record<string, Record<string, { predicted: Pred; ms: number[] }>>; crossOriginIsolated: boolean; hardwareConcurrency: number };
  try {
    const p = await browser.newPage();
    p.on("pageerror", (e) => console.error("page error:", e.message));
    await p.goto("http://localhost:5197/bench/names.html");
    await p.waitForFunction(() => window.__result || window.__error, null, { timeout: 20 * 60_000, polling: 1000 });
    const err = await p.evaluate(() => window.__error);
    if (err) throw new Error(err);
    page = (await p.evaluate(() => window.__result)) as typeof page;
  } finally {
    await browser.close();
    await server.close();
  }

  const rows = APPROACHES.map((a) => {
    const ex = page.results[a].existing;
    const lc = page.results[a].lowercase;
    const exAll = evaluateNames(existing, ex.predicted);
    const exCap = subset(existing, ex.predicted, (r) => r.category !== "lowercase");
    const exCapEval = evaluateNames(exCap.rows, exCap.pred);
    const exLow = subset(existing, ex.predicted, (r) => r.category === "lowercase");
    const exLowEval = evaluateNames(exLow.rows, exLow.pred);
    const pos = subset(lowercase, lc.predicted, (r) => r.set === "pos");
    const neg = subset(lowercase, lc.predicted, (r) => r.set === "neg");
    const posEval = evaluateNames(pos.rows, pos.pred);
    const negEval = evaluateNames(neg.rows, neg.pred);
    const negFpSentences = neg.pred.filter((s) => s.length > 0).length;
    const ms = [...ex.ms, ...lc.ms];
    return {
      approach: a,
      label: LABEL[a],
      lowercase: { recall: posEval.strict, gold: posEval.gold, overlap: posEval.overlap, byGroup: posEval.byCategory, missed: posEval.missed, fpOnPositiveSentences: posEval.falsePositives },
      negatives: { fpSpans: negEval.falsePositives, fpSentences: negFpSentences, n: neg.rows.length, byCategory: negEval.byCategory, fps: negEval.fps.map((f) => ({ id: f.id, text: f.text })) },
      existing: {
        cap: { recall: exCapEval.strict, gold: exCapEval.gold, missed: exCapEval.missed },
        lowercaseCategory: { recall: exLowEval.strict, gold: exLowEval.gold },
        all: { recall: exAll.strict, gold: exAll.gold },
        fpSpans: exAll.falsePositives,
        fps: exAll.fps,
      },
      latency: { medianMs: r1(pct(ms, 0.5)), p95Ms: r1(pct(ms, 0.95)), n: ms.length },
    };
  });

  // ---------------------------------------------------------------- pre-registered rule
  const A = rows.find((r) => r.approach === "A")!;
  const checks = rows
    .filter((r) => r.approach !== "A")
    .map((r) => {
      const gain = ((r.lowercase.recall - A.lowercase.recall) / r.lowercase.gold) * 100;
      const c = {
        noRecallDrop: r.existing.cap.recall >= A.existing.cap.recall,
        fpNegatives: r.negatives.fpSentences <= A.negatives.fpSentences + RULE.fpNegSentencesExtra && r.negatives.fpSentences <= RULE.fpNegSentencesMax,
        fpExisting: r.existing.fpSpans <= A.existing.fpSpans + RULE.fpExistingExtra,
        latency: r.latency.medianMs <= RULE.maxMedianMs,
        worthShipping: gain >= RULE.minGainPoints,
      };
      return { approach: r.approach, gainPoints: Math.round(gain * 10) / 10, ...c, eligible: Object.values(c).every(Boolean) };
    });
  const eligible = checks.filter((c) => c.eligible).map((c) => rows.find((r) => r.approach === c.approach)!);
  let selected: string | null = null;
  if (eligible.length) {
    const best = Math.max(...eligible.map((r) => r.lowercase.recall));
    const near = eligible.filter((r) => best - r.lowercase.recall <= RULE.tieNames).sort((x, y) => SIMPLICITY[x.approach] - SIMPLICITY[y.approach]);
    selected = near[0].approach;
  }
  const out = {
    protocol: "docs/decisions.md Day 3.12 (written before any approach, evaluation sentence or measurement)",
    rule: RULE,
    sets: { existing: "ml/data/pii_names_sample.jsonl (AI-drafted, not reviewed by the owner)", lowercase: "ml/data/pii_names_lowercase.jsonl (AI-drafted, not reviewed by the owner)" },
    env: { crossOriginIsolated: page.crossOriginIsolated, hardwareConcurrency: page.hardwareConcurrency, runtime: "headless Chromium, Transformers.js 3.8.1, WASM, single thread, warm" },
    measured: new Date().toISOString(),
    selected,
    checks,
    rows,
  };
  writeFileSync(join(REPO, "ml/results/ner_lowercase_eval.json"), JSON.stringify(out, null, 2) + "\n");

  const g = (r: (typeof rows)[number], k: string) => r.lowercase.byGroup[k] ?? { gold: 0, strict: 0, fp: 0 };
  const md = [
    "# Lowercase names: approaches compared (browser, WASM)",
    "",
    `Protocol and rule: docs/decisions.md Day 3.12 (committed before any approach, evaluation sentence or measurement existed). Sets: ${out.sets.lowercase}; ${out.sets.existing}.`,
    `Headless Chromium, Transformers.js 3.8.1, single-threaded WASM, warm; latency = per sentence over all ${rows[0].latency.n} sentences (includes re-casing and cue rules).`,
    "",
    "| approach | lowercase recall (66 names) | cue | no_cue | ambiguous | false-positive sentences on 60 negatives | existing: capitalised recall (49) | existing: false-positive spans | median / p95 ms |",
    "|---|---|---|---|---|---|---|---|---|",
    ...rows.map(
      (r) =>
        `| ${r.label} | **${r.lowercase.recall}/${r.lowercase.gold}** | ${g(r, "cue").strict}/${g(r, "cue").gold} | ${g(r, "no_cue").strict}/${g(r, "no_cue").gold} | ${g(r, "ambiguous").strict}/${g(r, "ambiguous").gold} | ${r.negatives.fpSentences} | ${r.existing.cap.recall}/${r.existing.cap.gold} | ${r.existing.fpSpans} | ${r.latency.medianMs} / ${r.latency.p95Ms} |`,
    ),
    "",
    "## Pre-registered rule",
    "",
    "| approach | gain over A (points) | no recall drop | FP negatives (<= A+3 and <= 6) | FP existing (<= A+2) | median <= 300 ms | gain >= 15 | eligible |",
    "|---|---|---|---|---|---|---|---|",
    ...checks.map(
      (c) =>
        `| ${c.approach} | ${c.gainPoints} | ${c.noRecallDrop ? "yes" : "NO"} | ${c.fpNegatives ? "yes" : "NO"} | ${c.fpExisting ? "yes" : "NO"} | ${c.latency ? "yes" : "NO"} | ${c.worthShipping ? "yes" : "NO"} | **${c.eligible ? "yes" : "no"}** |`,
    ),
    "",
    `**Selected by the rule: ${selected ? LABEL[selected] : "none (browser keeps approach A)"}**`,
    "",
    "## False positives on the negatives",
    "",
    ...rows.flatMap((r) => [`**${r.label}** (${r.negatives.fpSentences} sentences, ${r.negatives.fpSpans} spans): ${r.negatives.fps.map((f) => `#${f.id} "${f.text}"`).join(", ") || "none"}`, ""]),
    "## Missed lowercase names",
    "",
    ...rows.flatMap((r) => [`**${r.label}** (${r.lowercase.missed.length}): ${r.lowercase.missed.map((m) => `#${m.id} ${m.name}`).join(", ") || "none"}`, ""]),
    "## Existing sample: false positives and misses of capitalised names",
    "",
    ...rows.flatMap((r) => [
      `**${r.label}**: FP ${r.existing.fps.map((f) => `#${f.id} "${f.text}"`).join(", ") || "none"}; missed (capitalised) ${r.existing.cap.missed.map((m) => `#${m.id} ${m.name}`).join(", ") || "none"}`,
      "",
    ]),
  ];
  writeFileSync(join(REPO, "ml/results/ner_lowercase_eval.md"), md.join("\n"));
  console.log(md.slice(0, 22).join("\n"));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
