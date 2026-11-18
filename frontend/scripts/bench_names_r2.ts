/**
 * Lowercase names, ROUND 2 (docs/decisions.md Day 3.13). Two modes:
 *
 *   npx vite-node scripts/bench_names_r2.ts tune    development only: the 8 pre-registered word-list settings, approach B, on the
 *                                                     round-1 (now "seen") set + the existing sample. Never loads the sealed set.
 *                                                     Writes ml/results/ner_lowercase_r2_dev.* and freezes the chosen setting in
 *                                                     frontend/src/data/common_words_config.json.
 *   npx vite-node scripts/bench_names_r2.ts final   ONE evaluation of the frozen setting on the sealed round-2 set (Gemini) + the
 *                                                     existing sample, approaches A, B, C, E; applies the pre-registered rule by code.
 *                                                     Writes ml/results/ner_lowercase_r2_eval.*. Refuses to run twice.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { chromium } from "@playwright/test";
import { createServer } from "vite";
import { GRID, type CommonWordConfig } from "../src/pii/commonWords";
import { evaluateNames, type NameGold } from "../src/pii/nerCore";

const ROOT = resolve(__dirname, "..");
const REPO = resolve(ROOT, "..");
type Row = NameGold & { set?: string };
type Pred = { start: number; end: number }[][];
type Approach = "A" | "B" | "C" | "E";
interface Job { approach: Approach; config: CommonWordConfig | null; dataset: "existing" | "dev" | "fresh" }
interface JobResult { predicted: Pred; ms: number[]; nWords: number }

const FROZEN = join(ROOT, "src", "data", "common_words_config.json");
const FINAL_OUT = join(REPO, "ml", "results", "ner_lowercase_r2_eval.json");
// pre-registered rule, round 2 (docs/decisions.md Day 3.13)
const RULE = { fpNegSentencesMax: 6, fpExistingExtra: 2, maxMedianMs: 300, minGainPoints: 15 };
const LABEL: Record<Approach, string> = { A: "A. NER as typed (baseline)", B: "B2. NER on a re-cased copy, independent word list", C: "C2. NER + context cues (info only)", E: "E2. B2 + C2 (info only)" };

const read = (f: string): Row[] => readFileSync(join(REPO, f), "utf8").trim().split("\n").map((l) => JSON.parse(l));
const pct = (xs: number[], p: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1) + 0.5))];
};
const r1 = (x: number) => Math.round(x * 10) / 10;
function subset(rows: Row[], pred: Pred, keep: (r: Row) => boolean) {
  const idx = rows.map((r, i) => (keep(r) ? i : -1)).filter((i) => i >= 0);
  return { rows: idx.map((i) => rows[i]), pred: idx.map((i) => pred[i]) };
}

async function runJobs(jobs: Job[]): Promise<{ results: JobResult[]; env: { crossOriginIsolated: boolean; hardwareConcurrency: number } }> {
  const server = await createServer({ root: ROOT, configFile: join(ROOT, "vite.config.ts"), server: { port: 5196, strictPort: true }, logLevel: "warn" });
  await server.listen();
  const browser = await chromium.launch({ headless: true });
  try {
    const p = await browser.newPage();
    p.on("pageerror", (e) => console.error("page error:", e.message));
    await p.goto("http://localhost:5196/bench/names_r2.html");
    await p.waitForFunction(() => window.__ready || window.__error, null, { timeout: 20 * 60_000, polling: 1000 });
    const err = await p.evaluate(() => window.__error);
    if (err) throw new Error(err);
    const results = (await p.evaluate((j) => window.__run!(j as never), jobs)) as JobResult[];
    const env = await p.evaluate(() => ({ crossOriginIsolated: self.crossOriginIsolated, hardwareConcurrency: navigator.hardwareConcurrency }));
    return { results, env };
  } finally {
    await browser.close();
    await server.close();
  }
}

function score(existing: Row[], exPred: Pred, set: Row[], setPred: Pred, ms: number[]) {
  const exAll = evaluateNames(existing, exPred);
  const exCap = subset(existing, exPred, (r) => r.category !== "lowercase");
  const exCapEval = evaluateNames(exCap.rows, exCap.pred);
  const pos = subset(set, setPred, (r) => r.set === "pos");
  const neg = subset(set, setPred, (r) => r.set === "neg");
  const posEval = evaluateNames(pos.rows, pos.pred);
  const negEval = evaluateNames(neg.rows, neg.pred);
  return {
    lowercase: { recall: posEval.strict, gold: posEval.gold, byGroup: posEval.byCategory, missed: posEval.missed },
    negatives: { fpSentences: neg.pred.filter((s) => s.length > 0).length, fpSpans: negEval.falsePositives, n: neg.rows.length, fps: negEval.fps.map((f) => ({ id: f.id, text: f.text })) },
    existing: { capRecall: exCapEval.strict, capGold: exCapEval.gold, capMissed: exCapEval.missed, fpSpans: exAll.falsePositives, fps: exAll.fps },
    latency: { medianMs: r1(pct(ms, 0.5)), p95Ms: r1(pct(ms, 0.95)), n: ms.length },
  };
}

async function tune() {
  if (existsSync(FROZEN)) throw new Error(`${FROZEN} already exists: the configuration is frozen; delete it only to start a new round`);
  const existing = read("ml/data/pii_names_sample.jsonl");
  const dev = read("ml/data/pii_names_lowercase.jsonl");
  const jobs: Job[] = [
    { approach: "A", config: null, dataset: "existing" },
    { approach: "A", config: null, dataset: "dev" },
    ...GRID.flatMap((config) => [{ approach: "B" as const, config, dataset: "existing" as const }, { approach: "B" as const, config, dataset: "dev" as const }]),
  ];
  const { results } = await runJobs(jobs);
  const A = score(existing, results[0].predicted, dev, results[1].predicted, [...results[0].ms, ...results[1].ms]);
  const rows = GRID.map((config, i) => {
    const ex = results[2 + 2 * i];
    const dv = results[3 + 2 * i];
    const s = score(existing, ex.predicted, dev, dv.predicted, [...ex.ms, ...dv.ms]);
    const ok = s.negatives.fpSentences <= RULE.fpNegSentencesMax && s.existing.capRecall >= A.existing.capRecall && s.existing.fpSpans <= A.existing.fpSpans + RULE.fpExistingExtra;
    return { config, nWords: ex.nWords, ...s, meetsLimits: ok };
  });
  // objective (fixed in the plan): maximise dev lowercase recall among settings that meet the limits; ties: fewer flagged sentences, then fewer words.
  // If none meets the limits: the fewest flagged sentences (then higher recall).
  const ok = rows.filter((r) => r.meetsLimits);
  const pool = ok.length ? ok : rows;
  const best = [...pool].sort((x, y) => (ok.length ? y.lowercase.recall - x.lowercase.recall : 0) || x.negatives.fpSentences - y.negatives.fpSentences || y.lowercase.recall - x.lowercase.recall || x.nWords - y.nWords)[0];
  writeFileSync(join(REPO, "ml/results/ner_lowercase_r2_dev.json"), JSON.stringify({ note: "development run on the round-1 (seen) set; the sealed round-2 set was not loaded", rule: RULE, baselineA: A, rows, chosen: best.config, noneMetLimits: !ok.length, measured: new Date().toISOString() }, null, 2) + "\n");
  const md = [
    "# Lowercase names round 2: development run (round-1 set, now \"seen\"; the sealed round-2 set was NOT loaded)",
    "",
    `Baseline A on the development set: ${A.lowercase.recall}/${A.lowercase.gold} lowercase names, ${A.negatives.fpSentences} flagged harmless sentences; existing sample capitalised recall ${A.existing.capRecall}/${A.existing.capGold}, ${A.existing.fpSpans} false-positive spans.`,
    "",
    "| T_en | K_h | share | words | dev lowercase recall (66) | dev flagged harmless (of 60) | existing capitalised recall | existing FP spans | meets limits |",
    "|---|---|---|---|---|---|---|---|---|",
    ...rows.map((r) => `| ${r.config.tEn} | ${r.config.kH} | ${r.config.share} | ${r.nWords} | ${r.lowercase.recall} | ${r.negatives.fpSentences} | ${r.existing.capRecall}/${r.existing.capGold} | ${r.existing.fpSpans} | ${r.meetsLimits ? "yes" : "no"} |`),
    "",
    `**Chosen and frozen: T_en = ${best.config.tEn}, K_h = ${best.config.kH}, share = ${best.config.share}** ${ok.length ? "(best dev recall among settings meeting the limits)" : "(NO setting met the limits on the development set: the one with the fewest flagged sentences)"}.`,
  ];
  writeFileSync(join(REPO, "ml/results/ner_lowercase_r2_dev.md"), md.join("\n") + "\n");
  writeFileSync(FROZEN, JSON.stringify({ ...best.config, chosenOn: "round-1 development set (ml/data/pii_names_lowercase.jsonl) + existing sample; never on the round-2 test set", noneMetLimitsOnDev: !ok.length, frozen: new Date().toISOString() }, null, 2) + "\n");
  console.log(md.join("\n"));
}

async function final() {
  if (existsSync(FINAL_OUT)) throw new Error(`${FINAL_OUT} exists: the sealed set is evaluated once; a second look voids the round`);
  const cfg = JSON.parse(readFileSync(FROZEN, "utf8")) as CommonWordConfig & { frozen: string };
  const existing = read("ml/data/pii_names_sample.jsonl");
  const fresh = read("ml/data/pii_names_lowercase_r2.jsonl");
  const config: CommonWordConfig = { tEn: cfg.tEn, kH: cfg.kH, share: cfg.share };
  const approaches: Approach[] = ["A", "B", "C", "E"];
  const jobs: Job[] = approaches.flatMap((a) => [{ approach: a, config: a === "A" ? null : config, dataset: "existing" as const }, { approach: a, config: a === "A" ? null : config, dataset: "fresh" as const }]);
  const { results, env } = await runJobs(jobs);
  const rows = approaches.map((a, i) => ({ approach: a, label: LABEL[a], nWords: results[2 * i].nWords, ...score(existing, results[2 * i].predicted, fresh, results[2 * i + 1].predicted, [...results[2 * i].ms, ...results[2 * i + 1].ms]) }));
  const A = rows[0];
  const B = rows[1];
  const gain = ((B.lowercase.recall - A.lowercase.recall) / B.lowercase.gold) * 100;
  const checks = {
    fpNegatives: B.negatives.fpSentences <= RULE.fpNegSentencesMax,
    noRecallDrop: B.existing.capRecall >= A.existing.capRecall,
    fpExisting: B.existing.fpSpans <= A.existing.fpSpans + RULE.fpExistingExtra,
    latency: B.latency.medianMs < RULE.maxMedianMs,
    worthShipping: gain >= RULE.minGainPoints,
  };
  const eligible = Object.values(checks).every(Boolean);
  writeFileSync(FINAL_OUT, JSON.stringify({ protocol: "docs/decisions.md Day 3.13", rule: RULE, config: cfg, env, measured: new Date().toISOString(), gainPoints: Math.round(gain * 10) / 10, checks, b2Eligible: eligible, rows }, null, 2) + "\n");
  const g = (r: (typeof rows)[number], k: string) => r.lowercase.byGroup[k] ?? { gold: 0, strict: 0, fp: 0 };
  const md = [
    "# Lowercase names, round 2: ONE evaluation on the sealed Gemini set (browser, WASM)",
    "",
    `Protocol and rule: docs/decisions.md Day 3.13 (committed before the set was generated). Frozen word-list setting: T_en ${cfg.tEn}, K_h ${cfg.kH}, share ${cfg.share} (${rows[1].nWords} words plus the NER vocabulary).`,
    "",
    "| approach | lowercase names (60) | cue | no_cue | word_name | harmless sentences flagged (of 60) | existing capitalised recall (49) | existing FP spans | median / p95 ms |",
    "|---|---|---|---|---|---|---|---|---|",
    ...rows.map((r) => `| ${r.label} | **${r.lowercase.recall}/${r.lowercase.gold}** | ${g(r, "cue").strict}/${g(r, "cue").gold} | ${g(r, "no_cue").strict}/${g(r, "no_cue").gold} | ${g(r, "word_name").strict}/${g(r, "word_name").gold} | ${r.negatives.fpSentences} | ${r.existing.capRecall}/${r.existing.capGold} | ${r.existing.fpSpans} | ${r.latency.medianMs} / ${r.latency.p95Ms} |`),
    "",
    "## B2 against the pre-registered rule",
    "",
    "| check | limit | B2 | passes |",
    "|---|---|---|---|",
    `| harmless sentences flagged | <= ${RULE.fpNegSentencesMax} of 60 | ${B.negatives.fpSentences} | ${checks.fpNegatives ? "yes" : "NO"} |`,
    `| capitalised-name recall on the existing sample | >= A (${A.existing.capRecall}) | ${B.existing.capRecall} | ${checks.noRecallDrop ? "yes" : "NO"} |`,
    `| false-positive spans on the existing sample | <= A + ${RULE.fpExistingExtra} (${A.existing.fpSpans + RULE.fpExistingExtra}) | ${B.existing.fpSpans} | ${checks.fpExisting ? "yes" : "NO"} |`,
    `| median latency | < ${RULE.maxMedianMs} ms | ${B.latency.medianMs} ms | ${checks.latency ? "yes" : "NO"} |`,
    `| gain in lowercase recall over A | >= ${RULE.minGainPoints} points | ${Math.round(gain * 10) / 10} points | ${checks.worthShipping ? "yes" : "NO"} |`,
    "",
    `**B2 is ${eligible ? "ELIGIBLE: it ships" : "NOT eligible: nothing ships, negative result"}.**`,
    "",
    "## Flagged harmless sentences and missed names",
    "",
    ...rows.flatMap((r) => [`**${r.label}**: FP on negatives (${r.negatives.fpSentences} sentences): ${r.negatives.fps.map((f) => `#${f.id} "${f.text}"`).join(", ") || "none"}; missed lowercase names (${r.lowercase.missed.length}): ${r.lowercase.missed.map((m) => `#${m.id} ${m.name}`).join(", ") || "none"}`, ""]),
  ];
  writeFileSync(join(REPO, "ml/results/ner_lowercase_r2_eval.md"), md.join("\n") + "\n");
  console.log(md.slice(0, 26).join("\n"));
}

const mode = process.argv[2] ?? process.argv[process.argv.length - 1];
(mode === "tune" ? tune() : mode === "final" ? final() : Promise.reject(new Error("usage: bench_names_r2.ts tune|final"))).catch((e) => {
  console.error(e);
  process.exit(1);
});
