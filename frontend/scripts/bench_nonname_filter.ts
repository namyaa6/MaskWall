/**
 * Day 4.24: did the "terms that are never a name" filter (src/pii/nonNames.ts) change name recall or false positives?
 *
 *   cd frontend; npx vite-node scripts/bench_nonname_filter.ts
 * Output: ml/results/name_nonnames_filter_check.json and .md   (a NEW file: the earlier evaluation results are not rewritten)
 *
 * Method: run the browser name model once per (approach, set) in headless Chromium (the same page as bench_names_r2.ts), keep the RAW
 * predicted spans, then score them twice in Node from the same raw output: as they are ("before") and after dropNonNames() ("after").
 * So the only difference between the two columns is the filter. Sets: the 62-sentence existing sample (all categories), the round-1
 * lowercase set (120), the round-2 lowercase set (120) and a few sentences with ID-document terms. Approaches: A (standard mode) and
 * B with the frozen word-list setting (Strict mode).
 *
 * Honesty notes: the round-2 set was sealed and evaluated once in Day 3.13 (docs/decisions.md). It is looked at again here ONLY to compare
 * the same raw output with and without a filter that was written from the owner's bug report before this run; nothing is tuned on it, no
 * setting changes, and the Day 3.13 result files and the frozen config are untouched.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { chromium } from "@playwright/test";
import { createServer } from "vite";
import type { CommonWordConfig } from "../src/pii/commonWords";
import type { PiiSpan } from "../src/pii/detectors";
import { evaluateNames, type NameGold } from "../src/pii/nerCore";
import { dropNonNames } from "../src/pii/nonNames";

const ROOT = resolve(__dirname, "..");
const REPO = resolve(ROOT, "..");
type Row = NameGold & { set?: string };
type Pred = { start: number; end: number }[][];
type Approach = "A" | "B";
interface Job { approach: Approach; config: CommonWordConfig | null; dataset: "existing" | "dev" | "fresh" | "custom"; texts?: string[] }
interface JobResult { predicted: Pred; ms: number[]; nWords: number }

const read = (f: string): Row[] => readFileSync(join(REPO, f), "utf8").trim().split("\n").map((l) => JSON.parse(l));
const CUSTOM = [
  "Aadhaar 1234 5768 9881 and PAN ABCDE1234F",                                   // the exact sentence from the report
  "My Aadhaar is 1234 5768 9881 and my PAN is ABCDE1234F.",
  "Rahul Sharma's Aadhaar 1234 5768 9881 and PAN ABCDE1234F",
  "Please verify my Passport and Voter ID, my UPI id and the OTP for KYC.",
  "GSTIN 27ABCDE1234F1Z5 and IFSC HDFC0001234 are on the invoice",
  "aadhaar 1234 5768 9881 and pan abcde1234f",
];

async function runJobs(jobs: Job[]): Promise<JobResult[]> {
  const server = await createServer({ root: ROOT, configFile: join(ROOT, "vite.config.ts"), server: { port: 5197, strictPort: true }, logLevel: "warn" });
  await server.listen();
  const browser = await chromium.launch({ headless: true });
  try {
    const p = await browser.newPage();
    p.on("pageerror", (e) => console.error("page error:", e.message));
    await p.goto("http://localhost:5197/bench/names_r2.html");
    await p.waitForFunction(() => window.__ready || window.__error, null, { timeout: 20 * 60_000, polling: 1000 });
    const err = await p.evaluate(() => window.__error);
    if (err) throw new Error(err);
    return (await p.evaluate((j) => window.__run!(j as never), jobs)) as JobResult[];
  } finally {
    await browser.close();
    await server.close();
  }
}

const withKeys = (text: string, spans: { start: number; end: number }[]): PiiSpan[] =>
  spans.map((s) => ({ start: s.start, end: s.end, type: "NAME" as const, key: text.slice(s.start, s.end).toLowerCase() }));
const filtered = (rows: Row[], pred: Pred): Pred => rows.map((r, i) => dropNonNames(r.text, withKeys(r.text, pred[i])).map((s) => ({ start: s.start, end: s.end })));

function score(rows: Row[], pred: Pred) {
  const sub = (keep: (r: Row) => boolean) => {
    const idx = rows.map((r, i) => (keep(r) ? i : -1)).filter((i) => i >= 0);
    return evaluateNames(idx.map((i) => rows[i]), idx.map((i) => pred[i]));
  };
  const all = evaluateNames(rows, pred);
  const neg = sub((r) => r.set === "neg");
  const pos = sub((r) => r.set !== "neg");
  return { gold: all.gold, strict: all.strict, overlap: all.overlap, falsePositiveSpans: all.falsePositives, flaggedNoNameSentences: all.fpOnNoNameSentences, posGold: pos.gold, posStrict: pos.strict, negFalsePositiveSpans: neg.falsePositives };
}

async function main() {
  const cfg = JSON.parse(readFileSync(join(ROOT, "src", "data", "common_words_config.json"), "utf8")) as CommonWordConfig;
  const config: CommonWordConfig = { tEn: cfg.tEn, kH: cfg.kH, share: cfg.share };
  const sets = { existing: read("ml/data/pii_names_sample.jsonl"), dev: read("ml/data/pii_names_lowercase.jsonl"), fresh: read("ml/data/pii_names_lowercase_r2.jsonl"), custom: CUSTOM.map((text, id): Row => ({ id, category: "custom", text, names: [] })) };
  const approaches: Approach[] = ["A", "B"];
  const jobs: Job[] = approaches.flatMap((a) => (Object.keys(sets) as (keyof typeof sets)[]).map((d) => ({ approach: a, config: a === "A" ? null : config, dataset: d, ...(d === "custom" ? { texts: CUSTOM } : {}) })));
  const results = await runJobs(jobs);

  const out: Record<string, unknown>[] = [];
  const removedAll: { approach: string; set: string; text: string; removed: string }[] = [];
  jobs.forEach((j, k) => {
    const rows = sets[j.dataset];
    const raw = results[k].predicted;
    const after = filtered(rows, raw);
    rows.forEach((r, i) => {
      const keep = new Set(after[i].map((s) => `${s.start}:${s.end}`));
      for (const s of raw[i]) if (!keep.has(`${s.start}:${s.end}`)) removedAll.push({ approach: j.approach === "A" ? "A (standard)" : "B2 (strict)", set: j.dataset, text: r.text, removed: r.text.slice(s.start, s.end) });
    });
    out.push({ approach: j.approach === "A" ? "A (standard mode)" : "B2 (strict mode)", set: j.dataset, sentences: rows.length, before: j.dataset === "custom" ? undefined : score(rows, raw), after: j.dataset === "custom" ? undefined : score(rows, after) });
  });
  const customRows = jobs.map((j, k) => ({ j, k })).filter((x) => x.j.dataset === "custom").map(({ j, k }) =>
    CUSTOM.map((text, i) => ({
      approach: j.approach === "A" ? "A (standard)" : "B2 (strict)",
      text,
      namesBefore: results[k].predicted[i].map((s) => text.slice(s.start, s.end)),
      namesAfter: filtered(sets.custom, results[k].predicted)[i].map((s) => text.slice(s.start, s.end)),
    })),
  ).flat();

  const lost = out.flatMap((o) => (o.before && o.after ? [{ approach: o.approach, set: o.set, recallBefore: (o.before as { strict: number }).strict, recallAfter: (o.after as { strict: number }).strict }] : [])).filter((x) => x.recallAfter < x.recallBefore);
  writeFileSync(join(REPO, "ml", "results", "name_nonnames_filter_check.json"), JSON.stringify({ note: "Day 4.24: same raw model output scored with and without the non-name filter; see the header of frontend/scripts/bench_nonname_filter.ts", model: "onnx-community/distilbert-NER-ONNX (q8, WASM)", frozenConfig: config, measured: new Date().toISOString(), recallDropped: lost, results: out, removedSpans: removedAll, customSentences: customRows }, null, 2) + "\n");

  const r = (o: Record<string, unknown>) => {
    const b = o.before as ReturnType<typeof score>;
    const a = o.after as ReturnType<typeof score>;
    return `| ${o.approach} | ${o.set} (${o.sentences}) | ${b.strict}/${b.gold} | ${a.strict}/${a.gold} | ${b.falsePositiveSpans} | ${a.falsePositiveSpans} | ${b.negFalsePositiveSpans || "-"} | ${a.negFalsePositiveSpans || "-"} |`;
  };
  const md = [
    "# Non-name filter check (Day 4.24)",
    "",
    "Same raw model output scored without (before) and with (after) `dropNonNames`. Strict recall = every character of a gold name covered.",
    "",
    "| Mode | Set (sentences) | Names found before | after | False-positive spans before | after | on harmless sentences before | after |",
    "|---|---|---|---|---|---|---|---|",
    ...out.filter((o) => o.set !== "custom").map(r),
    "",
    lost.length ? `**Recall DROPPED in: ${JSON.stringify(lost)}**` : "**Name recall did not drop in any set or mode.**",
    "",
    `Spans removed by the filter across all sets: ${removedAll.length}.`,
    ...removedAll.map((x) => `- ${x.approach}, ${x.set}: removed "${x.removed}" in "${x.text.slice(0, 90)}"`),
    "",
    "## Sentences with ID-document terms (no gold names; model output before -> after)",
    ...customRows.map((c) => `- ${c.approach}: "${c.text}": before ${JSON.stringify(c.namesBefore)}, after ${JSON.stringify(c.namesAfter)}`),
  ];
  writeFileSync(join(REPO, "ml", "results", "name_nonnames_filter_check.md"), md.join("\n") + "\n");
  console.log(md.join("\n"));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
