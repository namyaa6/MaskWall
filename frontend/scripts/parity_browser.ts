/**
 * Browser (WASM) parity of the attack pre-check with the server, on the 25 golden prompts of the served model.
 * Pass: identical token counts, max |browser - server| <= 1e-3 in head_tail mode (how the fixture was made), and the same
 * verdict (allow / warn / block) as the server's score at the served thresholds. Also records load time and latency.
 *
 *   cd frontend; node scripts/prepare_browser_model.mjs; npx vite-node scripts/parity_browser.ts
 * Output: ml/results/browser_precheck_parity.json
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { chromium } from "@playwright/test";
import { createServer } from "vite";
import { verdictFor } from "../src/precheck/windows";

const ROOT = resolve(__dirname, "..");
const REPO = resolve(ROOT, "..");

interface Row { i: number; nTokens: number; headTail: number; headTailMs: number; sliding: number; slidingMs: number; nWindows: number }

async function main() {
  const golden = JSON.parse(readFileSync(join(REPO, "backend/tests/fixtures/golden_minilm.json"), "utf8"));
  const server = await createServer({ root: ROOT, configFile: join(ROOT, "vite.config.ts"), server: { port: 5198, strictPort: true }, logLevel: "warn" });
  await server.listen();
  const browser = await chromium.launch({ headless: true });
  let res: { loadMs: number; thresholds: { warn: number; block: number }; rows: Row[]; crossOriginIsolated: boolean };
  try {
    const page = await browser.newPage();
    page.on("pageerror", (e) => console.error("page error:", e.message));
    await page.goto("http://localhost:5198/bench/precheck.html");
    await page.waitForFunction(() => window.__result || window.__error, null, { timeout: 10 * 60_000, polling: 500 });
    const err = await page.evaluate(() => window.__error);
    if (err) throw new Error(err);
    res = (await page.evaluate(() => window.__result)) as typeof res;
  } finally {
    await browser.close();
    await server.close();
  }
  const t = res.thresholds;
  let maxDiff = 0;
  let countMismatch = 0;
  let verdictMismatch = 0;
  for (const r of res.rows) {
    maxDiff = Math.max(maxDiff, Math.abs(r.headTail - golden.scores[r.i]));
    if (r.nTokens !== golden.n_tokens[r.i]) countMismatch++;
    const cov = r.nTokens > 510 ? "partial" : "full";
    if (verdictFor(r.headTail, cov, t) !== verdictFor(golden.scores[r.i], cov, t)) verdictMismatch++;
  }
  const short = res.rows.filter((r) => r.nTokens <= 510).map((r) => r.slidingMs).sort((a, b) => a - b);
  const long = res.rows.filter((r) => r.nTokens > 510);
  const out = {
    runtime: "headless Chromium, Transformers.js 3.8.1 (onnxruntime-web), WASM, single-threaded",
    crossOriginIsolated: res.crossOriginIsolated, n: res.rows.length, thresholds: t,
    maxAbsDiffHeadTail: maxDiff, tokenCountMismatches: countMismatch, verdictMismatches: verdictMismatch,
    pass: maxDiff <= 1e-3 && countMismatch === 0 && verdictMismatch === 0,
    loadMs: Math.round(res.loadMs),
    shortMedianMs: Math.round(short[Math.floor(short.length / 2)] * 10) / 10,
    long: long.map((r) => ({ tokens: r.nTokens, windows: r.nWindows, ms: Math.round(r.slidingMs) })),
    rows: res.rows.map((r) => ({ ...r, server: golden.scores[r.i] })),
  };
  writeFileSync(join(REPO, "ml/results/browser_precheck_parity.json"), JSON.stringify(out, null, 2) + "\n");
  console.log(JSON.stringify({ ...out, rows: undefined }, null, 2));
  if (!out.pass) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
