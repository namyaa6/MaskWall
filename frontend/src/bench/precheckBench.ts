// Dev-only page (bench/precheck.html), driven by scripts/parity_browser.ts: the in-browser pre-check (WASM) on the server's
// golden prompts, in head_tail mode (how the fixture was made) and in sliding mode (what the app uses), with timings.
import raw from "../../../backend/tests/fixtures/golden_minilm.json?raw";
import { loadPrecheck, MODEL_DIR, type ModelMeta } from "../precheck/precheckModel";

async function main() {
  const golden = JSON.parse(raw) as { texts: string[]; scores: number[]; n_tokens: number[]; mode: string };
  const base = new URL("/models/", location.origin).href;
  const t0 = performance.now();
  const pc = await loadPrecheck({ base, loadMeta: async () => (await (await fetch(`${base}${MODEL_DIR}/model_meta.json`)).json()) as ModelMeta });
  const loadMs = performance.now() - t0;
  await pc.scan("warm-up");
  const rows = [];
  for (let i = 0; i < golden.texts.length; i++) {
    const ht = await pc.scan(golden.texts[i], "head_tail");
    const sl = await pc.scan(golden.texts[i], "sliding");
    rows.push({ i, nTokens: ht.nTokens, headTail: ht.score, headTailMs: ht.ms, sliding: sl.score, slidingMs: sl.ms, nWindows: sl.nWindows });
  }
  window.__result = { loadMs, thresholds: pc.thresholds, rows, crossOriginIsolated: self.crossOriginIsolated };
  document.getElementById("status")!.textContent = "done";
}

main().catch((e) => {
  window.__error = String(e?.stack ?? e);
});
