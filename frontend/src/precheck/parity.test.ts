/**
 * Parity: the browser pre-check must score exactly like the server. For the 25 golden prompts of the served model
 * (backend/tests/fixtures/golden_minilm.json, server pipeline, sliding windows) the token counts must be identical and the
 * scores within 1e-3. Runs in Node (onnxruntime-node) on the same ONNX file the browser loads (WASM); skipped when the
 * gitignored model copy is absent (run `node scripts/prepare_browser_model.mjs`).
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { loadPrecheck, type ModelMeta } from "./precheckModel";
import { verdictFor } from "./windows";

const base = fileURLToPath(new URL("../../public/models/", import.meta.url));
const metaPath = `${base}maskwall-minilm/model_meta.json`;
const golden = JSON.parse(readFileSync(new URL("../../../backend/tests/fixtures/golden_minilm.json", import.meta.url), "utf8")) as {
  texts: string[];
  scores: number[];
  n_tokens: number[];
  mode: string;
};
const servedMeta = new URL("../../../backend/model/minilm/model_meta.json", import.meta.url);

describe.skipIf(!existsSync(metaPath))("browser pre-check vs server golden scores", () => {
  it("copied model is the served model", () => {
    const a = JSON.parse(readFileSync(metaPath, "utf8"));
    const b = JSON.parse(readFileSync(servedMeta, "utf8"));
    expect(a).toEqual(b);
  });

  it("same token counts and scores within 1e-3 on all golden prompts", async () => {
    const pc = await loadPrecheck({ base, device: "cpu", loadMeta: async () => JSON.parse(readFileSync(metaPath, "utf8")) as ModelMeta });
    expect(pc.thresholds.block).toBeCloseTo(0.61, 3);
    expect(pc.thresholds.warn).toBeCloseTo(0.368, 3);
    expect(golden.mode).toBe("head_tail");     // the fixture was generated in head_tail mode: compare like with like
    let maxDiff = 0;
    for (let i = 0; i < golden.texts.length; i++) {
      const r = await pc.scan(golden.texts[i], "head_tail");
      expect(r.nTokens, `token count, text ${i}`).toBe(golden.n_tokens[i]);
      const cov = r.nTokens > 510 ? "partial" : "full";
      expect(verdictFor(r.score, cov, pc.thresholds), `verdict, text ${i}`).toBe(verdictFor(golden.scores[i], cov, pc.thresholds));
      const d = Math.abs(r.score - golden.scores[i]);
      if (d > 1e-3) console.log(`text ${i} (${r.nTokens} tokens): browser ${r.score.toFixed(5)} server ${golden.scores[i].toFixed(5)}`);
      maxDiff = Math.max(maxDiff, d);
    }
    console.log(`parity: ${golden.texts.length} golden texts (${golden.mode}), max |browser - server| = ${maxDiff.toExponential(2)}`);
    expect(maxDiff).toBeLessThanOrEqual(1e-3);
  }, 120_000);
});
