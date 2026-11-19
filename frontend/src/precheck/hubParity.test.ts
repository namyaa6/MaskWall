/**
 * The model the DEPLOYED site downloads (Hugging Face Hub, pinned commit) must load and score like the server's golden scores.
 * This is the check that production's pre-check would work, since the model is not part of the site. Needs network, so it only runs
 * when RUN_HUB_TESTS=1 (CI sets it; locally: RUN_HUB_TESTS=1 npm test). Runs in Node (onnxruntime-node) on the same ONNX file the
 * browser would download.
 *
 * int8 scores are not bit-identical across CPUs (docs/decisions.md Day 4.9), so scores are compared in logit space (tolerance 0.5)
 * and the allow/warn/block verdict must be IDENTICAL. Token counts must match exactly (that is the tokenizer check).
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadPrecheck, type ModelMeta } from "./precheckModel";
import { verdictFor } from "./windows";

const MODEL_ID = "namya06/maskwall-injection-minilm";
const REVISION = "78085d047839aef62e6bc4bad01dad4b4e9cf370"; // keep equal to DEFAULT_REVISION in workers/precheck.worker.ts and backend/scripts/fetch_model.py
const golden = JSON.parse(readFileSync(new URL("../../../backend/tests/fixtures/golden_minilm.json", import.meta.url), "utf8")) as {
  texts: string[];
  scores: number[];
  n_tokens: number[];
};
const logit = (p: number) => {
  const q = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
  return Math.log(q / (1 - q));
};

describe.skipIf(process.env.RUN_HUB_TESTS !== "1")("pre-check model from the Hugging Face Hub (what production downloads)", () => {
  it("loads from the pinned commit and matches the server's golden scores and verdicts", async () => {
    const pc = await loadPrecheck({
      base: "",
      remote: { modelId: MODEL_ID, revision: REVISION },
      device: "cpu",
      loadMeta: async () => {
        const r = await fetch(`https://huggingface.co/${MODEL_ID}/resolve/${REVISION}/model_meta.json`);
        expect(r.ok).toBe(true);
        return (await r.json()) as ModelMeta;
      },
    });
    expect(pc.thresholds.block).toBeCloseTo(0.61, 3);
    expect(pc.thresholds.warn).toBeCloseTo(0.368, 3);
    let maxLogit = 0;
    for (let i = 0; i < golden.texts.length; i++) {
      const r = await pc.scan(golden.texts[i], "head_tail");
      expect(r.nTokens, `token count, text ${i}`).toBe(golden.n_tokens[i]);
      const cov = r.nTokens > 510 ? "partial" : "full";
      expect(verdictFor(r.score, cov, pc.thresholds), `verdict, text ${i}`).toBe(verdictFor(golden.scores[i], cov, pc.thresholds));
      maxLogit = Math.max(maxLogit, Math.abs(logit(r.score) - logit(golden.scores[i])));
    }
    console.log(`hub model parity: max |logit difference| vs the server's golden scores = ${maxLogit.toFixed(3)}`);
    expect(maxLogit).toBeLessThanOrEqual(0.5);
  }, 300_000);
});
