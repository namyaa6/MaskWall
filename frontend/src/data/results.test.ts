import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { buildResults, RESULTS_PATH } from "../../scripts/exportResults";

const repo = fileURLToPath(new URL("../../../", import.meta.url));

describe("results.json (numbers shown on How it works and Benchmarks)", () => {
  it("is up to date with ml/results (run: npm run export-results)", () => {
    const committed = JSON.parse(readFileSync(RESULTS_PATH(repo), "utf8"));
    expect(committed).toEqual(JSON.parse(JSON.stringify(buildResults(repo))));
  });

  it.skipIf(!existsSync(`${repo}backend/model/minilm/model_meta.json`))("describes the served model with the thresholds the backend uses", () => {
    const meta = JSON.parse(readFileSync(`${repo}backend/model/minilm/model_meta.json`, "utf8"));
    const r = buildResults(repo);
    expect(r.served.source).toBe(meta.source);
    expect(r.served.block).toBeCloseTo(Math.max(meta.thr_f1, meta.thr_fpr1pct), 4);
  });
});
