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

describe("hosting numbers (Day 4.8 and 4.11)", () => {
  it("are labelled with the hardware and keep the measured CPU quota next to the nominal 0.1 vCPU", async () => {
    const { R } = await import("./results");
    expect(R.hosting.label).toBe("Render free tier (0.1 vCPU)");
    expect(R.hosting.cgroup_vcpu).toBe(0.15);
    expect(R.hosting.memory_limit_mb).toBe(512);
  });

  it("report both measurement conditions and the memory gate", async () => {
    const { R } = await import("./results");
    const h = R.hosting;
    expect(h.sporadic.short.p50_ms).toBeLessThan(10);                    // spaced requests: not throttled
    expect(h.back_to_back.short.p50_ms).toBeGreaterThan(50);             // back-to-back: throttled, and the 50 ms rule is NOT met
    expect(h.sporadic.document.p50_ms).toBeGreaterThan(h.sporadic.window.p50_ms);
    expect(h.cgroup_peak_mb).toBeLessThan(h.memory_limit_mb);             // the gate: the served model fits in memory
  });

  it("formats durations for humans", async () => {
    const { dur } = await import("./results");
    expect(dur(3.8)).toBe("3.8 ms");
    expect(dur(91.9)).toBe("92 ms");
    expect(dur(1170.3)).toBe("1.2 s");
    expect(dur(6815.4)).toBe("6.8 s");
    expect(dur(null)).toBe("–");
  });
});
