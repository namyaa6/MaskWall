import { describe, expect, it } from "vitest";
import { coveredTokens, selectWindows, thresholdsFromMeta, verdictFor, windowSpans } from "./windows";

// Same expectations as backend/tests/test_windowing.py, so the browser windows match the server's.
describe("windows (port of windowing.py)", () => {
  it("short input is one window", () => expect(windowSpans(100)).toEqual([[0, 100]]));
  it("exactly 510 tokens is one window", () => expect(windowSpans(510)).toEqual([[0, 510]]));
  it("long input: stride 255, last window aligned to the end", () => {
    expect(windowSpans(1000)).toEqual([[0, 510], [255, 765], [490, 1000]]);
  });
  it("any 255-token span lies fully inside some window", () => {
    for (const n of [600, 1000, 1920, 3000]) {
      const spans = windowSpans(n);
      for (let s = 0; s + 255 <= n; s += 7) expect(spans.some(([a, b]) => a <= s && s + 255 <= b)).toBe(true);
    }
  });
  it("cap: first (cap-1) windows plus the last, reported as partial", () => {
    const spans = windowSpans(3000);
    const sel = selectWindows(spans, 6);
    expect(sel.partial).toBe(true);
    expect(sel.spans).toEqual([...spans.slice(0, 5), spans[spans.length - 1]]);
    expect(selectWindows(windowSpans(1000), 6)).toEqual({ spans: windowSpans(1000), partial: false });
    expect(selectWindows(spans, 1)).toEqual({ spans: [spans[spans.length - 1]], partial: true });
  });
  it("covered tokens counts each position once", () => {
    expect(coveredTokens([[0, 510], [255, 765]])).toBe(765);
    expect(coveredTokens([[0, 10], [20, 30]])).toBe(20);
  });
  it("thresholds and verdicts follow the server policy", () => {
    const t = thresholdsFromMeta({ thr_f1: 0.368, thr_fpr1pct: 0.61 });
    expect(t).toEqual({ block: 0.61, warn: 0.368 });
    expect(verdictFor(0.61, "full", t)).toBe("block");
    expect(verdictFor(0.5, "full", t)).toBe("warn");
    expect(verdictFor(0.1, "full", t)).toBe("allow");
    expect(verdictFor(0.1, "partial", t)).toBe("warn");
  });
});
