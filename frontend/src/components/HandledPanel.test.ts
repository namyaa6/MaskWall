/** Day 4.19: a message refused right after tokenising (too long or too dense to scan completely) has NO score: the panel must say so instead of showing "0.00 out of 1". */
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ChatResponse, InputSection } from "../api";
import { UNSCORED_HEADLINE, UNSCORED_NEXT_STEP } from "../explain";
import { HandledPanel, type HandledTurn } from "./HandledPanel";

const input = (over: Partial<InputSection>): InputSection => ({
  score: 0.01, scored: true, verdict: "allow", reasons: [], notes: [], hidden_characters_removed: 0, coverage: "full", mode: "sliding", n_tokens: 12,
  n_windows: 1, n_windows_needed: 1, scanned_tokens: 12, thresholds: { warn: 0.368, block: 0.61 }, pii_masked: {}, masked_input: "hello", windows: [], ...over,
});
const turn = (res: Partial<ChatResponse> & { input: InputSection }): HandledTurn => ({
  privacyOn: true, namesChecked: true, counts: {}, sent: "hello", maskMs: 5, precheck: undefined, status: "done",
  res: { request_id: "x", action: "allow", response: "hi", latency_ms: { pii_ms: 0, tokenize_ms: 1, inference_ms: 0, total_ms: 3 }, truncated: false, output: null, llm: null, ...res } as ChatResponse,
});
const html = (t: HandledTurn) => renderToStaticMarkup(createElement(HandledPanel, { turn: t }));

describe("HandledPanel, step 2 (checked for attacks)", () => {
  it("an unscored, refused message shows the explanation and no 'out of 1' score and no early-warning line", () => {
    const out = html(turn({ action: "block", response: "blocked", input: input({ score: null, scored: false, verdict: "block", reasons: ["partial_scan"], coverage: "partial", n_windows: 0, n_tokens: 2348, n_windows_needed: 9, scanned_tokens: 0 }) }));
    expect(out).toContain("Not scored.");
    expect(out).toContain(UNSCORED_HEADLINE);
    expect(out).toContain(UNSCORED_NEXT_STEP);
    expect(out).not.toContain("out of 1");
    expect(out).not.toContain("Early warning on your device");
    expect(out).not.toContain("0.00");
  });

  it("a scored message still shows its score and the early-warning line", () => {
    const out = html(turn({ input: input({ score: 0.0123 }) }));
    expect(out).toContain("0.01 out of 1");
    expect(out).toContain("Early warning on your device");
    expect(out).not.toContain("Not scored");
  });

  it("while the message is still being checked, the early-warning line shows 'running'", () => {
    const t: HandledTurn = { privacyOn: true, namesChecked: true, counts: {}, sent: "hello", maskMs: 5, status: "sending" };
    expect(html(t)).toContain("Early warning on your device");
  });
});
