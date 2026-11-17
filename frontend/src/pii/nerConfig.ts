// The browser name model, selected by the pre-registered rule in docs/decisions.md Day 3.2
// (measurements: ml/results/ner_browser_eval.md). Changing it means re-running scripts/bench_ner_browser.ts.
export const NER_MODEL = {
  id: "onnx-community/distilbert-NER-ONNX",
  approxMB: 66,
} as const;
