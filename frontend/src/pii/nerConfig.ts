// The browser name model, selected by the pre-registered rule in docs/decisions.md Day 3.2
// (measurements: ml/results/ner_browser_eval.md). Changing it means re-running scripts/bench_ner_browser.ts.
export const NER_MODEL = {
  id: "onnx-community/distilbert-NER-ONNX",
  // pinned to the commit that was measured (the repo was last changed 2025-09-01, before the 2026-11-17 evaluation): a changed or
  // compromised third-party repo must not be able to silently stop names from being masked
  revision: "3a19fe9404a4469d91aa3d551558a97f68872f67",
  approxMB: 66,
} as const;
