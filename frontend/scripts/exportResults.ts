/**
 * Builds frontend/src/data/results.json from ml/results/*.json. Every number on the How it works and Benchmarks pages comes
 * from here; none is typed by hand. The generated file is committed (the Vercel build does not need ml/), and
 * src/data/results.test.ts fails if it is stale.
 *
 *   cd frontend; npm run export-results      (scripts/export_results_cli.ts)
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

type J = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

const r4 = (x: number | null | undefined) => (x == null ? null : Math.round(x * 10000) / 10000);
const r1 = (x: number | null | undefined) => (x == null ? null : Math.round(x * 10) / 10);

function trickyTotal(t: J): { correct: number; total: number } {
  let correct = 0;
  let total = 0;
  for (const [k, v] of Object.entries(t)) {
    if (k.startsWith("_")) continue;
    correct += (v as J).correct;
    total += (v as J).total;
  }
  return { correct, total };
}

function classifierRow(name: string, m: J) {
  const t = m.test;
  const tr = trickyTotal(m.tricky);
  return {
    model: name,
    f1: r4(t.f1),
    f1_ci: t.ci95?.f1?.map(r4) ?? null,
    precision: r4(t.precision),
    recall: r4(t.recall),
    fpr: r4(t.fpr),
    fp: t.confusion_matrix?.fp ?? null,
    benign: t.confusion_matrix ? t.confusion_matrix.tn + t.confusion_matrix.fp : null,
    attacks: t.confusion_matrix ? t.confusion_matrix.tp + t.confusion_matrix.fn : null,
    roc_auc: r4(t.roc_auc),
    tricky_correct: tr.correct,
    tricky_total: tr.total,
    xstest_fp: m.xstest_safe?.false_positives ?? null,
    xstest_n: m.xstest_safe?.n ?? null,
  };
}

export function buildResults(repo: string) {
  const read = (f: string): J => JSON.parse(readFileSync(join(repo, "ml", "results", f), "utf8"));
  const metrics = read("metrics.json");
  const trans = read("metrics_transformers.json");
  const bench = read("benchmark.json");
  const hard = read("hardneg_int8_eval.json");
  const pg = read("prompt_guard_reference.json");
  const ner = read("ner_browser_eval.json");
  const parity = read("browser_precheck_parity.json");

  const BASE_NAMES: Record<string, string> = {
    length_lr: "Length only + logistic regression (floor)",
    word_tfidf: "Word TF-IDF + logistic regression",
    wordchar: "Word + character TF-IDF + logistic regression",
    minilm_emb: "Frozen MiniLM embeddings + logistic regression",
  };
  const baselines = Object.entries(metrics.baselines.cleaned_aug as J).map(([k, v]) => classifierRow(BASE_NAMES[k] ?? k, v));
  const transformers = Object.entries(trans.transformers.cleaned_aug as J).map(([k, v]) => ({
    ...classifierRow(k === "minilm" ? "MiniLM-L6 fine-tuned" : k === "deberta_v3_small" ? "DeBERTa-v3-small fine-tuned" : k, v),
    params_m: (v as J).n_params_m ?? null,
  }));

  const quantization = (["cleaned_aug/minilm", "cleaned_aug/deberta_v3_small"] as const).flatMap((job) => {
    const b = bench[job];
    return (["fp32", "int8"] as const).map((prec) => {
      const p = b[prec];
      const t = p.accuracy?.at_val_threshold?.test ?? {};
      return {
        model: job.endsWith("minilm") ? "MiniLM-L6" : "DeBERTa-v3-small",
        precision: prec,
        size_mb: r1(p.size_mb),
        p50_ms: r1(p.latency.natural.p50_ms),
        p50_512_ms: r1(p.latency.len512.p50_ms),
        f1: r4(t.f1),
        fpr: r4(t.fpr),
      };
    });
  });

  const outcomes = Object.entries(hard.outcomes as J).map(([name, o]) => ({
    set: name,
    n: o.n,
    before: o.served,
    after: o.candidate,
    p: r4(o.p),
    kind: o.kind as "fp" | "hit",
  }));
  const oc = (prefix: string) => outcomes.find((o) => o.set.startsWith(prefix))!;
  const served = hard.candidate_meta;

  const pgRow = (key: "pg22" | "pg86", point: "shipped_0.5" | "val_f1_opt") => {
    const p = pg[key].points[point];
    return {
      model: key === "pg22" ? "Prompt Guard 2 (22M)" : "Prompt Guard 2 (86M)",
      point: point === "shipped_0.5" ? "as shipped (0.5)" : "tuned on our validation set",
      threshold: r4(p.threshold),
      int8_mb: r1(pg[key].info.int8_mb),
      test_f1: r4(p.test.f1),
      test_fpr: r4(p.test.fpr),
      test_recall: r4(p.test.recall),
      indirect_recall: r4(p.indirect_test.recall),
      xstest_fp: p.xstest_fp,
      hinglish_benign_fp: p.human_hinglish_benign_fp.k,
      prompt_talk_fp: p.prompt_talk_benign_fp.k,
      hinglish_attacks_caught: p.ai_hinglish_attacks_caught.k,
    };
  };
  const testAttacks = oc("test attacks");
  const testBenign = oc("test benign");
  const ours = {
    model: "MiniLM int8 (ours, served)",
    point: "served block threshold",
    threshold: r4(Math.max(served.thr_f1, served.thr_fpr1pct)),
    int8_mb: r1(hard.export_report.int8_mb),
    test_f1: r4(hard.test_f1.candidate),
    test_fpr: r4(testBenign.after / testBenign.n),
    test_recall: r4(testAttacks.after / testAttacks.n),
    indirect_recall: r4(oc("indirect_test attacks").after / oc("indirect_test attacks").n),
    xstest_fp: oc("xstest_safe").after,
    hinglish_benign_fp: oc("E1 human Hinglish").after,
    prompt_talk_fp: oc("E2 prompt-talk").after,
    hinglish_attacks_caught: oc("E3 Hinglish").after,
  };

  return {
    _generated_by: "frontend/scripts/exportResults.ts from ml/results (do not edit by hand)",
    served: {
      source: served.source,
      block: r4(Math.max(served.thr_f1, served.thr_fpr1pct)),
      warn: r4(Math.min(served.thr_f1, served.thr_fpr1pct)),
      int8_mb: r1(hard.export_report.int8_mb),
      fp32_mb: r1(hard.export_report.fp32_mb),
      params_m: hard.export_report.params_m,
      test_f1: r4(hard.test_f1.candidate),
      test_recall: { k: testAttacks.after, n: testAttacks.n },
      test_fp: { k: testBenign.after, n: testBenign.n },
      indirect_recall: { k: oc("indirect_test attacks").after, n: oc("indirect_test attacks").n },
      xstest_fp: { k: oc("xstest_safe").after, n: oc("xstest_safe").n },
      hinglish_benign_fp: { k: oc("E1 human Hinglish").after, n: oc("E1 human Hinglish").n },
      prompt_talk_fp: { k: oc("E2 prompt-talk").after, n: oc("E2 prompt-talk").n },
      hinglish_attacks: { k: oc("E3 Hinglish").after, n: oc("E3 Hinglish").n },
      long_benign_fp: { k: oc("long benign").after, n: oc("long benign").n },
      long_middle_caught: { k: oc("long middle").after, n: oc("long middle").n },
      tricky_attacks: { k: oc("tricky attacks").after, n: oc("tricky attacks").n },
      tricky_benign_fp: { k: oc("tricky benign").after, n: oc("tricky benign").n },
    },
    baselines,
    transformers,
    quantization,
    quantization_env: { cpu_threads: bench.env.threads, onnxruntime: bench.env.onnxruntime },
    hard_negatives: {
      thresholds_before: { block: r4(hard.thresholds.served.block), warn: r4(hard.thresholds.served.warn) },
      thresholds_after: { block: r4(hard.thresholds.candidate.block), warn: r4(hard.thresholds.candidate.warn) },
      test_f1_before: r4(hard.test_f1.served),
      test_f1_after: r4(hard.test_f1.candidate),
      test_f1_diff_ci: hard.test_f1.ci95.map(r4),
      all_criteria_pass: hard.all_pass,
      outcomes,
    },
    prompt_guard: {
      caveats: [
        "Different label definition: Prompt Guard flags only explicit attempts to override instructions; ours also counts jailbreak personas and instructions smuggled into documents.",
        "Its training data is undisclosed and may overlap our public test sources.",
        "It is designed as a first-line filter with a very low false-positive target and recommends splitting long inputs.",
        "Prompt Guard was scored in fp32; ours in int8 at the served threshold. Our row uses the served model's numbers on the same sets.",
      ],
      rows: [pgRow("pg22", "shipped_0.5"), pgRow("pg22", "val_f1_opt"), pgRow("pg86", "shipped_0.5"), pgRow("pg86", "val_f1_opt"), ours],
    },
    ner: {
      selected: ner.selected,
      n_sentences: ner.n_sentences,
      n_names: ner.n_names,
      rows: (ner.results as J[]).map((r) => ({
        model: r.id,
        q8_mb: r.q8MB,
        median_ms: r.medianMs,
        strict: r.eval.strict,
        gold: r.eval.gold,
        false_positives: r.eval.falsePositives,
        lowercase: r.eval.byCategory.lowercase,
        hinglish: r.eval.byCategory.hinglish,
      })),
    },
    browser_precheck: {
      max_abs_diff: parity.maxAbsDiffHeadTail,
      verdict_mismatches: parity.verdictMismatches,
      n: parity.n,
      load_ms: parity.loadMs,
      short_median_ms: parity.shortMedianMs,
      long: parity.long,
    },
  };
}

export const RESULTS_PATH = (repo: string) => join(repo, "frontend", "src", "data", "results.json");

export function writeResults(repo = resolve(__dirname, "..", "..")): string {
  writeFileSync(RESULTS_PATH(repo), JSON.stringify(buildResults(repo), null, 2) + "\n");
  return RESULTS_PATH(repo);
}
