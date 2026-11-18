import type { ReactNode } from "react";
import { PageHead } from "../components/PageHead";
import { f3, kn, ms, pct, R } from "../data/results";

type Row = (typeof R.baselines)[number];

function Section({ id, title, intro, children }: { id: string; title: string; intro: ReactNode; children: ReactNode }) {
  return (
    <section className="section card" aria-labelledby={id}>
      <h2 id={id}>{title}</h2>
      <p className="muted">{intro}</p>
      <div className="table-wrap">{children}</div>
    </section>
  );
}

function ClassifierTable({ rows, caption }: { rows: Row[]; caption: string }) {
  return (
    <table>
      <caption className="sr-only">{caption}</caption>
      <thead>
        <tr>
          <th scope="col">Model</th>
          <th scope="col">F1 [95% CI]</th>
          <th scope="col">Precision</th>
          <th scope="col">Attacks caught (recall)</th>
          <th scope="col">False alarms</th>
          <th scope="col">Tricky set correct</th>
          <th scope="col">Safe-but-scary false alarms</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.model}>
            <th scope="row">{r.model}</th>
            <td className="num">
              {f3(r.f1)} {r.f1_ci && <span className="muted">[{f3(r.f1_ci[0])}, {f3(r.f1_ci[1])}]</span>}
            </td>
            <td className="num">{f3(r.precision)}</td>
            <td className="num">{pct(r.recall)}</td>
            <td className="num">
              {pct(r.fpr)} <span className="muted">({r.fp} of {r.benign})</span>
            </td>
            <td className="num">
              {r.tricky_correct} of {r.tricky_total}
            </td>
            <td className="num">
              {r.xstest_fp} of {r.xstest_n}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function BenchmarksPage() {
  const H = R.hard_negatives;
  const L = R.lowercase_names;
  return (
    <>
      <PageHead title="Benchmarks">
        How accurate and how fast the attack detector is, measured on held-out test data, including its mistakes.
      </PageHead>
      <p className="card privacy-note">
        <strong>How to read these.</strong> “Attacks caught” is the share of real attacks that were flagged; “false alarms” is the share of harmless messages that were
        wrongly flagged. Thresholds were always chosen on separate validation data, never on the test set. All numbers are generated from the project's result files, not
        typed in.
      </p>

      <Section
        id="baselines"
        title="1. Simple baselines"
        intro={`Classic models on the same data and splits (public test set: ${R.baselines[0].benign! + R.baselines[0].attacks!} messages, ${R.baselines[0].attacks} attacks). The length-only model is a floor: it shows how much a model can get from message length alone.`}
      >
        <ClassifierTable rows={R.baselines} caption="Baselines on the test set" />
      </Section>

      <Section
        id="transformers"
        title="2. Fine-tuned transformers"
        intro="Small language models fine-tuned for the task on a Colab GPU (same splits). These are the models before quantisation and before the hard-example round."
      >
        <ClassifierTable rows={R.transformers} caption="Fine-tuned transformers on the test set" />
      </Section>

      <Section
        id="int8"
        title="3. Shrinking the models (int8 quantisation)"
        intro={`Size and CPU speed (${R.quantization_env.cpu_threads} threads, one message at a time, ONNX Runtime ${R.quantization_env.onnxruntime}) before and after converting the weights to 8-bit integers. F1 and false-alarm rate on the test set.`}
      >
        <table>
          <caption className="sr-only">fp32 versus int8</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Weights</th>
              <th scope="col">Size</th>
              <th scope="col">Typical message</th>
              <th scope="col">Full 512-token chunk</th>
              <th scope="col">F1</th>
              <th scope="col">False alarms</th>
            </tr>
          </thead>
          <tbody>
            {R.quantization.map((q) => (
              <tr key={q.model + q.precision}>
                <th scope="row">{q.model}</th>
                <td>{q.precision}</td>
                <td className="num">{q.size_mb} MB</td>
                <td className="num">{ms(q.p50_ms)}</td>
                <td className="num">{ms(q.p50_512_ms)}</td>
                <td className="num">{f3(q.f1)}</td>
                <td className="num">{pct(q.fpr)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section
        id="hardneg"
        title="4. Hard-example retraining (the model in use now)"
        intro={
          <>
            The int8 MiniLM was retrained with hard harmless examples (Hinglish, questions about prompts) and matching attacks. It replaced the previous model only because it
            passed four criteria written before the results existed. Test F1 {f3(H.test_f1_before)} → {f3(H.test_f1_after)} (difference 95% CI [{f3(H.test_f1_diff_ci[0])},{" "}
            {f3(H.test_f1_diff_ci[1])}], so “not worse”, not “better”). Thresholds: block {f3(H.thresholds_before.block)} → {f3(H.thresholds_after.block)}, warn{" "}
            {f3(H.thresholds_before.warn)} → {f3(H.thresholds_after.warn)}. p = exact McNemar test on the same messages; a change is labelled better or
            worse only when p &lt; 0.05.
          </>
        }
      >
        <table>
          <caption className="sr-only">Before and after the hard-example round</caption>
          <thead>
            <tr>
              <th scope="col">Evaluation set</th>
              <th scope="col">Measures</th>
              <th scope="col">Before</th>
              <th scope="col">After</th>
              <th scope="col">p</th>
            </tr>
          </thead>
          <tbody>
            {H.outcomes.map((o) => {
              // only call a change better/worse when the paired test says it is unlikely to be noise
              const sig = o.p < 0.05;
              const better = sig && (o.kind === "fp" ? o.after < o.before : o.after > o.before);
              const worse = sig && (o.kind === "fp" ? o.after > o.before : o.after < o.before);
              const noise = !sig && o.after !== o.before;
              return (
                <tr key={o.set}>
                  <th scope="row">{o.set.replace(/ \((FP|caught|recall)[^)]*\)/, "")}</th>
                  <td>{o.kind === "fp" ? "false alarms (lower is better)" : "attacks caught (higher is better)"}</td>
                  <td className="num">{kn({ k: o.before, n: o.n })}</td>
                  <td className="num">
                    {kn({ k: o.after, n: o.n })} {better && <span className="pill pill-safe">better</span>}
                    {worse && <span className="pill pill-suspicious">worse</span>}
                    {noise && <span className="muted small"> (within noise)</span>}
                  </td>
                  <td className="num">{o.p < 0.001 ? "< 0.001" : o.p.toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="muted small">
          E1 = real Hinglish questions (human-written); E2 = harmless questions about prompts and E3 = Hinglish attacks are AI-written and therefore optimistic. Long documents
          are synthetic. One training run (seed 42).
        </p>
      </Section>

      <section className="section card" aria-labelledby="pg">
        <h2 id="pg">5. Compared with Meta's Prompt Guard 2 (an outside reference)</h2>
        <p className="muted">Read the caveats first; this is a reference point, not a ranking:</p>
        <ul className="plain-list small">
          {R.prompt_guard.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
        <div className="table-wrap">
          <table>
            <caption className="sr-only">Prompt Guard 2 versus the served model</caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Threshold</th>
                <th scope="col">int8 size</th>
                <th scope="col">Test F1</th>
                <th scope="col">Attacks caught</th>
                <th scope="col">False alarms</th>
                <th scope="col">Hidden in documents caught</th>
                <th scope="col">Hinglish questions blocked (of 104)</th>
                <th scope="col">Prompt questions blocked (of 100)</th>
                <th scope="col">Hinglish attacks caught (of 24)</th>
              </tr>
            </thead>
            <tbody>
              {R.prompt_guard.rows.map((r) => (
                <tr key={r.model + r.point} className={r.model.includes("ours") ? "row-ours" : undefined}>
                  <th scope="row">{r.model}</th>
                  <td>
                    {r.point} <span className="muted">({f3(r.threshold)})</span>
                  </td>
                  <td className="num">{r.int8_mb} MB</td>
                  <td className="num">{f3(r.test_f1)}</td>
                  <td className="num">{pct(r.test_recall)}</td>
                  <td className="num">{pct(r.test_fpr)}</td>
                  <td className="num">{pct(r.indirect_recall)}</td>
                  <td className="num">{r.hinglish_benign_fp}</td>
                  <td className="num">{r.prompt_talk_fp}</td>
                  <td className="num">{r.hinglish_attacks_caught}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <Section
        id="browser"
        title="6. The models that run in your browser"
        intro={`Name detection: three candidates measured in Chrome (WebAssembly, one thread) on ${R.ner.n_sentences} AI-drafted sentences with ${R.ner.n_names} names, not reviewed by the project owner; chosen by a rule written before measuring. A name counts as hidden only if all of it is hidden.`}
      >
        <table>
          <caption className="sr-only">Browser name models</caption>
          <thead>
            <tr>
              <th scope="col">Name model</th>
              <th scope="col">Download</th>
              <th scope="col">Time per message</th>
              <th scope="col">Names hidden</th>
              <th scope="col">Lowercase names</th>
              <th scope="col">Hinglish names</th>
              <th scope="col">Words wrongly hidden</th>
            </tr>
          </thead>
          <tbody>
            {R.ner.rows.map((r) => (
              <tr key={r.model} className={r.model === R.ner.selected ? "row-ours" : undefined}>
                <th scope="row">
                  {r.model}
                  {r.model === R.ner.selected && <span className="muted"> (used)</span>}
                </th>
                <td className="num">{r.q8_mb} MB</td>
                <td className="num">{ms(r.median_ms)}</td>
                <td className="num">
                  {r.strict} of {r.gold}
                </td>
                <td className="num">
                  {r.lowercase.strict} of {r.lowercase.gold}
                </td>
                <td className="num">
                  {r.hinglish.strict} of {r.hinglish.gold}
                </td>
                <td className="num">{r.false_positives}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">
          Early attack warning: the served detector in the browser agrees with the server on all {R.browser_precheck.n} reference messages (largest score difference{" "}
          {R.browser_precheck.max_abs_diff.toFixed(4)}, {R.browser_precheck.verdict_mismatches} different verdicts). In Chrome it loads in about{" "}
          {(R.browser_precheck.load_ms / 1000).toFixed(1)} s and takes {ms(R.browser_precheck.short_median_ms)} for a short message and about{" "}
          {ms(R.browser_precheck.long[R.browser_precheck.long.length - 1].ms)} for a {R.browser_precheck.long[R.browser_precheck.long.length - 1].tokens}-token document.
        </p>
      </Section>
      <Section
        id="lowercase"
        title="7. Trying to fix lowercase names (a result that did not ship)"
        intro={
          <>
            Names typed in lowercase ("pls tell priya") are the browser layer's biggest gap. Three fixes were written and frozen, then measured on {L.n_positive_names} lowercase names and{" "}
            {L.n_negatives} harmless lowercase sentences (AI-drafted, not reviewed by the project owner). The rule for shipping one was fixed before measuring: no drop in recall on
            capitalised names, at most {L.rule.fpNegSentencesExtra} more harmless sentences wrongly flagged than today (and no more than {L.rule.fpNegSentencesMax} of {L.n_negatives}), at most{" "}
            {L.rule.fpExistingExtra} more wrong spans on the earlier sample, under {L.rule.maxMedianMs} ms, and a gain of at least {L.rule.minGainPoints} points.{" "}
            <strong>{L.selected ? `Selected: ${L.selected}.` : "No approach met the rule, so none was shipped and the browser keeps the plain method."}</strong>
          </>
        }
      >
        <table>
          <caption className="sr-only">Lowercase name approaches</caption>
          <thead>
            <tr>
              <th scope="col">Approach</th>
              <th scope="col">Lowercase names hidden</th>
              <th scope="col">With a cue ("tell X")</th>
              <th scope="col">Without a cue</th>
              <th scope="col">Name is also a word (joy, rose)</th>
              <th scope="col">Harmless sentences wrongly flagged</th>
              <th scope="col">Capitalised names hidden (earlier sample)</th>
              <th scope="col">Median time</th>
              <th scope="col">Passes the rule?</th>
            </tr>
          </thead>
          <tbody>
            {L.rows.map((r) => {
              const chk = L.checks.find((c) => c.approach === r.approach);
              return (
                <tr key={r.approach}>
                  <th scope="row">{r.label}</th>
                  <td className="num">
                    {r.recall} of {r.gold}
                  </td>
                  <td className="num">
                    {r.cue.strict} of {r.cue.gold}
                  </td>
                  <td className="num">
                    {r.no_cue.strict} of {r.no_cue.gold}
                  </td>
                  <td className="num">
                    {r.ambiguous.strict} of {r.ambiguous.gold}
                  </td>
                  <td className="num">
                    {r.fp_sentences} of {L.n_negatives}
                  </td>
                  <td className="num">
                    {r.existing_cap_recall} of {r.existing_cap_gold}
                  </td>
                  <td className="num">{ms(r.median_ms)}</td>
                  <td>{chk ? (chk.eligible ? "yes" : "no") : "(baseline)"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="muted small">
          Re-casing finds most lowercase names, but it also hides ordinary Hinglish words, festivals and brands that the word list did not cover ("barish", "khatam", "holi", "swiggy"),
          so it failed the false-alarm limit. Cue rules find most names written with a cue ("tell X", "X ne") but nothing extra without one, and they too flag some harmless words. Names that are also words
          (joy, rose, grace, deep) are missed by every approach. Every sentence was written by the same assistant that wrote the rules, which favours the cue approach; the
          results are indicative, not a benchmark.
        </p>
      </Section>
    </>
  );
}
