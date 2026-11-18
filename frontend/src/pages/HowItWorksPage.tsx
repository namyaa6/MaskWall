import { Link } from "react-router-dom";
import { PageHead } from "../components/PageHead";
import { kn, ms, pct, R } from "../data/results";

const S = R.served;
const NER = R.ner.rows.find((r) => r.model === R.ner.selected)!;
const P = R.browser_precheck;
const LC2_BASE = R.lowercase_names_r2.rows.find((r) => r.approach === "A")!;
const LC2_B = R.lowercase_names_r2.rows.find((r) => r.approach === "B")!;
const Q_INT8 = R.quantization.find((q) => q.model === "MiniLM-L6" && q.precision === "int8")!;

const JOURNEY = [
  { title: "You type", where: "Your device", text: "Your message stays in the browser tab until the next step is done." },
  { title: "Hide personal data", where: "Your device", text: "Names, emails, phone, Aadhaar and PAN numbers become placeholders like [EMAIL_1]. The originals stay in this tab's memory." },
  { title: "Check for attacks", where: "Server (and an early warning on your device)", text: "A trained detector scores the message from 0 to 1. Attacks are blocked before the AI sees them." },
  { title: "Get the answer", where: "Groq, with Gemini as backup", text: "Only the masked message is sent. If the main AI fails, the backup answers." },
  { title: "Check the reply", where: "Server", text: "The reply is scanned for leaked instructions, keys and secrets before it is returned." },
  { title: "You see the answer", where: "Your device", text: "Placeholders are swapped back for your details, in your browser only." },
];

const PII_ROWS = [
  { type: "Name", how: `A small AI model running in your browser (${NER.model.split("/")[1]}, about ${Math.round(NER.q8_mb)} MB, downloaded once)`, example: "Priya Sharma", ph: "[NAME_1]" },
  { type: "Email address", how: "Pattern: something@domain.tld", example: "asha@example.com", ph: "[EMAIL_1]" },
  { type: "Indian mobile number", how: "Pattern: 10 digits starting with 6 to 9, with or without +91, 0, spaces or hyphens", example: "+91 98765 43210", ph: "[PHONE_1]" },
  { type: "Aadhaar number", how: "12 digits not starting with 0 or 1, and the Verhoeff checksum must be valid (so random 12-digit numbers are not hidden)", example: "2345 6789 0124 (made up)", ph: "[AADHAAR_1]" },
  { type: "PAN", how: "5 letters, 4 digits, 1 letter, and the 4th letter must be a valid holder type (P, C, H, F, A, T, B, L, J or G)", example: "ABCPE1234F", ph: "[PAN_1]" },
];

export function HowItWorksPage() {
  return (
    <>
      <PageHead title="How it works">
        The journey of one message, from your keyboard to the AI and back, and what is checked at each step.
      </PageHead>

      <section aria-labelledby="journey-title" className="section">
        <h2 id="journey-title">The journey of a message</h2>
        <ol className="journey">
          {JOURNEY.map((s, i) => (
            <li key={s.title} className="card journey-step">
              <span className="step-num" aria-hidden="true">{i + 1}</span>
              <h3>{s.title}</h3>
              <p className="muted small where">{s.where}</p>
              <p>{s.text}</p>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="pii-title" className="section card">
        <h2 id="pii-title">Personal data that is hidden</h2>
        <p className="muted">
          All of this happens on your device before sending. The server repeats the four pattern checks as a backup; anything it catches is shown as{" "}
          <span className="ph ph-server">[S_EMAIL_1]</span>, which means your device missed it.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Type</th>
                <th scope="col">How it is found</th>
                <th scope="col">Example</th>
                <th scope="col">Becomes</th>
              </tr>
            </thead>
            <tbody>
              {PII_ROWS.map((r) => (
                <tr key={r.type}>
                  <th scope="row">{r.type}</th>
                  <td>{r.how}</td>
                  <td>{r.example}</td>
                  <td>
                    <span className="ph">{r.ph}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section aria-labelledby="attacks-title" className="section card">
        <h2 id="attacks-title">Attacks it catches</h2>
        <ul className="plain-list">
          <li>
            <strong>Direct overrides</strong>: “Ignore all previous instructions and …”, requests for the AI's hidden instructions.
          </li>
          <li>
            <strong>Jailbreaks and role-play</strong>: “Pretend you are an AI with no rules…”, invented personas meant to switch off safety rules.
          </li>
          <li>
            <strong>Attacks hidden in documents</strong>: instructions buried inside a pasted email, review or web page. Caught in {kn(S.indirect_recall)} held-out examples.
          </li>
          <li>
            <strong>Hinglish</strong>: the same attacks written in Hindi words with Latin letters. Caught in {kn(S.hinglish_attacks)} test examples (AI-written, so optimistic).
          </li>
        </ul>

        <h3 className="subhead">How it handles tricky inputs</h3>
        <dl className="notes">
          <dt>Long messages: overlapping chunks</dt>
          <dd>
            The detector reads at most 510 word pieces at a time. Longer messages are read in overlapping chunks (each new chunk starts halfway through the previous
            one), so any instruction up to 255 word pieces long fits entirely inside one chunk. Up to 6 chunks are read; if a message needs more, the first five and the
            last one are read and the message is marked “suspicious” because part of it was not checked.
          </dd>
          <dt>Disguised text: cleaned first</dt>
          <dd>
            Look-alike letters (such as fullwidth “Ｉｇｎｏｒｅ”) are turned back into plain letters, digits in other scripts become 0 to 9, and invisible characters are
            removed before anything is checked. Hidden “tag” characters, which some AIs can read but people cannot see, are removed and make the message suspicious.
          </dd>
          <dt>Fewer false alarms: training on hard examples</dt>
          <dd>
            Earlier, harmless Hinglish questions and questions about writing prompts were often blocked. The detector was retrained with hard harmless examples and matching
            attacks. False alarms on real Hinglish questions fell from {R.hard_negatives.outcomes.find((o) => o.set.startsWith("E1 human"))!.before} to{" "}
            {S.hinglish_benign_fp.k} in {S.hinglish_benign_fp.n}, and on prompt-writing questions from {R.hard_negatives.outcomes.find((o) => o.set.startsWith("E2"))!.before}{" "}
            to {S.prompt_talk_fp.k} in {S.prompt_talk_fp.n}, without catching fewer attacks.
          </dd>
        </dl>
      </section>

      <section aria-labelledby="answer-title" className="section card">
        <h2 id="answer-title">Getting the answer, and checking it</h2>
        <dl className="notes">
          <dt>Main AI and backup</dt>
          <dd>Messages go to Groq. If Groq times out, fails or is rate-limited, Gemini answers instead, and the chat says that the backup was used.</dd>
          <dt>Cut-off replies</dt>
          <dd>If a reply reaches the length limit, it is marked “Reply was cut off” instead of silently ending mid-sentence.</dd>
          <dt>Reply checks</dt>
          <dd>
            Before a reply is returned it is checked for a secret marker planted in the AI's instructions, for long passages of those instructions, and for things that
            look like API keys, tokens or passwords. If any is found, the reply is withheld.
          </dd>
          <dt>Early warning on your device</dt>
          <dd>
            Your browser runs the same detector ({S.int8_mb} MB, same thresholds) for an instant warning: it agrees with the server on all {P.n} reference
            messages (scores within {P.max_abs_diff.toFixed(4)}). It can be switched off by anyone who controls the browser, so the server always makes the final decision.
          </dd>
        </dl>
      </section>

      <section aria-labelledby="numbers-title" className="section">
        <h2 id="numbers-title">How well it works</h2>
        <p className="muted">
          Measured on data the detector never saw during training. Full tables, including the mistakes, are on the <Link to="/benchmarks">Benchmarks</Link> page.
        </p>
        <div className="tiles">
          <div className="card tile">
            <p className="tile-label">Attacks caught</p>
            <p className="tile-value">{pct(S.test_recall.k / S.test_recall.n, 0)}</p>
            <p className="muted small">{kn(S.test_recall)} attacks in the public test set</p>
          </div>
          <div className="card tile">
            <p className="tile-label">False alarms</p>
            <p className="tile-value">{pct(S.test_fp.k / S.test_fp.n)}</p>
            <p className="muted small">{kn(S.test_fp)} harmless test messages blocked; {kn(S.xstest_fp)} on a set of safe-but-scary-sounding questions</p>
          </div>
          <div className="card tile">
            <p className="tile-label">Hidden in documents</p>
            <p className="tile-value">{pct(S.indirect_recall.k / S.indirect_recall.n, 0)}</p>
            <p className="muted small">{kn(S.indirect_recall)} caught</p>
          </div>
          <div className="card tile">
            <p className="tile-label">Names hidden in the browser</p>
            <p className="tile-value">{NER.strict} of {NER.gold}</p>
            <p className="muted small">On a small AI-drafted sample; lowercase names {NER.lowercase.strict} of {NER.lowercase.gold}</p>
          </div>
        </div>
        <p className="muted small">
          Overall test F1 {S.test_f1.toFixed(3)}. Speed: the detector takes about {ms(Q_INT8.p50_ms)} for a typical message on a 2-thread server CPU (
          {ms(Q_INT8.p50_512_ms)} for a full 510-word-piece chunk; measured on the previous version of the model, which has the same architecture and size); the early warning in your browser takes about {ms(P.short_median_ms)} for a short message.
        </p>
      </section>

      <section aria-labelledby="limits-title" className="section card limits">
        <h2 id="limits-title">What it can't do yet</h2>
        <ul className="plain-list">
          <li>
            <strong>Lowercase names are mostly missed</strong> (“pls tell priya…”): only {LC2_BASE.recall} of {LC2_BASE.gold} in a fresh lowercase test. The name model learned from news text,
            where names are capitalised. Two rounds of fixes were measured; the best finds {LC2_B.recall} of {LC2_B.gold} but also hides ordinary words in {LC2_B.fp_sentences} of{" "}
            {R.lowercase_names_r2.n_negatives} harmless sentences, which breaks the limit set beforehand, so none was shipped (see Benchmarks). Addresses, dates of birth and bank details are
            not detected at all.
          </li>
          <li>
            <strong>It is a detector with error rates, not a guarantee.</strong> About {pct(1 - S.test_recall.k / S.test_recall.n, 0)} of test attacks get through, and new
            attack styles will do better than that.
          </li>
          <li>
            <strong>Long harmless documents are flagged more often</strong>: {kn(S.long_benign_fp)} long harmless documents were blocked in testing (synthetic documents), a
            cost of the retraining that fixed the Hinglish false alarms.
          </li>
          <li>
            <strong>Encoded attacks</strong> (base64, letters spaced out, other scripts) are not decoded before checking.
          </li>
          <li>
            <strong>A paraphrased or translated leak</strong> of the AI's instructions, or a secret in an unusual format, can pass the reply check.
          </li>
          <li>
            <strong>No conversation memory</strong>: each message is answered on its own. Checking a whole conversation history is a different problem.
          </li>
          <li>
            <strong>Narrow evidence for Hinglish</strong>: no Devanagari script in training, and the Hinglish attack tests are AI-written. The retrained model is one training run.
          </li>
          <li>
            <strong>Hugging Face sees the name-model download</strong>: the browser fetches the name model from the Hugging Face Hub on first use. No message text is
            involved, but Hugging Face sees your IP address. Everything else (the attack model and the AI runtime) is served by this site.
          </li>
        </ul>
      </section>
    </>
  );
}
