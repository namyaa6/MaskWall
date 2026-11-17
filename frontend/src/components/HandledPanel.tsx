import type { ChatResponse } from "../api";
import { fmtMs, fmtScore, mainReason, PII_LABELS, plural, type Tone } from "../explain";
import type { PrecheckResult } from "../precheck/types";
import { MaskedText, ResultPill } from "./Pills";

export interface HandledTurn {
  privacyOn: boolean;
  namesChecked: boolean;
  counts: Partial<Record<string, number>>;
  sent: string;
  maskMs: number;
  precheck?: PrecheckResult | "error";
  res?: ChatResponse;
  status: "masking" | "sending" | "done" | "error";
}

function countsText(counts: Partial<Record<string, number>>): string {
  const parts = Object.entries(counts)
    .filter(([, n]) => n)
    .map(([t, n]) => plural(n!, PII_LABELS[t] ?? t.toLowerCase(), t === "PAN" ? "PANs" : undefined));
  return parts.join(", ");
}

const verdictWord = (v: string) => (v === "block" ? "attack" : v === "warn" ? "suspicious" : "safe");
const verdictTone = (v: string): Tone => (v === "block" ? "blocked" : v === "warn" ? "suspicious" : "safe");

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <li className="step">
      <span className="step-num" aria-hidden="true">{n}</span>
      <div>
        <h3>
          <span className="sr-only">Step {n}: </span>
          {title}
        </h3>
        <div className="step-body">{children}</div>
      </div>
    </li>
  );
}

/** Side panel: how one message was handled, in four numbered steps, plus what the AI actually received. */
export function HandledPanel({ turn }: { turn: HandledTurn | null }) {
  if (!turn)
    return (
      <aside className="card handled" aria-label="How this message was handled">
        <h2>How this message was handled</h2>
        <p className="muted">Send a message and this panel shows, step by step, what MaskWall did with it.</p>
      </aside>
    );
  const res = turn.res;
  const inp = res?.input;
  const serverPii = inp ? Object.values(inp.pii_masked).reduce((a, b) => a + b, 0) : 0;
  const local = countsText(turn.counts);
  const llmMs = res?.latency_ms.llm_ms ?? 0;
  const serverAdded = res ? Math.max(0, res.latency_ms.total_ms - llmMs) : null;
  const pc = turn.precheck;

  return (
    <aside className="card handled" aria-label="How this message was handled" aria-live="polite">
      <h2>How this message was handled</h2>
      <ol className="steps">
        <Step n={1} title="Personal data hidden">
          {turn.privacyOn ? (
            <p>
              {local ? `On your device: ${local}.` : "On your device: nothing personal found."}
              {!turn.namesChecked && " Names were not checked (the privacy model was not available)."}
            </p>
          ) : (
            <p>Off. Your message was sent as you typed it.</p>
          )}
          {inp && serverPii > 0 && (
            <p className="note-suspicious">
              The server's backup check hid {plural(serverPii, "more detail")} ({countsText(inp.pii_masked)}), shown as S_ pills below. Your
              device missed {serverPii === 1 ? "it" : "them"}, so {serverPii === 1 ? "it" : "they"} reached the server but not the AI.
            </p>
          )}
        </Step>

        <Step n={2} title="Checked for attacks">
          {inp ? (
            <p>
              Risk score <strong>{fmtScore(inp.score)} out of 1</strong>:{" "}
              <ResultPill tone={verdictTone(inp.verdict)}>{verdictWord(inp.verdict)}</ResultPill>
              <br />
              <span className="muted">{mainReason(inp.verdict, inp.reasons)}</span>
            </p>
          ) : (
            <p className="muted">{turn.status === "error" ? "Not checked: the message did not reach the firewall." : "Checking…"}</p>
          )}
          <p className="muted small">
            Early warning on your device:{" "}
            {pc === undefined ? "running…" : pc === "error" ? "not available" : `${fmtScore(pc.score)} (${verdictWord(pc.verdict)}, ${fmtMs(pc.ms)})`}. The
            server makes the final decision.
          </p>
        </Step>

        <Step n={3} title="Answered by the AI">
          {!res ? (
            <p className="muted">{turn.status === "error" ? "No answer." : "Waiting…"}</p>
          ) : res.llm ? (
            <p>
              {res.llm.provider === "groq" ? "Groq" : res.llm.provider === "gemini" ? "Gemini" : res.llm.provider} (
              <span className="mono">{res.llm.model}</span>) in {fmtMs(res.latency_ms.llm_ms)}.
              {res.llm.failed_over && " The main AI did not answer, so the backup AI did."}
              {res.truncated && " The reply was cut off at the length limit."}
            </p>
          ) : (
            <p>Not sent to the AI: the firewall blocked it first.</p>
          )}
        </Step>

        <Step n={4} title="Reply checked">
          {!res ? (
            <p className="muted">Waiting…</p>
          ) : !res.output ? (
            <p className="muted">Nothing to check: there was no reply.</p>
          ) : res.output.flagged ? (
            <p>
              <ResultPill tone="blocked">withheld</ResultPill> {mainReason("output_blocked", [], res.output.kinds)}
            </p>
          ) : (
            <p>No private instructions, keys or secrets found in the reply. Placeholders were filled back in on your device.</p>
          )}
        </Step>
      </ol>

      <div className="time-added">
        <h3>Time MaskWall added</h3>
        <p>
          {serverAdded === null ? "–" : <strong>{fmtMs(serverAdded + turn.maskMs)}</strong>}
          {serverAdded !== null && (
            <span className="muted">
              {" "}
              ({fmtMs(serverAdded)} on the server + {fmtMs(turn.maskMs)} on your device; the AI's own time is not counted)
            </span>
          )}
        </p>
      </div>

      <div className="received">
        <h3>{res?.action === "block" ? "What the firewall received (not sent to the AI)" : "What the AI actually received"}</h3>
        <p className="received-text">
          <MaskedText text={inp ? inp.masked_input : turn.sent} />
        </p>
        <p className="muted small">
          <span className="ph">[EMAIL_1]</span> hidden on your device · <span className="ph ph-server">[S_EMAIL_1]</span> hidden by the server's backup check
        </p>
      </div>
    </aside>
  );
}
