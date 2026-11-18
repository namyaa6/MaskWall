import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { Link } from "react-router-dom";
import { describeError, postChat, type ChatResponse } from "../api";
import { HandledPanel, type HandledTurn } from "../components/HandledPanel";
import { ModelStatusLine } from "../components/ModelStatus";
import { PageHead } from "../components/PageHead";
import { ResultPill } from "../components/Pills";
import { Reply } from "../components/Reply";
import { chatLabel, mainReason, nextStep, plural, toneFor } from "../explain";
import type { PiiSpan } from "../pii/detectors";
import { isMostlyLowercase } from "../pii/nameMode";
import { NER_MODEL } from "../pii/nerConfig";
import { normalizeText } from "../pii/normalize";
import { PlaceholderVault } from "../pii/vault";
import type { PrecheckResult } from "../precheck/types";
import type { NerResult } from "../workers/protocol";
import { createPrecheckWorker, createPrivacyWorker, useModelWorker } from "../workers/useModelWorker";

interface Turn extends HandledTurn {
  id: number;
  original: string;      // what the user typed: shown only here, on the device
  hiddenCount: number;
  reply?: string;        // the AI's reply with placeholders filled back in (on the device)
  error?: string;
}

const INTRO_KEY = "maskwall.chatIntroDismissed";
const readDismissed = () => {
  try {
    return localStorage.getItem(INTRO_KEY) === "1";
  } catch {
    return false;
  }
};

function Intro({ onDismiss }: { onDismiss: () => void }) {
  return (
    <section className="card intro" aria-labelledby="intro-title">
      <div className="intro-head">
        <h2 id="intro-title">How MaskWall protects your chat</h2>
        <button type="button" className="btn btn-small" onClick={onDismiss}>
          Got it, hide this
        </button>
      </div>
      <ol className="intro-list">
        <li>Names, emails, phone numbers, Aadhaar and PAN numbers are swapped for placeholders like <span className="ph">[EMAIL_1]</span> on your device, before anything is sent.</li>
        <li>The firewall checks every message for attacks on the AI (attempts to override its instructions) and blocks them.</li>
        <li>The AI's reply is checked for leaks, then your details are put back into it on your device. The server never sees them.</li>
      </ol>
    </section>
  );
}

export function ChatPage() {
  const vault = useRef(new PlaceholderVault());   // memory only; never stored or sent
  const [turns, setTurns] = useState<Turn[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [privacyOn, setPrivacyOn] = useState(true);
  const [skipNames, setSkipNames] = useState(false);
  // Strict name hiding (approach B2): OFF by default, opt-in, per visit. Added after the round-2 results were known (docs/decisions.md Day 3.14).
  const [strict, setStrict] = useState(false);
  const [busy, setBusy] = useState(false);
  const [introHidden, setIntroHidden] = useState(readDismissed);
  const ner = useModelWorker<NerResult>(createPrivacyWorker);
  const precheck = useModelWorker<PrecheckResult>(createPrecheckWorker);
  const nextId = useRef(1);
  const endRef = useRef<HTMLDivElement>(null);

  const update = useCallback((id: number, patch: Partial<Turn>) => setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch } : t))), []);

  useEffect(() => {
    // braces on purpose: an effect must return nothing or a cleanup function, and scrollIntoView's return value
    // (a Promise under some scroll polyfills/extensions) crashed React when it was returned implicitly
    endRef.current?.scrollIntoView?.({ block: "end" });
  }, [turns.length]);

  const warmUp = () => {
    if (privacyOn && ner.status.state === "idle") ner.load();
    if (precheck.status.state === "idle") precheck.load();
  };

  const dismissIntro = () => {
    setIntroHidden(true);
    try {
      localStorage.setItem(INTRO_KEY, "1");
    } catch {
      /* private mode: the intro just comes back next time */
    }
  };

  async function send(e?: FormEvent) {
    e?.preventDefault();
    const raw = draft;
    if (!raw.trim() || busy) return;
    const id = nextId.current++;
    const norm = normalizeText(raw);        // same normalisation as the server, before masking
    const base: Turn = { id, original: raw, sent: norm.text, privacyOn, strict, namesChecked: false, counts: {}, hiddenCount: 0, maskMs: 0, status: "masking" };
    setTurns((ts) => [...ts, base]);
    setSelected(id);
    setDraft("");
    setBusy(true);
    try {
      let sent = norm.text;
      if (privacyOn) {
        let names: PiiSpan[] = [];
        let nerMs = 0;
        let namesChecked = false;
        if (!(skipNames && ner.status.state === "error")) {
          try {
            const r = await ner.run(norm.text, { strict });      // waits for the first download if needed (shown as a progress bar)
            names = r.spans;
            nerMs = r.ms;
            namesChecked = true;
          } catch (err) {
            update(id, { status: "error", error: `Not sent: the privacy model for names did not load (${(err as Error).message}). Try again, or tick “Send without checking for names”.` });
            setDraft(raw);
            return;
          }
        }
        const t1 = performance.now();
        const out = vault.current.mask(norm.text, names);
        sent = out.masked;
        // time on the device = name model inference + regex masking (a first-time download is not counted)
        update(id, { sent, counts: out.counts, hiddenCount: out.findings.length, namesChecked, maskMs: nerMs + (performance.now() - t1), status: "sending" });
      } else {
        update(id, { status: "sending" });
      }
      // early warning on the device: runs alongside the request, never delays it
      precheck
        .run(sent)
        .then((r) => update(id, { precheck: r }))
        .catch(() => update(id, { precheck: "error" }));
      const res: ChatResponse = await postChat(sent);
      update(id, { res, status: "done", reply: res.response != null ? vault.current.unmask(res.response) : undefined });
    } catch (err) {
      update(id, { status: "error", error: describeError(err) });
    } finally {
      setBusy(false);
    }
  }

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) send();
  };

  const newConversation = () => {
    vault.current.clear();
    setTurns([]);
    setSelected(null);
  };

  const sel = turns.find((t) => t.id === selected) ?? turns[turns.length - 1] ?? null;

  return (
    <>
      <PageHead title="Chat">
        Talk to an AI assistant. Your personal details are hidden on your device before anything is sent, and every message is checked for attacks.
      </PageHead>
      {!introHidden && <Intro onDismiss={dismissIntro} />}

      <div className="chat-layout">
        <section className="chat-main card" aria-label="Conversation">
          <div className="conversation" aria-live="polite">
            {turns.length === 0 && <p className="muted empty">No messages yet. Try one with your email address in it and watch the panel on the right.</p>}
            {turns.map((t) => (
              <article
                key={t.id}
                className={`turn${sel?.id === t.id ? " turn-selected" : ""}`}
                aria-label={`Message ${t.id}`}
              >
                <div className="bubble bubble-user">
                  <p className="bubble-text">{t.original}</p>
                  <p className="bubble-meta">
                    {t.privacyOn
                      ? `${plural(t.hiddenCount, "personal detail")} hidden before sending`
                      : "Hiding personal data was off: sent as typed"}
                    {t.privacyOn && t.status !== "masking" && !t.namesChecked && " (names not checked)"}
                  </p>
                </div>
                <TurnResult t={t} />
                <button type="button" className="link-btn" onClick={() => setSelected(t.id)} aria-pressed={sel?.id === t.id}>
                  Show how this message was handled
                </button>
              </article>
            ))}
            <div ref={endRef} />
          </div>

          <form className="composer" onSubmit={send}>
            <div className="composer-options">
              <label className="switch">
                <input
                  type="checkbox"
                  role="switch"
                  checked={privacyOn}
                  onChange={(e) => {
                    setPrivacyOn(e.target.checked);
                    if (e.target.checked && ner.status.state === "idle") ner.load();
                  }}
                />
                <span>Hide my personal data</span>
              </label>
              {privacyOn && (
                <div className="strict-option">
                  <label className="switch">
                    <input type="checkbox" role="switch" checked={strict} onChange={(e) => setStrict(e.target.checked)} aria-describedby="strict-help" />
                    <span>Strict name hiding</span>
                  </label>
                  <p id="strict-help" className="muted small">
                    Finds most lowercase names but may also hide some ordinary words. <Link to="/benchmarks#lowercase2">See the measured numbers</Link>
                  </p>
                </div>
              )}
              {ner.status.state === "error" && privacyOn && (
                <label className="switch">
                  <input type="checkbox" checked={skipNames} onChange={(e) => setSkipNames(e.target.checked)} />
                  <span>Send without checking for names (emails, phone, Aadhaar and PAN are still hidden)</span>
                </label>
              )}
              {turns.length > 0 && (
                <button type="button" className="btn btn-small" onClick={newConversation}>
                  New conversation
                </button>
              )}
            </div>
            {privacyOn && (
              <ModelStatusLine status={ner.status} name="privacy model" approxMB={NER_MODEL.approxMB} readyText="Privacy model ready: names are found on your device." />
            )}
            <ModelStatusLine
              status={precheck.status}
              name="attack check"
              approxMB={23}
              readyText="Early attack warning ready on your device. The server makes the final decision."
            />
            <label htmlFor="chat-input" className="sr-only">
              Your message
            </label>
            <textarea
              id="chat-input"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onFocus={warmUp}
              onKeyDown={onKey}
              rows={3}
              maxLength={8000}
              placeholder="Type a message. Enter sends, Shift+Enter adds a new line."
            />
            {privacyOn && !strict && isMostlyLowercase(draft) && (
              <p className="notice" role="note">
                This message is mostly lowercase, so names in it may not be hidden.{" "}
                <button type="button" className="link-btn" onClick={() => setStrict(true)}>
                  Turn on Strict name hiding
                </button>{" "}
                to catch more of them.
              </p>
            )}
            <div className="composer-foot">
              <p className="muted small">Each message is answered on its own; the AI doesn't see earlier messages.</p>
              <button type="submit" className="btn btn-primary" disabled={busy || !draft.trim()}>
                {busy ? "Sending…" : "Send"}
              </button>
            </div>
          </form>
        </section>

        <HandledPanel turn={sel} />
      </div>
    </>
  );
}

function TurnResult({ t }: { t: Turn }) {
  if (t.status === "masking") return <p className="bubble bubble-ai muted">Hiding personal data on your device…</p>;
  if (t.status === "sending") return <p className="bubble bubble-ai muted">Checking for attacks and asking the AI…</p>;
  if (t.status === "error")
    return (
      <div className="bubble bubble-ai" role="alert">
        <ResultPill tone="blocked">Not answered</ResultPill>
        <p>{t.error}</p>
      </div>
    );
  const res = t.res!;
  const step = nextStep(res.action, res.input.reasons);
  return (
    <div className="bubble bubble-ai">
      <p className="result-line">
        <ResultPill tone={toneFor(res.action)}>{chatLabel(res.action)}</ResultPill>
        {res.action !== "allow" && <span>{mainReason(res.action, res.input.reasons, res.output?.kinds)}</span>}
      </p>
      {step && <p className="next-step">{step}</p>}
      {res.action !== "block" && res.action !== "output_blocked" && t.reply != null && <Reply text={t.reply} />}
      {res.truncated && (
        <p className="cutoff" role="note">
          Reply was cut off: it reached the length limit. Ask for a shorter answer or for the rest.
        </p>
      )}
    </div>
  );
}
