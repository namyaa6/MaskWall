import { Fragment } from "react";
import { PII_LABELS, toneFor, type Tone } from "../explain";

export function ResultPill({ tone, children }: { tone: Tone; children: string }) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

export function ActionPill({ action, label }: { action: string; label: string }) {
  return <ResultPill tone={toneFor(action)}>{label}</ResultPill>;
}

const PLACEHOLDER = /\[(S_)?(NAME|EMAIL|PHONE|AADHAAR|PAN)_(\d+)\]/g;

/**
 * The placeholder types that actually occur in `text`, one example placeholder per type in order of first appearance, split into
 * "hidden on your device" ([EMAIL_1]) and "hidden by the server's backup check" ([S_EMAIL_1]). Used for the legend under
 * "What the AI actually received", so it never names a type (for example an email) that is not in the message.
 */
export function placeholderLegend(text: string): { device: string[]; server: string[] } {
  const seen = new Set<string>();
  const out = { device: [] as string[], server: [] as string[] };
  for (const m of text.matchAll(PLACEHOLDER)) {
    const server = Boolean(m[1]);
    const key = `${server}:${m[2]}`;
    if (seen.has(key)) continue;
    seen.add(key);
    (server ? out.server : out.device).push(m[0]);
  }
  return out;
}

/**
 * Renders text with placeholders as light monospace pills. [EMAIL_1] = hidden on your device; [S_EMAIL_1] = hidden by the
 * server's backup scanner (the browser missed it). Plain text otherwise (no HTML is ever interpreted).
 */
export function MaskedText({ text }: { text: string }) {
  const parts: (string | { ph: string; server: boolean; type: string })[] = [];
  let pos = 0;
  for (const m of text.matchAll(PLACEHOLDER)) {
    if (m.index! > pos) parts.push(text.slice(pos, m.index));
    parts.push({ ph: m[0], server: Boolean(m[1]), type: m[2] });
    pos = m.index! + m[0].length;
  }
  if (pos < text.length) parts.push(text.slice(pos));
  return (
    <span className="masked-text">
      {parts.map((p, i) =>
        typeof p === "string" ? (
          <Fragment key={i}>{p}</Fragment>
        ) : (
          <span
            key={i}
            className={`ph${p.server ? " ph-server" : ""}`}
            title={`${PII_LABELS[p.type] ?? p.type}, hidden ${p.server ? "by the server's backup check" : "on your device"}`}
          >
            {p.ph}
          </span>
        ),
      )}
    </span>
  );
}
