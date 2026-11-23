// Plain-English wording for verdicts, reasons and placeholders, shared by every page so the same thing is always
// described the same way. Colours are always paired with these words, never used alone.
import type { ChatAction } from "./api";

export type Tone = "safe" | "suspicious" | "blocked";

export function toneFor(action: string): Tone {
  if (action === "block" || action === "output_blocked" || action === "rejected" || action === "error") return "blocked";
  if (action === "warn") return "suspicious";
  return "safe";
}

/** Chat result label (under the AI's reply). */
export function chatLabel(action: ChatAction): string {
  switch (action) {
    case "allow": return "Safe · answered";
    case "warn": return "Suspicious · answered with a warning";
    case "block": return "Message blocked";
    case "output_blocked": return "Reply withheld";
  }
}

/** Short label for a logged action (Dashboard). */
export function actionLabel(action: string): string {
  switch (action) {
    case "allow": return "Safe";
    case "warn": return "Suspicious";
    case "block": return "Blocked";
    case "output_blocked": return "Reply withheld";
    case "rejected": return "Rejected";
    case "error": return "AI unavailable";
    default: return action;
  }
}

const REASONS: Record<string, string> = {
  injection_detected: "It reads like an attempt to override the AI's instructions or bypass its rules.",
  injection_suspected: "Parts of it resemble instructions aimed at the AI, but the detector is not sure.",
  partial_scan: "The message was too long or too dense (for example long runs of numbers or code) to check completely, so it was not sent.",
  hidden_characters: "It contained invisible characters that can be used to hide instructions.",
  strict_mode: "The firewall is in strict mode, which blocks anything suspicious.",
  empty_after_normalization: "It was empty once invisible characters were removed.",
  input_too_long: "It was over the 8,000-character limit.",
  llm_unavailable: "The AI services did not answer.",
  "output_leak:canary": "The reply contained a secret marker from the AI's instructions.",
  "output_leak:system_prompt": "The reply repeated the AI's private instructions.",
  "output_leak:known_secret": "The reply contained a protected secret.",
  "output_leak:secret": "The reply contained something that looks like an API key or password.",
  output_leak: "The reply appeared to leak protected information.",
};

/** Shown instead of a score when the server refuses a message it cannot scan completely (Day 4.19): no score, no meter. */
export const UNSCORED_HEADLINE = "Too long to check completely, so it is blocked without scanning.";
export const UNSCORED_NEXT_STEP = "Shorten it (or paste less text) and try again.";

export function reasonText(reason: string): string {
  return REASONS[reason] ?? reason.replace(/[_:]/g, " ");
}

/** One-line reason for a result, in the order a person cares about. */
export function mainReason(action: string, reasons: string[], outputKinds: string[] = []): string {
  if (action === "output_blocked") {
    const k = outputKinds[0] ?? reasons.find((r) => r.startsWith("output_leak"))?.split(":")[1];
    return reasonText(k ? `output_leak:${k}` : "output_leak");
  }
  const first = reasons.find((r) => !r.startsWith("output_leak")) ?? reasons[0];
  if (first) return reasonText(first);
  return action === "allow" ? "Nothing in it looks like an attack." : "";
}

/** What the user can do next. */
export function nextStep(action: ChatAction, reasons: string[]): string | null {
  if (action === "block") {
    if (reasons.includes("partial_scan") && !reasons.includes("injection_detected")) return "Send a shorter message, or paste less text, so all of it can be checked.";
    return reasons.includes("hidden_characters")
      ? "Retype the message instead of pasting it, so invisible characters are left out."
      : "If this was a genuine question, rephrase it without telling the AI to ignore or change its instructions. If you pasted a document, remove any instructions inside it.";
  }
  if (action === "warn") {
    return reasons.includes("partial_scan")
      ? "Send a shorter message if you want all of it checked."
      : "The AI answered, but treat the reply with care. If you pasted text from elsewhere, check it for hidden instructions.";
  }
  if (action === "output_blocked") return "Ask the question differently. The AI's reply was held back to protect private information.";
  return null;
}

export const PII_LABELS: Record<string, string> = {
  NAME: "name",
  EMAIL: "email address",
  PHONE: "phone number",
  AADHAAR: "Aadhaar number",
  PAN: "PAN",
};

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

/**
 * "2 emails, 1 Aadhaar number (only looks like one: it failed the strict check, hidden anyway)".
 * Aadhaar and PAN are hidden by SHAPE (Day 4.21); `unverified` says how many of them failed the strict check
 * (Aadhaar: first digit 2-9 and checksum; PAN: holder letter), so the panel can say "verified" vs "looks like".
 */
export function describeCounts(counts: Partial<Record<string, number>>, unverified: Partial<Record<string, number>> = {}): string {
  return Object.entries(counts)
    .filter(([, n]) => n)
    .map(([t, n]) => {
      const base = plural(n!, PII_LABELS[t] ?? t.toLowerCase(), t === "PAN" ? "PANs" : undefined);
      const u = Math.min(unverified[t] ?? 0, n!);
      if (!u) return base;
      return u === n ? `${base} (looks like ${n === 1 ? "one" : "them"}, not verified)` : `${base} (${n! - u} verified, ${u} ${u === 1 ? "only looks like one" : "only look like them"})`;
    })
    .join(", ");
}

/** One line under the counts when anything was hidden by shape only. */
export const LOOKS_LIKE_NOTE =
  "“Looks like” means the value has the shape of an Aadhaar number (12 digits) or a PAN (5 letters, 4 digits, 1 letter) but failed the strict check (checksum, first digit or holder letter). It was hidden anyway: hiding too much is better than leaking.";

export const fmtScore = (s: number) => s.toFixed(2);
export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "–";
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.round(ms)} ms`;
}
