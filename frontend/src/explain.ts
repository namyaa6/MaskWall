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
  partial_scan: "The message was too long to check completely, so part of it was not checked.",
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

export const fmtScore = (s: number) => s.toFixed(2);
export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "–";
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.round(ms)} ms`;
}
