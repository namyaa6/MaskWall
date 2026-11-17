// Thin client for the MaskWall firewall API (backend/README.md). Base URL from VITE_API_URL, default: local uvicorn.
export const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export interface Health {
  status: string;
  version: string;
  server_model: string;
  scan_mode: string;
  model_loaded: boolean;
  max_chars: number;
  llm_providers: string[];
  thresholds?: { warn: number; block: number };
}

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(`HTTP ${status}`);
  }
}

export async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(`${API_URL}${path}`, { signal });
  if (!r.ok) throw new ApiError(r.status, await r.json().catch(() => null));
  return (await r.json()) as T;
}

export async function postJSON<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const r = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) throw new ApiError(r.status, await r.json().catch(() => null));
  return (await r.json()) as T;
}

export const getHealth = (signal?: AbortSignal) => getJSON<Health>("/health", signal);

// ---------------------------------------------------------------- /scan and /chat (backend/app/main.py input_section)
export type InputVerdict = "allow" | "warn" | "block";
export type ChatAction = InputVerdict | "output_blocked";

export interface InputSection {
  score: number;
  verdict: InputVerdict;
  reasons: string[];              // injection_detected | injection_suspected | partial_scan | hidden_characters | strict_mode
  notes: string[];                // pii_masked_server_side | unicode_normalized
  hidden_characters_removed: number;
  coverage: "full" | "partial";
  mode: string;
  n_tokens: number;
  n_windows: number;
  n_windows_needed: number;
  scanned_tokens: number;
  thresholds: { warn: number; block: number };
  pii_masked: Record<string, number>;   // what the SERVER's backup masking caught (placeholders [S_...])
  masked_input: string;                 // exactly what the scanner and the AI received
  windows: { start: number; end: number; score: number; ms: number }[];
}

export interface Latency {
  pii_ms: number;
  tokenize_ms: number;
  inference_ms: number;
  llm_ms?: number;
  output_ms?: number;
  total_ms: number;
}

export interface ScanResponse {
  request_id: string;
  action: InputVerdict;
  input: InputSection;
  latency_ms: Latency;
}

export interface ChatResponse {
  request_id: string;
  action: ChatAction;
  response: string | null;
  input: InputSection;
  latency_ms: Latency;
  truncated: boolean;
  output: { flagged: boolean; kinds: string[]; findings: { kind: string; name: string; count: number }[] } | null;
  llm: {
    provider: string;
    model: string;
    failed_over: boolean;
    finish_reason: string | null;
    max_output_tokens: number;
    attempts: { provider: string; ok: boolean; kind: string; status: number | null; ms: number }[];
  } | null;
}

export const postScan = (text: string, signal?: AbortSignal) => postJSON<ScanResponse>("/scan", { text }, signal);
export const postChat = (message: string, signal?: AbortSignal) => postJSON<ChatResponse>("/chat", { message }, signal);

// ---------------------------------------------------------------- /stats (backend/app/logging_db.py stats)
export interface LayerLatency {
  n: number;
  mean: number | null;
  p50: number | null;
  p95: number | null;
}

export interface RecentEvent {
  ts: number;
  endpoint: string;
  action: string;                 // allow | warn | block | output_blocked | rejected | error
  score: number | null;
  reasons: string[];
  pii_server_side: number;
  total_ms: number | null;
  llm_ms: number | null;
}

export interface Stats {
  total_requests: number;
  attacks_blocked: number;
  outputs_blocked: number;
  warnings: number;
  by_action: Record<string, number>;
  by_endpoint: Record<string, number>;
  pii_caught_server_side: Record<string, number>;
  pii_total: number;
  truncated_replies: number;
  llm: { providers: Record<string, number>; failovers: number };
  latency_ms: Record<string, LayerLatency>;
  recent: RecentEvent[];
  first_event_ts: number | null;
}

export const getStats = (signal?: AbortSignal) => getJSON<Stats>("/stats", signal);

/** Plain-English message for an HTTP error from the firewall. */
export function describeError(e: unknown): string {
  if (e instanceof ApiError) {
    switch (e.status) {
      case 413: return "That message is too long for the firewall (over 8,000 characters). Shorten it and try again.";
      case 422: return "That message is empty once hidden characters are removed, so there was nothing to send.";
      case 429: return "Too many messages in a short time. Wait a minute and try again.";
      case 502: return "The AI services did not answer (both the main one and the backup). Try again in a moment.";
      case 503: return "The firewall is running, but no AI service is set up, so it can check messages but not answer them.";
      default: return `The firewall returned an error (HTTP ${e.status}). Try again.`;
    }
  }
  if (e instanceof DOMException && e.name === "AbortError") return "Cancelled.";
  return "Could not reach the firewall. Check that it is running and try again.";
}
