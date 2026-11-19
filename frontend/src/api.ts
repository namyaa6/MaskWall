// Thin client for the MaskWall firewall API (backend/README.md). Base URL from VITE_API_URL (apiConfig.ts), default: local uvicorn.
// Every call first makes sure the firewall is awake (serverState.ts): free hosting sleeps when idle and takes a minute or two to wake.
import { API_URL, type Health } from "./apiConfig";
import { ensureAwake, noteServerOk, noteServerUnreachable, ServerDownError } from "./serverState";

export { API_URL };
export type { Health };

/** The firewall accepted the request but did not answer in time (the message may or may not have been processed). */
export class TimeoutError extends Error {
  constructor() {
    super("request timed out");
  }
}

/** A request that is not answered within this long is abandoned, so a stalled server cannot leave the page stuck. */
export const REQUEST_TIMEOUT_MS = 90_000;

/** The request could not reach the firewall at all (network error), as opposed to the firewall answering with an error. */
export class NetworkError extends Error {
  constructor() {
    super("network error");
  }
}

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(`HTTP ${status}`);
  }
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  await ensureAwake(); // queues behind the wake-up; throws ServerDownError after about 3 minutes
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, REQUEST_TIMEOUT_MS);
  const onCallerAbort = () => ctrl.abort();
  if (init.signal?.aborted) ctrl.abort();
  else init.signal?.addEventListener("abort", onCallerAbort, { once: true });
  try {
    let r: Response;
    try {
      r = await fetch(`${API_URL}${path}`, { ...init, signal: ctrl.signal });
    } catch (e) {
      if (timedOut) throw new TimeoutError();
      if (e instanceof DOMException && e.name === "AbortError") throw e;
      noteServerUnreachable(); // never re-sent automatically: a POST may already have reached the server
      throw new NetworkError();
    }
    noteServerOk();
    if (!r.ok) throw new ApiError(r.status, await r.json().catch(() => null));
    return (await r.json()) as T;
  } catch (e) {
    if (timedOut && !(e instanceof TimeoutError) && !(e instanceof ApiError)) throw new TimeoutError();
    throw e;
  } finally {
    clearTimeout(timer);
    init.signal?.removeEventListener("abort", onCallerAbort);
  }
}

export const getJSON = <T>(path: string, signal?: AbortSignal) => request<T>(path, { signal });

export const postJSON = <T>(path: string, body: unknown, signal?: AbortSignal) =>
  request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });

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

/** The firewall's own explanation for a 429 that is the global daily quota (not the per-visitor rate limit). */
export function dailyLimitMessage(e: unknown): string | null {
  if (!(e instanceof ApiError) || e.status !== 429) return null;
  const d = (e.detail as { detail?: { error?: string; message?: string } } | null)?.detail;
  return d?.error === "daily_llm_limit_reached" || d?.error === "client_daily_llm_limit_reached" ? (d.message ?? null) : null;
}

/** Plain-English message for an error from the firewall. */
export function describeError(e: unknown): string {
  if (e instanceof ServerDownError) return "The firewall is not answering. Free hosting can take a while to start up: wait a minute and try again.";
  if (e instanceof TimeoutError) return "The firewall took too long to answer. Your message may or may not have been processed; wait a moment and try again.";
  if (e instanceof NetworkError) return "Could not reach the firewall. Check your connection, then try again.";
  if (e instanceof ApiError) {
    switch (e.status) {
      case 413: return "That message is too long for the firewall (over 8,000 characters). Shorten it and try again.";
      case 422: return "That message is empty once hidden characters are removed, so there was nothing to send.";
      case 429: return dailyLimitMessage(e) ?? "Too many messages in a short time. Wait a minute and try again.";
      case 502: return "The AI services did not answer (both the main one and the backup). Try again in a moment.";
      case 503: return "The firewall is running, but no AI service is set up, so it can check messages but not answer them.";
      default: return `The firewall returned an error (HTTP ${e.status}). Try again.`;
    }
  }
  if (e instanceof DOMException && e.name === "AbortError") return "Cancelled.";
  return "Could not reach the firewall. Check that it is running and try again.";
}
