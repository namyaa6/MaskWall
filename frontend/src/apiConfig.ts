// Where the firewall API lives. VITE_API_URL is set at build time (Vercel); the default is the local uvicorn.
export const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") || "http://127.0.0.1:8000";

/** A local backend is either up or not: no "waking up" wait (the hosted free tier sleeps; a laptop does not). */
export const IS_LOCAL_API = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/.test(API_URL);

export interface Health {
  status: string;
  version: string;
  server_model: string;
  scan_mode: string;
  model_loaded: boolean;
  max_chars: number;
  llm_providers: string[];
  thresholds?: { warn: number; block: number };
  /** global cap on LLM calls per UTC day (null = no cap) and how many are left today */
  daily_llm_limit?: number | null;
  daily_llm_remaining?: number | null;
}
