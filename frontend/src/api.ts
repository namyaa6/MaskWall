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
