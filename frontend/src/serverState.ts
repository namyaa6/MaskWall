/**
 * "Is the firewall awake?" Free hosting puts the server to sleep after about 15 minutes without traffic, and waking it takes up to
 * a minute or two. During that time a request either hangs, or is answered by the host's own page, which has no CORS headers, so
 * the browser reports only a generic network error (indistinguishable from a real misconfiguration). So: a first contact that is
 * slow or fails is treated as "waking up", not as an error, and is retried until a cap (about 3 minutes); only then is it "down".
 *
 * Everything that talks to the API goes through ensureAwake() (see api.ts), so a message typed while the server sleeps is simply
 * sent when it wakes. The state is a tiny external store (useSyncExternalStore in WakeBanner.tsx).
 */
import { API_URL, IS_LOCAL_API, type Health } from "./apiConfig";

export type ServerStatus = "unknown" | "waking" | "awake" | "down";
export interface ServerState {
  status: ServerStatus;
  /** when the current wake-up started (ms since epoch), null when not waking */
  since: number | null;
  attempt: number;
  health: Health | null;
}

export const WAKE_BANNER_AFTER_MS = 4_000;        // a /health that takes longer than this means "waking", so the banner appears
export const WAKE_RETRY_EVERY_MS = 3_000;
export const WAKE_GIVE_UP_AFTER_MS = 180_000;     // about 3 minutes, then "down"
export const LOCAL_GIVE_UP_AFTER_MS = 8_000;
const ATTEMPT_TIMEOUT_MS = 75_000;
/** The host sleeps after ~15 min idle; a "known awake" state older than this is re-checked before the next request. */
export const AWAKE_TTL_MS = 10 * 60_000;

export class ServerDownError extends Error {
  constructor() {
    super("The firewall did not wake up in time");
  }
}

let state: ServerState = { status: "unknown", since: null, attempt: 0, health: null };
let lastOkAt = 0;
let inflight: Promise<Health> | null = null;
const listeners = new Set<() => void>();

export const getServerState = (): ServerState => state;
export function subscribeServerState(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}
function set(patch: Partial<ServerState>) {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}

/** Test helper: back to a fresh state. */
export function resetServerState() {
  state = { status: "unknown", since: null, attempt: 0, health: null };
  lastOkAt = 0;
  inflight = null;
  listeners.clear();
}

/** Called by api.ts after any successful response: the server is demonstrably awake right now. */
export function noteServerOk() {
  lastOkAt = Date.now();
  if (state.status !== "awake") set({ status: "awake", since: null });
}
/** Called by api.ts when a request failed at the network level after the server had been awake: re-check before the next one. */
export function noteServerUnreachable() {
  lastOkAt = 0;
  if (state.status === "awake") set({ status: "unknown" });
}

async function probe(timeoutMs: number): Promise<Health> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(`${API_URL}/health`, { signal: ctrl.signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return (await r.json()) as Health;
  } finally {
    clearTimeout(timer);
  }
}

const sleep = (ms: number) => new Promise<void>((res) => setTimeout(res, ms));

export interface WakeOptions {
  giveUpAfterMs?: number;
  bannerAfterMs?: number;
  retryEveryMs?: number;
}

/** Wake the firewall (or confirm it is awake). One attempt loop is shared by all callers. Rejects with ServerDownError after the cap. */
export function wakeUp(o: WakeOptions = {}): Promise<Health> {
  if (inflight) return inflight;
  const giveUp = o.giveUpAfterMs ?? (IS_LOCAL_API ? LOCAL_GIVE_UP_AFTER_MS : WAKE_GIVE_UP_AFTER_MS);
  const retryEvery = o.retryEveryMs ?? WAKE_RETRY_EVERY_MS;
  const started = Date.now();
  set({ since: started, attempt: 0 });
  // a first contact that is merely slow also means "waking": show the banner without waiting for the request to fail
  const bannerTimer = IS_LOCAL_API
    ? undefined
    : setTimeout(() => {
        if (state.status !== "awake") set({ status: "waking" });
      }, o.bannerAfterMs ?? WAKE_BANNER_AFTER_MS);
  inflight = (async () => {
    try {
      for (let attempt = 1; ; attempt++) {
        set({ attempt });
        const left = giveUp - (Date.now() - started);
        try {
          const health = await probe(Math.min(ATTEMPT_TIMEOUT_MS, Math.max(left, 1_000)));
          lastOkAt = Date.now();
          set({ status: "awake", since: null, health });
          return health;
        } catch {
          if (!IS_LOCAL_API) set({ status: "waking" });
        }
        if (Date.now() - started >= giveUp) {
          set({ status: "down", since: null });
          throw new ServerDownError();
        }
        await sleep(retryEvery);
      }
    } finally {
      clearTimeout(bannerTimer);
      inflight = null;
    }
  })();
  return inflight;
}

/** Resolve when the firewall is known to be awake (immediately if it answered recently). */
export async function ensureAwake(o?: WakeOptions): Promise<void> {
  if (state.status === "awake" && Date.now() - lastOkAt < AWAKE_TTL_MS) return;
  await wakeUp(o);
}
