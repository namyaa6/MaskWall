/**
 * The "Waking up the firewall" logic (serverState.ts + api.ts). fetch is mocked and time is fake, so this covers: awake at once,
 * a slow first contact, a host that answers with network errors while it starts, a host that never wakes (cap), request queueing
 * behind the wake-up, no automatic re-send of a POST, and the daily-limit message.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./apiConfig", () => ({ API_URL: "https://firewall.example", IS_LOCAL_API: false }));

import { ApiError, describeError, getHealth, NetworkError, postChat } from "./api";
import { WAKING_TEXT } from "./components/WakeBanner";
import { AWAKE_TTL_MS, ensureAwake, getServerState, resetServerState, ServerDownError, wakeUp } from "./serverState";

const HEALTH = { status: "ok", version: "0.1.0", server_model: "minilm", scan_mode: "sliding", model_loaded: true, max_chars: 8000, llm_providers: ["groq"], daily_llm_limit: 500, daily_llm_remaining: 498 };
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const CHAT = { request_id: "x", action: "allow", response: "hi", input: {}, latency_ms: {}, truncated: false, output: null, llm: null };

let fetchMock: ReturnType<typeof vi.fn>;
beforeEach(() => {
  vi.useFakeTimers();
  resetServerState();
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("waking up the firewall", () => {
  it("the banner text is the one the owner asked for", () => {
    expect(WAKING_TEXT).toBe("Waking up the firewall. This can take up to a minute or two.");
  });

  it("an awake server answers at once: no banner state at any point", async () => {
    fetchMock.mockImplementation(async () => json(HEALTH));   // a fresh Response per call: a body can be read once
    const seen: string[] = [];
    const p = wakeUp();
    seen.push(getServerState().status);
    const h = await p;
    await vi.advanceTimersByTimeAsync(10_000);
    seen.push(getServerState().status);
    expect(h.model_loaded).toBe(true);
    expect(seen.every((s) => s !== "waking")).toBe(true);
    expect(getServerState()).toMatchObject({ status: "awake", since: null });
  });

  it("a slow first contact shows the banner after 4 s and clears it when the server answers", async () => {
    let answer!: () => void;
    fetchMock.mockImplementationOnce(() => new Promise<Response>((res) => (answer = () => res(json(HEALTH)))));
    const p = wakeUp();
    await vi.advanceTimersByTimeAsync(3_900);
    expect(getServerState().status).not.toBe("waking");
    await vi.advanceTimersByTimeAsync(200);
    expect(getServerState().status).toBe("waking");
    await vi.advanceTimersByTimeAsync(40_000);
    expect(getServerState().status).toBe("waking");      // still hanging: still the banner, not an error
    answer();
    await p;
    expect(getServerState().status).toBe("awake");
  });

  it("network errors while the host starts mean 'waking' and are retried every 3 s until it answers", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch")).mockRejectedValueOnce(new TypeError("Failed to fetch")).mockImplementation(async () => json(HEALTH));
    const p = wakeUp();
    await vi.advanceTimersByTimeAsync(1);
    expect(getServerState().status).toBe("waking");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(3_000);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(3_000);
    await p;
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(getServerState().status).toBe("awake");
  });

  it("gives up after the cap with ServerDownError and the 'down' state (default cap is about 3 minutes)", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const p = wakeUp({ giveUpAfterMs: 20_000 });
    const failed = p.catch((e) => e);
    await vi.advanceTimersByTimeAsync(25_000);
    expect(await failed).toBeInstanceOf(ServerDownError);
    expect(getServerState().status).toBe("down");
    expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(6);
    expect((await import("./serverState")).WAKE_GIVE_UP_AFTER_MS).toBe(180_000);
  });

  it("one wake-up loop is shared by all callers", async () => {
    fetchMock.mockImplementation(async () => json(HEALTH));   // a fresh Response per call: a body can be read once
    await Promise.all([wakeUp(), wakeUp(), ensureAwake()]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("a recent answer is trusted until the TTL, then re-checked (free hosting sleeps after about 15 minutes idle)", async () => {
    fetchMock.mockImplementation(async () => json(HEALTH));   // a fresh Response per call: a body can be read once
    await ensureAwake();
    await vi.advanceTimersByTimeAsync(AWAKE_TTL_MS - 1_000);
    await ensureAwake();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(2_000);
    await ensureAwake();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("requests while the server sleeps", () => {
  it("a message sent during the wake-up is held, then sent exactly once when the server is awake", async () => {
    const calls: string[] = [];
    fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
      calls.push(`${init?.method ?? "GET"} ${url.replace("https://firewall.example", "")}`);
      if (url.endsWith("/health")) {
        if (calls.filter((c) => c.endsWith("/health")).length < 3) throw new TypeError("Failed to fetch");
        return json(HEALTH);
      }
      return json(CHAT);
    });
    const sent = postChat("hello");
    await vi.advanceTimersByTimeAsync(1);
    expect(getServerState().status).toBe("waking");
    expect(calls).toEqual(["GET /health"]);                 // nothing else has been sent
    await vi.advanceTimersByTimeAsync(7_000);
    const res = await sent;
    expect(res.response).toBe("hi");
    expect(calls.filter((c) => c === "POST /chat")).toHaveLength(1);
    expect(calls[calls.length - 1]).toBe("POST /chat");
    expect(getServerState().status).toBe("awake");
  });

  it("a POST that fails at the network level is NOT re-sent (it may have reached the server) and the next call re-checks health", async () => {
    fetchMock.mockResolvedValueOnce(json(HEALTH)).mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await expect(postChat("hello")).rejects.toBeInstanceOf(NetworkError);
    expect(fetchMock).toHaveBeenCalledTimes(2);             // health + one POST, no retry
    expect(getServerState().status).toBe("unknown");
    fetchMock.mockResolvedValueOnce(json(HEALTH)).mockResolvedValueOnce(json(HEALTH));
    await getHealth();
    expect(fetchMock).toHaveBeenCalledTimes(4);             // the next call checked health first
  });

  it("when the server never wakes, the request fails with ServerDownError instead of hanging forever", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const r = postChat("hello").catch((e) => e);
    await vi.advanceTimersByTimeAsync(200_000);
    expect(await r).toBeInstanceOf(ServerDownError);
    expect(fetchMock.mock.calls.every((c) => String(c[0]).endsWith("/health"))).toBe(true);   // the message itself was never sent
  });
});

describe("error messages", () => {
  const limit = new ApiError(429, { detail: { error: "daily_llm_limit_reached", message: "The demo's free daily quota for AI answers has been used up. Try the Attack playground." } });

  it("the global daily quota shows the firewall's own message (which points to the Attack playground)", () => {
    expect(describeError(limit)).toContain("Attack playground");
  });
  it("the per-visitor rate limit keeps its own wording", () => {
    expect(describeError(new ApiError(429, { detail: { error: "rate_limited" } }))).toMatch(/Too many messages/);
    expect(describeError(new ApiError(429, null))).toMatch(/Too many messages/);
  });
  it("down and network errors say what to do", () => {
    expect(describeError(new ServerDownError())).toMatch(/wait a minute/);
    expect(describeError(new NetworkError())).toMatch(/connection/);
  });
});
