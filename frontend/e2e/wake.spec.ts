/**
 * "Waking up the firewall": the site opened while the free-tier server is asleep.
 *
 * Run with:  npm run e2e:wake   (builds with VITE_API_URL=https://firewall.e2e.test, a hosted-looking URL, because a local API
 * URL deliberately has no wake-up wait). No backend is needed: Playwright answers for the "server", which fails its first two
 * health checks (as a sleeping Render service does: the host answers without CORS headers, so the browser sees a network error)
 * and then wakes up. Skipped in the normal e2e run.
 */
import { expect, test, type Route } from "@playwright/test";

const API = "https://firewall.e2e.test";
test.skip(process.env.WAKE_E2E !== "1", "run with npm run e2e:wake (needs a build with a hosted-looking VITE_API_URL)");

const CORS = { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "GET,POST,OPTIONS" };
const HEALTH = { status: "ok", version: "0.1.0", server_model: "minilm", scan_mode: "sliding", model_loaded: true, max_chars: 8000, llm_providers: ["groq"], thresholds: { warn: 0.368, block: 0.61 }, daily_llm_limit: 500, daily_llm_remaining: 321 };
const SCAN = {
  request_id: "e2e", action: "block",
  input: { score: 0.997, verdict: "block", reasons: ["injection_detected"], notes: [], hidden_characters_removed: 0, coverage: "full", mode: "sliding", n_tokens: 12, n_windows: 1, n_windows_needed: 1, scanned_tokens: 12, thresholds: { warn: 0.368, block: 0.61 }, pii_masked: {}, masked_input: "x", windows: [{ start: 0, end: 12, score: 0.997, ms: 3 }] },
  latency_ms: { pii_ms: 0.1, tokenize_ms: 0.1, inference_ms: 3, total_ms: 4 },
};

test("asleep server: the banner appears, a message is held and sent once it wakes, then the banner goes", async ({ page }) => {
  let healthCalls = 0;
  const scans: number[] = [];
  await page.route(`${API}/**`, async (route: Route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS });
    if (path === "/health") {
      healthCalls++;
      if (healthCalls <= 2) return route.abort("failed");                 // asleep: no CORS headers, the browser reports a network error
      return route.fulfill({ status: 200, headers: CORS, contentType: "application/json", body: JSON.stringify(HEALTH) });
    }
    if (path === "/scan") {
      scans.push(healthCalls);                                           // how many health checks had happened when the message was sent
      return route.fulfill({ status: 200, headers: CORS, contentType: "application/json", body: JSON.stringify(SCAN) });
    }
    return route.fulfill({ status: 404, headers: CORS, body: "{}" });
  });

  await page.goto("/playground");
  const banner = page.getByTestId("wake-banner");
  await expect(banner).toBeVisible();
  await expect(banner).toContainText("Waking up the firewall. This can take up to a minute or two.");
  await expect(page.getByRole("status").filter({ hasText: "Waking up…" })).toBeVisible();

  // typed while the server is still asleep: the message is held, not lost and not an error
  await page.getByRole("button", { name: /Hinglish attack/ }).click();
  await page.getByRole("button", { name: "Check message" }).click();
  await expect(page.getByRole("heading", { name: /Blocked: this looks like an attack\./ })).toBeVisible({ timeout: 30_000 });
  expect(scans).toHaveLength(1);                                         // sent exactly once
  expect(scans[0]).toBeGreaterThanOrEqual(3);                            // and only after the third health check succeeded

  await expect(banner).toBeHidden();
  await expect(page.getByText("Firewall online")).toBeVisible();

  // the daily limit the server reported is stated on the How it works page
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "How it works" }).click();
  await expect(page.getByTestId("daily-quota")).toContainText("500 per day");
});
