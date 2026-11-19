// Drives the DEPLOYED site in headless Chromium and checks what a visitor would see; also captures the README screenshots.
//
//   cd frontend
//   node scripts/live_check.mjs https://mask-wall.vercel.app --shots ../docs/screenshots
//
// Makes ONE real AI call (a Chat message with personal data) and otherwise only attacks/scans, which never reach the AI.
// It prints a PASS/FAIL line per check and exits 1 if any failed. Requires `npx playwright install chromium` once.
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";

const site = (process.argv[2] ?? "https://mask-wall.vercel.app").replace(/\/$/, "");
const shotsDir = process.argv.includes("--shots") ? process.argv[process.argv.indexOf("--shots") + 1] : null;
if (shotsDir) mkdirSync(shotsDir, { recursive: true });

const results = [];
const check = (name, ok, detail = "") => {
  results.push(ok);
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  | " + detail : ""}`);
};

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1280, height: 1000 }, colorScheme: "dark" });
const page = await ctx.newPage();
const problems = [];
const requests = [];
const chatBodies = [];
page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
page.on("console", (m) => {
  if (m.type() === "error") problems.push(`console: ${m.text().slice(0, 200)}`);
});
page.on("request", (r) => {
  requests.push(r.url());
  if (r.method() === "POST" && r.url().endsWith("/chat")) chatBodies.push(r.postData() ?? "");
});
const shot = async (name) => shotsDir && (await page.screenshot({ path: join(shotsDir, name) }));

try {
  // 1. loads, firewall reachable from the site's origin (CORS), no console errors
  await page.goto(`${site}/chat`);
  await page.getByText("Firewall online").waitFor({ timeout: 4 * 60_000 });
  check("site loads and shows 'Firewall online' (CORS from the site's origin works)", true);

  // 2. Chat: personal data masked on the device; the request that left the browser carried only placeholders; one real AI answer
  const email = "rahul.v@example.in";
  const text = `Email Rahul Verma at ${email} or call 98765 43210 about the PAN ABCPE1234F.`;
  await page.getByLabel("Your message").fill(text);
  await page.getByRole("button", { name: "Send" }).click();
  await page.locator(".received-text").waitFor({ timeout: 5 * 60_000 });   // first visit downloads the name model from the Hub
  await page.locator(".result-line").first().waitFor({ timeout: 2 * 60_000 });
  const received = await page.locator(".received-text").innerText();
  check("Chat: the server received placeholders, not the personal data", /\[EMAIL_1\]/.test(received) && /\[PHONE_1\]/.test(received) && /\[PAN_1\]/.test(received) && !received.includes(email), received.slice(0, 90));
  check("Chat: the name was hidden on the device too", /\[NAME_1\]/.test(received), "");
  check("Chat: the POST body that left the browser has no email, phone or PAN", chatBodies.length === 1 && !/rahul\.v@example\.in|98765 43210|ABCPE1234F/.test(chatBodies[0]));
  const reply = await page.locator(".bubble-ai").last().innerText();
  check("Chat: an AI answer came back through the firewall", reply.length > 20 && !/Not answered/.test(reply), reply.slice(0, 80).replace(/\n/g, " "));
  await shot("chat.png");

  // 3. the early-warning model came from the Hub (pinned commit), not from the site
  const hubModel = requests.filter((u) => u.includes("huggingface.co") && u.includes("namya06/maskwall-injection-minilm"));
  check("pre-check model was downloaded from the Hugging Face Hub", hubModel.some((u) => u.includes("model_quantized.onnx")), `${hubModel.length} requests`);
  check("the site did not serve a model copy itself", !requests.some((u) => u.startsWith(site) && u.includes("/models/maskwall-minilm")));
  const pre = await page.getByText(/Early warning|early warning/).first().isVisible().catch(() => false);
  check("the early-warning panel is shown", pre);

  // 4. Playground: scan only (no AI)
  await page.goto(`${site}/playground`);
  await page.getByRole("button", { name: /Hinglish attack/ }).click();
  await page.getByRole("button", { name: "Check message" }).click();
  await page.getByRole("heading", { name: /Blocked: this looks like an attack\./ }).waitFor({ timeout: 60_000 });
  check("Playground: the Hinglish attack is blocked by the live server", true);
  await shot("playground.png");

  // 5. Dashboard shows live numbers
  await page.goto(`${site}/dashboard`);
  await page.getByText(/never saved|never stored/i).first().waitFor({ timeout: 60_000 });
  await page.waitForTimeout(1500);
  const dash = await page.locator("main").innerText();
  check("Dashboard: loads with live counts and the privacy statement", /attack/i.test(dash) && !/could not reach/i.test(dash), "");
  await shot("dashboard.png");

  // 6. How it works: Render numbers, daily limit from /health
  await page.goto(`${site}/how-it-works`);
  const how = await page.locator("main").innerText();
  check("How it works: hosting label and the daily limit are shown", how.includes("Render free tier (0.1 vCPU)") && /500 per day/.test(how), "");
  await shot("how-it-works.png");

  // 7. Benchmarks section 3b
  await page.goto(`${site}/benchmarks#speed`);
  const bench = await page.locator("main").innerText();
  check("Benchmarks: section 3b (speed and memory on the deployed server) is present", bench.includes("3b. Speed and memory on the deployed server") && bench.includes("50 ms speed target"));
  await page.locator("#speed").scrollIntoViewIfNeeded();
  await shot("benchmarks.png");

  // 8. deep link refresh works (SPA rewrite) and no console errors anywhere
  const r = await page.goto(`${site}/benchmarks`);
  check("deep link /benchmarks loads after a hard navigation", r?.status() === 200);
  check("no console errors or page errors on any page", problems.length === 0, problems.slice(0, 3).join(" || "));
} catch (e) {
  check("script completed", false, String(e).slice(0, 300));
} finally {
  await browser.close();
}
const failed = results.filter((x) => !x).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
