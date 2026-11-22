// Measures what a visitor sees when the (free, sleeping) server has to wake up, in headless Chromium.
//
//   cd frontend
//   node scripts/cold_start.mjs https://mask-wall.vercel.app --wait-min 22 --out ../ml/results/cold_start.json
//
// Waits --wait-min minutes WITHOUT sending anything to the API (Render's free tier sleeps after about 15 minutes idle), then opens the
// site and records, in seconds from the moment navigation starts: when the "Waking up the firewall" banner first appeared, when it
// disappeared, and when "Firewall online" showed. It also timestamps the first /health response seen on the network. Run it only when
// nothing else is using the API: any request during the wait resets the idle timer.
import { chromium } from "@playwright/test";
import { writeFileSync } from "node:fs";

const site = (process.argv[2] ?? "https://mask-wall.vercel.app").replace(/\/$/, "");
const arg = (name, dflt) => (process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : dflt);
const waitMin = Number(arg("--wait-min", "22"));
const out = arg("--out", null);

console.log(`waiting ${waitMin} minutes with no API traffic (started ${new Date().toISOString()})`);
await new Promise((r) => setTimeout(r, waitMin * 60_000));

const browser = await chromium.launch();
const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
const healthResponses = [];
page.on("response", (r) => {
  if (r.url().endsWith("/health")) healthResponses.push({ t: (performance.now() - t0) / 1000, status: r.status() });
});
const t0 = performance.now();
const since = () => Math.round(((performance.now() - t0) / 1000) * 10) / 10;
await page.goto(`${site}/chat`);
const loadedAt = since();

let bannerShownAt = null;
let bannerGoneAt = null;
let onlineAt = null;
const banner = page.getByTestId("wake-banner");
const deadline = performance.now() + 6 * 60_000;
while (performance.now() < deadline && onlineAt === null) {
  const visible = await banner.isVisible().catch(() => false);
  if (visible && bannerShownAt === null) bannerShownAt = since();
  if (!visible && bannerShownAt !== null && bannerGoneAt === null) bannerGoneAt = since();
  if (await page.getByText("Firewall online").isVisible().catch(() => false)) onlineAt = since();
  await new Promise((r) => setTimeout(r, 250));
}
const bannerText = bannerShownAt !== null ? await banner.innerText().catch(() => "") : "";
await browser.close();

const result = {
  measured_at: new Date().toISOString(),
  site,
  idle_minutes_before: waitMin,
  how: "headless Chromium on the developer machine (India), plain HTTPS; times in seconds from the start of navigation",
  page_loaded_s: loadedAt,
  banner_first_visible_s: bannerShownAt,
  banner_gone_s: bannerGoneAt,
  firewall_online_s: onlineAt,
  banner_visible_for_s: bannerShownAt !== null && bannerGoneAt !== null ? Math.round((bannerGoneAt - bannerShownAt) * 10) / 10 : null,
  first_health_responses: healthResponses.slice(0, 5),
  banner_text_seen: bannerText.split("\n")[0] ?? "",
  was_it_asleep: bannerShownAt !== null,
};
console.log(JSON.stringify(result, null, 2));
if (out) writeFileSync(out, JSON.stringify(result, null, 2) + "\n");
