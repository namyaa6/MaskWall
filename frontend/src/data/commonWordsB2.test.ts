import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { buildB2File, B2_PATH, STATS_PATH } from "../../scripts/commonWordsB2";
import data from "./commonWordsB2.json";

const repo = fileURLToPath(new URL("../../../", import.meta.url));

describe("commonWordsB2.json (the list behind Strict name hiding)", () => {
  const words = new Set<string>(data.words);

  it("is the frozen round-2 setting, with its licences and attribution recorded", () => {
    expect(data.config).toEqual({ tEn: 500, kH: 2, share: 0.8, maxPersonShare: 0.5 });
    expect(data.sources.map((s) => s.license)).toEqual(["Apache-2.0", "CC BY-SA 3.0 and GFDL"]);
    expect(data.notice).toMatch(/Wikipedia/);
    expect(data.notice).toMatch(/Hinglish-TOP/);
  });
  it("has the 5,450 words that were evaluated in round 2, sorted and without duplicates", () => {
    expect(data.words).toHaveLength(5450);
    expect(words.size).toBe(5450);
    expect([...data.words]).toEqual([...data.words].sort());
  });
  it("contains everyday function words and not common first names", () => {
    for (const w of ["the", "ko", "hai", "kal", "karo"]) expect(words.has(w), w).toBe(true);
    for (const w of ["priya", "rahul", "john", "sarah", "amit", "alex", "neha", "pooja", "emily", "michael"]) expect(words.has(w), w).toBe(false);
  });
  it("KNOWN LIMIT: a few first names leak in from non-person slots of the small TOP corpus, so Strict mode will not find them", () => {
    // 2 of 52 common first names checked ("david", "jane"): the frozen list is what was evaluated in round 2 (it missed "david" there too).
    expect(words.has("david")).toBe(true);
    expect(words.has("jane")).toBe(true);
  });
  it("stays small enough to load on demand (< 100 KB)", () => {
    expect(readFileSync(B2_PATH(repo)).length).toBeLessThan(100_000);
  });
  it.skipIf(!existsSync(STATS_PATH(repo)))("is up to date with ml/cache statistics (run: npm run export-common-words)", () => {
    expect(JSON.parse(JSON.stringify(buildB2File(repo)))).toEqual(data);
  });
});
