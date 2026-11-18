import { describe, expect, it } from "vitest";
import { buildCommonWords, GRID, type CommonWordStats } from "./commonWords";
import { detectNames, recase, type NameModel } from "./nameApproaches";

const stats: CommonWordStats = {
  english: [["the", 1_000_000, 0.97], ["rose", 400, 0.8], ["mark", 900, 0.6], ["priya", 60, 0.05], ["grace", 700, 0.45], ["deep", 5000, 0.9], ["rare", 20, 0.99]],
  hinglish: [["hai", 300, 0], ["kal", 40, 0.02], ["barish", 2, 0], ["peene", 1, 0], ["ko", 900, 0.04], ["danny", 6, 1], ["mom", 8, 0.5]],
};

describe("buildCommonWords", () => {
  it("applies the count and lowercase-share thresholds (the share is the name filter)", () => {
    const w = buildCommonWords(stats, { tEn: 50, kH: 2, share: 0.5 });
    // priya (lowercase share 0.05) and grace (0.45) stay re-castable; danny and mom (person share >= 0.5) are names, ko (0.04) is a function word
    expect([...w].sort()).toEqual(["barish", "deep", "hai", "kal", "ko", "mark", "rose", "the"]);
    expect(buildCommonWords(stats, { tEn: 500, kH: 5, share: 0.8 })).toEqual(new Set(["the", "deep", "hai", "kal", "ko"]));
  });
  it("the grid is the eight pre-registered settings", () => {
    expect(GRID).toHaveLength(8);
    expect(new Set(GRID.map((g) => `${g.tEn}/${g.kH}/${g.share}`)).size).toBe(8);
  });
});

describe("round-2 word set replaces the hand-written list", () => {
  const model = { hasToken: (w: string) => w === "party" };
  it("without a set the round-1 Hinglish list applies; with a set only the set and the vocabulary do", () => {
    expect(recase("asha ne bola kal", model)).toBe("Asha ne bola kal");                       // round 1: ne, bola, kal are in the old list
    expect(recase("asha ne bola kal", model, new Set(["kal"]))).toBe("Asha ne Bola kal");     // round 2: only what the corpora gave ("ne" has 2 letters: never re-cased)
  });
  it("detectNames passes the set through to re-casing", async () => {
    let seen = "";
    const m: NameModel = { ...model, detect: async (t) => ((seen = t), []) };
    await detectNames("B", "asha ne party", m, new Set(["ne"]));
    expect(seen).toBe("Asha ne party");
  });
});
