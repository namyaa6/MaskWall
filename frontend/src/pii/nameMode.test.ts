import { describe, expect, it } from "vitest";
import { detectNamesForMode, isMostlyLowercase } from "./nameMode";
import type { NameModel } from "./nameApproaches";

describe("isMostlyLowercase", () => {
  it("is true for casual lowercase messages", () => {
    expect(isMostlyLowercase("pls tell priya that the cab is coming at 6")).toBe(true);
    expect(isMostlyLowercase("mera naam kavya hai aur mujhe help chahiye")).toBe(true);
  });
  it("tolerates a few capitals (a lone I, one capital letter) but not normal capitalisation", () => {
    expect(isMostlyLowercase("Tell priya that the cab is coming at six")).toBe(true);
    expect(isMostlyLowercase("Please tell Priya that the cab is coming at six, Rahul will pay")).toBe(false);
    expect(isMostlyLowercase("MY NAME IS ASHA AND I NEED HELP")).toBe(false);
  });
  it("ignores short text, digits, punctuation and scripts without case", () => {
    expect(isMostlyLowercase("ok thanks")).toBe(false);
    expect(isMostlyLowercase("12345 67890 !!! ???")).toBe(false);
    expect(isMostlyLowercase("मेरा नाम आशा है और मुझे मदद चाहिए")).toBe(false);
    expect(isMostlyLowercase("")).toBe(false);
  });
});

describe("detectNamesForMode", () => {
  const vocab = new Set(["tell", "that", "the", "cab", "is", "coming"]);
  let seen = "";
  const model: NameModel = {
    hasToken: (w) => vocab.has(w),
    detect: async (t) => {
      seen = t;
      // pretend the cased model only recognises a capitalised name
      const i = t.indexOf("Priya");
      return i >= 0 ? [{ start: i, end: i + 5, type: "NAME", key: "priya" }] : [];
    },
  };
  const words = async () => new Set(["kal"]) as ReadonlySet<string>;

  it("standard mode runs the model on the text as typed (a lowercase name is missed)", async () => {
    expect(await detectNamesForMode("standard", "tell priya that the cab is coming", model, words)).toEqual([]);
    expect(seen).toBe("tell priya that the cab is coming");
  });
  it("strict mode (B2) re-casts unknown lowercase words first, using the loaded word list, and finds the name", async () => {
    const spans = await detectNamesForMode("strict", "tell priya that the cab is coming kal", model, words);
    expect(seen).toBe("tell Priya that the cab is coming kal");        // 'kal' is in the list, so it is not capitalised
    expect(spans).toHaveLength(1);
    expect(spans[0]).toMatchObject({ start: 5, end: 10, key: "priya" });
  });
  it("the word list is loaded only in strict mode", async () => {
    let loads = 0;
    const counting = async () => (loads++, new Set<string>() as ReadonlySet<string>);
    await detectNamesForMode("standard", "tell priya", model, counting);
    expect(loads).toBe(0);
    await detectNamesForMode("strict", "tell priya", model, counting);
    expect(loads).toBe(1);
  });
});
