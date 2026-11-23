/**
 * Day 4.24: the name model tagged "Aadhaar" as a person in "Aadhaar 1234 5768 9881 and PAN ABCDE1234F" (live bug report). Document and
 * payment terms are never names: they are removed from the NAME spans in standard and in Strict mode, and real names are left alone.
 */
import { describe, expect, it } from "vitest";
import type { PiiSpan } from "./detectors";
import type { NameModel } from "./nameApproaches";
import { detectNamesForMode } from "./nameMode";
import { dropNonNames, NON_NAME_TERMS } from "./nonNames";
import { PlaceholderVault } from "./vault";

const SENTENCE = "Aadhaar 1234 5768 9881 and PAN ABCDE1234F";                    // the exact sentence from the report
const name = (text: string, w: string, from = 0): PiiSpan => {
  const start = text.indexOf(w, from);
  return { start, end: start + w.length, type: "NAME", key: w.toLowerCase() };
};
const shown = (text: string, spans: PiiSpan[]) => spans.map((s) => text.slice(s.start, s.end));

describe("dropNonNames", () => {
  it("drops a span that is only a document or payment term, in any case", () => {
    for (const term of ["Aadhaar", "AADHAAR", "aadhaar", "Aadhar", "PAN", "Pan", "UPI", "GST", "GSTIN", "IFSC", "OTP", "KYC", "CVV", "Passport", "Voter ID", "voter id", "PAN card", "Aadhaar Card", "Driving Licence"]) {
      const text = `my ${term} is here`;
      expect(dropNonNames(text, [name(text, term)]), term).toEqual([]);
    }
  });

  it("keeps real names, and spans of other types", () => {
    const text = "Please tell Rahul Sharma and Asha about it, mail a@b.com";
    const email: PiiSpan = { start: 52, end: 59, type: "EMAIL", key: "a@b.com" };
    const spans = [name(text, "Rahul Sharma"), name(text, "Asha"), email];
    expect(dropNonNames(text, spans)).toEqual(spans);
  });

  it("removes a term at the start or end of a longer span and keeps the name", () => {
    const a = "send PAN Rahul Sharma now";
    expect(shown(a, dropNonNames(a, [name(a, "PAN Rahul Sharma")]))).toEqual(["Rahul Sharma"]);
    const b = "ask Priya Aadhaar card today";
    const out = dropNonNames(b, [name(b, "Priya Aadhaar card")]);
    expect(shown(b, out)).toEqual(["Priya"]);
    expect(out[0].key).toBe("priya");
  });

  it("a multi-word term is removed whole, so no stray word is left", () => {
    const text = "my PAN card and voter ID are ready";
    expect(dropNonNames(text, [name(text, "PAN card"), name(text, "voter ID")])).toEqual([]);
  });

  it("returns a span untouched when there is nothing to remove (even if it starts with punctuation, as the model's span did for 'Ms. Banerjee')", () => {
    const text = "Dear Ms. Banerjee, thank you";
    const span: PiiSpan = { start: 7, end: 17, type: "NAME", key: ". banerjee" };       // ". Banerjee"
    const out = dropNonNames(text, [span]);
    expect(out).toEqual([span]);
    expect(out[0]).toBe(span);
  });

  it("leaves a name that merely contains a term in the middle", () => {
    const text = "ask Rahul Pan Sharma";
    expect(shown(text, dropNonNames(text, [name(text, "Rahul Pan Sharma")]))).toEqual(["Rahul Pan Sharma"]);
  });

  it("the list is small, lowercase and has no duplicates", () => {
    expect(NON_NAME_TERMS.length).toBeLessThan(40);
    expect(NON_NAME_TERMS.every((t) => t === t.toLowerCase())).toBe(true);
    expect(new Set(NON_NAME_TERMS).size).toBe(NON_NAME_TERMS.length);
  });
});

describe("the reported sentence through the real pipeline", () => {
  // a stand-in for the name model that behaves like the live bug: any capitalised word directly in front of a number is "a person"
  const model: NameModel = {
    hasToken: () => false,
    detect: async (t) => [...t.matchAll(/\b[A-Z][a-z]+(?= \d)/g)].map((m) => ({ start: m.index!, end: m.index! + m[0].length, type: "NAME" as const, key: m[0].toLowerCase() })),
  };
  const words = async () => new Set<string>() as ReadonlySet<string>;

  it("sanity: without the filter the stand-in does tag 'Aadhaar' (so the tests below are meaningful)", async () => {
    expect(shown(SENTENCE, await model.detect(SENTENCE))).toEqual(["Aadhaar"]);
  });

  for (const mode of ["standard", "strict"] as const) {
    it(`${mode} mode: no NAME span for 'Aadhaar', so exactly 2 details are hidden and the AI gets "Aadhaar [AADHAAR_1] and PAN [PAN_1]"`, async () => {
      const names = await detectNamesForMode(mode, SENTENCE, model, words);
      expect(names).toEqual([]);
      const r = new PlaceholderVault().mask(SENTENCE, names);
      expect(r.masked).toBe("Aadhaar [AADHAAR_1] and PAN [PAN_1]");
      expect(r.findings).toHaveLength(2);
      expect(r.counts).toEqual({ AADHAAR: 1, PAN: 1 });
      expect(r.unverified).toEqual({ AADHAAR: 1, PAN: 1 });
    });
  }

  it("strict mode re-cases lowercase words first: lowercase 'aadhaar' and 'pan' are also not names", async () => {
    const lower = "my aadhaar 1234 5768 9881 and pan ABCDE1234F";
    const recasing: NameModel = { hasToken: () => false, detect: async (t) => [...t.matchAll(/\b(Aadhaar|Pan)\b/g)].map((m) => ({ start: m.index!, end: m.index! + m[0].length, type: "NAME" as const, key: m[0].toLowerCase() })) };
    expect(await detectNamesForMode("strict", lower, recasing, words)).toEqual([]);
  });

  it("a real name next to the terms is still hidden", async () => {
    const text = "Rahul Sharma: Aadhaar 1234 5768 9881";
    const both: NameModel = { hasToken: () => false, detect: async (t) => [name(t, "Rahul Sharma"), name(t, "Aadhaar")] };
    const names = await detectNamesForMode("standard", text, both, words);
    expect(shown(text, names)).toEqual(["Rahul Sharma"]);
    expect(new PlaceholderVault().mask(text, names).masked).toBe("[NAME_1]: Aadhaar [AADHAAR_1]");
  });
});
