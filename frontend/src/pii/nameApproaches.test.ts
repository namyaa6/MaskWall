import { describe, expect, it } from "vitest";
import { cueSpans, detectNames, mergeNameSpans, recase, type NameModel } from "./nameApproaches";
import type { PiiSpan } from "./detectors";

// A stand-in for the NER model's vocabulary: ordinary lowercase words are tokens, names and places are not.
// (The real vocabulary has the same property: see docs/decisions.md Day 3.12.)
const VOCAB = new Set("tell call ask the is cab coming that please rose deep joy will grace mark apple bush garden meeting to and hello at five mail see today".split(" "));
const vocab = { hasToken: (w: string) => VOCAB.has(w) };
const slices = (text: string, spans: PiiSpan[]) => spans.map((s) => text.slice(s.start, s.end));

describe("recase (approach B)", () => {
  it("capitalises only non-common lowercase words and keeps the length", () => {
    const t = "tell asha that the cab is coming, joy";
    const r = recase(t, vocab);
    expect(r).toBe("tell Asha that the cab is coming, joy");
    expect(r).toHaveLength(t.length);
  });
  it("leaves words in emails, URLs and handles, short words and capitalised words alone", () => {
    expect(recase("mail asha@example.com or see www.example.org, ok", vocab)).toBe("mail asha@example.com or see www.example.org, ok");
    expect(recase("Hello kiran", vocab)).toBe("Hello Kiran");
    expect(recase("om is at five", vocab)).toBe("om is at five");
  });
  it("treats Hinglish function words as common", () => {
    expect(recase("kal asha ne bola", vocab)).toBe("kal Asha ne bola");
  });
});

describe("cueSpans (approach C)", () => {
  const found = (t: string) => slices(t, cueSpans(t, vocab));
  it("name after a verb or greeting", () => {
    expect(found("please tell asha that")).toEqual(["asha"]);
    expect(found("hello kiran")).toEqual(["kiran"]);
    expect(found("call mr verma at five")).toEqual(["verma"]);
  });
  it("two-word names, but never a common word", () => {
    expect(found("tell asha verma to call")).toEqual(["asha verma"]);
    expect(found("tell the garden")).toEqual([]);
  });
  it("name before a Hinglish particle, honorific or reporting verb", () => {
    expect(found("kiran ne bola")).toEqual(["kiran"]);
    expect(found("verma ji aaye")).toEqual(["verma"]);
    expect(found("asha said hello")).toEqual(["asha"]);
  });
  it("introductions and 'talk to'", () => {
    expect(found("my name is asha rao")).toEqual(["asha rao"]);
    expect(found("mera naam kiran hai")).toEqual(["kiran"]);
    expect(found("talk to kiran today")).toEqual(["kiran"]);
  });
  it("list continuation after a found name", () => {
    expect(found("ask arun and meera")).toEqual(["arun", "meera"]);
  });
  it("does not read places or unknown words as names without a cue", () => {
    expect(found("meeting in delhi on monday")).toEqual([]);
    expect(found("delhi is hot")).toEqual([]);
  });
  it("ignores tokens inside emails and numbers", () => {
    expect(found("mail asha@example.com")).toEqual([]);
    expect(found("call 98765")).toEqual([]);
  });
});

describe("mergeNameSpans and detectNames", () => {
  const sp = (start: number, end: number): PiiSpan => ({ start, end, type: "NAME", key: "" });
  it("merges overlapping spans and spans one space apart, and rebuilds keys from the text", () => {
    const t = "asha verma left";
    const m = mergeNameSpans(t, [sp(0, 4), sp(5, 10), sp(2, 6)]);
    expect(m).toHaveLength(1);
    expect([m[0].start, m[0].end, m[0].key]).toEqual([0, 10, "asha verma"]);
  });

  const fake = (spans: (t: string) => [number, number][]): NameModel => ({
    ...vocab,
    detect: async (t: string) => spans(t).map(([start, end]) => ({ start, end, type: "NAME" as const, key: t.slice(start, end).toLowerCase() })),
  });

  it("A passes the model's spans through untouched", async () => {
    const m = fake(() => [[5, 9]]);
    expect(await detectNames("A", "tell asha", m)).toEqual([{ start: 5, end: 9, type: "NAME", key: "asha" }]);
  });
  it("B runs the model on the re-cased copy and maps spans back to the original text", async () => {
    let seen = "";
    const m = fake((t) => {
      seen = t;
      return [[5, 9]];
    });
    const r = await detectNames("B", "tell asha", m);
    expect(seen).toBe("tell Asha");
    expect(r[0].key).toBe("asha");
  });
  it("C and E add cue spans to the model's and merge duplicates", async () => {
    const m = fake(() => [[5, 9]]);
    expect(slices("tell asha verma", await detectNames("C", "tell asha verma", m))).toEqual(["asha verma"]);
    expect(slices("tell asha verma", await detectNames("E", "tell asha verma", m))).toEqual(["asha verma"]);
  });
});
