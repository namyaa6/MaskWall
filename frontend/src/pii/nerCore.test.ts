import { describe, expect, it } from "vitest";
import { alignTokens, chunkText, evaluateNames, personSpans } from "./nerCore";

describe("alignTokens", () => {
  it("maps word pieces and punctuation back to the text", () => {
    const text = "Hi Venkatesh, call O'Brien.";
    const toks = ["[CLS]", "Hi", "Ven", "##kat", "##esh", ",", "call", "O", "'", "Brien", ".", "[SEP]"];
    const offs = alignTokens(text, toks);
    expect(offs[0]).toBeNull();
    expect(offs.slice(1, -1).map((o) => text.slice(o![0], o![1]))).toEqual(["Hi", "Ven", "kat", "esh", ",", "call", "O", "'", "Brien", "."]);
  });
  it("covers a whole word for [UNK]", () => {
    const text = "hello 😀😀 world";
    const offs = alignTokens(text, ["hello", "[UNK]", "world"]);
    expect(text.slice(offs[1]![0], offs[1]![1])).toBe("😀😀");
    expect(text.slice(offs[2]![0], offs[2]![1])).toBe("world");
  });
});

describe("personSpans", () => {
  it("groups B-PER/I-PER words, uses the first sub-token's label, ignores other entities", () => {
    const text = "Ask Priya Sharma in Delhi";
    const toks = ["[CLS]", "Ask", "Pri", "##ya", "Sharma", "in", "Delhi", "[SEP]"];
    const labels = ["O", "O", "B-PER", "I-LOC", "I-PER", "O", "B-LOC", "O"];
    const s = personSpans(text, toks, labels);
    expect(s.map((x) => text.slice(x.start, x.end))).toEqual(["Priya Sharma"]);
    expect(s[0]).toMatchObject({ type: "NAME", key: "priya sharma" });
  });
  it("B-PER after B-PER starts a new entity; offsets are shifted for chunks", () => {
    const text = "Tom Rachel";
    const s = personSpans(text, ["Tom", "Rachel"], ["B-PER", "B-PER"], 100);
    expect(s.map((x) => [x.start, x.end])).toEqual([[100, 103], [104, 110]]);
  });
});

describe("chunkText", () => {
  it("splits at whitespace and keeps offsets", () => {
    const text = "aaaa bbbb cccc dddd";
    const chunks = chunkText(text, 10);
    expect(chunks.map((c) => c.text).join("")).toBe(text);
    for (const c of chunks) expect(text.slice(c.offset, c.offset + c.text.length)).toBe(c.text);
    expect(chunks.every((c) => c.text.length <= 10)).toBe(true);
  });
});

describe("evaluateNames", () => {
  it("strict recall needs the full name; partial masking counts only as overlap; FPs are spans touching no name", () => {
    const rows = [
      { id: 1, category: "a", text: "Priya Sharma met Rahul", names: ["Priya Sharma", "Rahul"] },
      { id: 2, category: "b", text: "Visit Goa", names: [] },
    ];
    const r = evaluateNames(rows, [[{ start: 0, end: 5 }, { start: 17, end: 22 }], [{ start: 6, end: 9 }]]);
    expect([r.gold, r.strict, r.overlap, r.falsePositives, r.fpOnNoNameSentences]).toEqual([2, 1, 2, 1, 1]);
    expect(r.missed).toEqual([{ id: 1, name: "Priya Sharma" }]);
  });
});
