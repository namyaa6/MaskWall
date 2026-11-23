import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { aadhaarVerified, detect, panVerified, verhoeffCheckDigit, verhoeffValid } from "./detectors";
import { normalizeText } from "./normalize";
import { PlaceholderVault } from "./vault";

// The same cases the backend's pytest suite runs (backend/tests/test_pii_regex.py), with the browser prefix "".
interface PiiCase { id: string; text: string; counts: Record<string, number>; masked?: string; unverified?: Record<string, number> }
interface NormCase { id: string; text: string; normalized: string; smuggling: number; invisible: number }
const FIXTURE = JSON.parse(
  readFileSync(new URL("../../../backend/tests/fixtures/pii_cases.json", import.meta.url), "utf8"),
) as { cases: PiiCase[]; normalize_cases: NormCase[] };

const aadhaar = (payload = "23456789012") => payload + verhoeffCheckDigit(payload);

describe("shared fixture: detectors and masking (same cases as the server)", () => {
  for (const c of FIXTURE.cases) {
    it(c.id, () => {
      const r = new PlaceholderVault().mask(c.text);
      expect(r.counts).toEqual(c.counts);
      expect(r.unverified).toEqual(c.unverified ?? {});          // hidden by shape only ("looks like"): strict check failed
      if (c.masked !== undefined) expect(r.masked).toBe(c.masked.replaceAll("{P}", ""));
    });
  }
});

describe("shared fixture: Unicode normalisation (same cases as the server)", () => {
  for (const c of FIXTURE.normalize_cases) {
    it(c.id, () => {
      const n = normalizeText(c.text);
      expect([n.text, n.smuggling, n.invisible]).toEqual([c.normalized, c.smuggling, c.invisible]);
    });
  }
});

describe("Verhoeff", () => {
  it("known vector and the fixture's numbers", () => {
    expect(verhoeffCheckDigit("236")).toBe("3");
    expect(verhoeffValid("2363")).toBe(true);
    expect(verhoeffValid("2364")).toBe(false);
    expect(aadhaar()).toBe("234567890124");
    expect(aadhaar("34567890123")).toBe("345678901238");
  });
  it("catches every single-digit error and adjacent swap", () => {
    const n = aadhaar();
    for (let i = 0; i < n.length; i++)
      for (const d of "0123456789") if (d !== n[i]) expect(verhoeffValid(n.slice(0, i) + d + n.slice(i + 1))).toBe(false);
    for (let i = 0; i < n.length - 1; i++)
      if (n[i] !== n[i + 1]) expect(verhoeffValid(n.slice(0, i) + n[i + 1] + n[i] + n.slice(i + 2))).toBe(false);
  });
  it("rejects non-digits", () => {
    expect(verhoeffValid("23a3")).toBe(false);
    expect(verhoeffValid("")).toBe(false);
  });
});

describe("vault", () => {
  it("round trip: mask then unmask restores the user's text exactly", () => {
    const v = new PlaceholderVault();
    const text = "Mail Priya.Sharma@example.co.in or call +91 98765 43210, PAN ABCPE-1234-F, aadhaar 2345 6789 0124.";
    const r = v.mask(text);
    expect(r.masked).toBe("Mail [EMAIL_1] or call [PHONE_1], PAN [PAN_1], aadhaar [AADHAAR_1].");
    expect(v.unmask(r.masked)).toBe(text);
  });

  it("never unmasks the server's [S_...] placeholders", () => {
    const v = new PlaceholderVault();
    v.mask("a@b.com");
    expect(v.unmask("You wrote [EMAIL_1]; the server hid [S_EMAIL_1].")).toBe("You wrote a@b.com; the server hid [S_EMAIL_1].");
  });

  it("browser output never contains S_ placeholders", () => {
    const r = new PlaceholderVault().mask("x@y.com 9876543210 ABCPE1234F 234567890124");
    expect(r.masked).not.toMatch(/\[S_/);
  });

  it("numbering is per conversation: a later message cannot reuse a number for a different value", () => {
    const v = new PlaceholderVault();
    expect(v.mask("first a@b.com").masked).toBe("first [EMAIL_1]");
    expect(v.mask("then c@d.com and again a@b.com").masked).toBe("then [EMAIL_2] and again [EMAIL_1]");
    expect(v.unmask("[EMAIL_1] [EMAIL_2]")).toBe("a@b.com c@d.com");
  });

  it("skips a placeholder the user typed literally, and leaves that literal text alone when unmasking (owner's addition 4)", () => {
    const v = new PlaceholderVault();
    const r = v.mask("[EMAIL_1] is a label, my mail is a@b.com");
    expect(r.masked).toBe("[EMAIL_1] is a label, my mail is [EMAIL_2]");
    expect(v.unmask("[EMAIL_1] and [EMAIL_2]")).toBe("[EMAIL_1] and a@b.com");
  });

  it("a literal placeholder in an earlier message is still skipped later", () => {
    const v = new PlaceholderVault();
    v.mask("what does [PHONE_1] mean?");
    expect(v.mask("my number is 9876543210").masked).toBe("my number is [PHONE_2]");
  });

  it("unknown placeholders the AI invents are left as they are", () => {
    const v = new PlaceholderVault();
    v.mask("a@b.com");
    expect(v.unmask("[EMAIL_7] [NAME_1]")).toBe("[EMAIL_7] [NAME_1]");
  });

  it("names from the NER model are masked; regex findings win overlaps", () => {
    const v = new PlaceholderVault();
    const text = "Asha Rao (asha@x.com) and asha rao again";
    const names = [
      { start: 0, end: 8, type: "NAME" as const, key: "asha rao" },
      { start: 10, end: 20, type: "NAME" as const, key: "asha" },      // overlaps the email: dropped
      { start: 26, end: 34, type: "NAME" as const, key: "asha rao" },
    ];
    const r = v.mask(text, names);
    expect(r.masked).toBe("[NAME_1] ([EMAIL_1]) and [NAME_1] again");
    expect(r.counts).toEqual({ NAME: 2, EMAIL: 1 });
    expect(v.unmask("Hi [NAME_1]")).toBe("Hi Asha Rao");
  });

  it("findings carry no raw values", () => {
    const r = new PlaceholderVault().mask("a@b.com 9876543210");
    expect(JSON.stringify(r.findings)).not.toMatch(/a@b\.com|9876543210/);
  });

  it("clear() forgets everything", () => {
    const v = new PlaceholderVault();
    v.mask("a@b.com");
    v.clear();
    expect(v.size).toBe(0);
    expect(v.unmask("[EMAIL_1]")).toBe("[EMAIL_1]");
  });

  it("large input", () => {
    const big = "lorem ipsum 123456 dolor ".repeat(2000) + " mail x@y.com";
    const r = new PlaceholderVault().mask(big);
    expect(r.counts).toEqual({ EMAIL: 1 });
    expect(r.masked.endsWith("mail [EMAIL_1]")).toBe(true);
  });

  it("detect() spans point at the original text", () => {
    const n = aadhaar();
    const text = `x ${n} y`;
    const [s] = detect(text);
    expect(text.slice(s.start, s.end)).toBe(n);
  });
});

describe("shape-based Aadhaar and PAN (Day 4.21): the strict checks are labels, not gates", () => {
  it("the strict checks themselves", () => {
    const n = aadhaar();
    expect(aadhaarVerified(n)).toBe(true);
    expect(aadhaarVerified(n.slice(0, -1) + String((Number(n.at(-1)) + 1) % 10))).toBe(false);   // bad checksum
    expect(aadhaarVerified("0" + n.slice(1))).toBe(false);                                         // first digit 0
    expect(aadhaarVerified("1" + n.slice(1))).toBe(false);                                         // first digit 1
    expect(panVerified("ABCPE1234F") && panVerified("abcpe1234f")).toBe(true);
    expect(panVerified("ABCDE1234F")).toBe(false);
  });

  it("the exact input from the live bug report is hidden and labelled 'looks like'", () => {
    const r = new PlaceholderVault().mask("My name is Asha Rao, email asha@example.com, Aadhaar 1234 5768 9881.");
    expect(r.masked).toBe("My name is Asha Rao, email [EMAIL_1], Aadhaar [AADHAAR_1].");
    expect(r.unverified).toEqual({ AADHAAR: 1 });
  });

  it("an invalid checksum, a leading 0 or 1, ABCDE1234F and a lowercase PAN are all hidden", () => {
    const cases: [string, string][] = [
      ["234567890125", "[AADHAAR_1]"],
      ["034567890128", "[AADHAAR_1]"],
      ["1345-6789-0129", "[AADHAAR_1]"],
      ["ABCDE1234F", "[PAN_1]"],
      ["abcde1234f", "[PAN_1]"],
    ];
    for (const [text, masked] of cases) {
      const r = new PlaceholderVault().mask(text);
      expect(r.masked, text).toBe(masked);
      expect(Object.values(r.unverified).reduce((a, b) => a! + b!, 0), text).toBe(1);
    }
  });

  it("a verified Aadhaar and PAN carry no 'looks like' label", () => {
    expect(new PlaceholderVault().mask(`${aadhaar()} ABCPE1234F`).unverified).toEqual({});
  });

  it("a 12-digit number inside a longer run is NOT cut out (card numbers, 16-digit ids)", () => {
    for (const text of ["id 1234567890123456", "card 4111 1111 1111 1111", "ref 1234 5768 9881 7", "13 digits 1234567890124"]) {
      expect(detect(text), text).toEqual([]);
    }
  });

  it("a +91 phone number (also 12 digits) stays a phone number", () => {
    expect(new PlaceholderVault().mask("call +919876543210")).toMatchObject({ masked: "call [PHONE_1]", counts: { PHONE: 1 }, unverified: {} });
  });
});
