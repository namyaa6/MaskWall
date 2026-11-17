import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

// WCAG 2.x contrast ratio of every text colour on every background it is used on (target 4.5:1).
const css = readFileSync(new URL("./tokens.css", import.meta.url), "utf8");
const token = (name: string): string => {
  const m = css.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`));
  if (!m) throw new Error(`token --${name} not found`);
  return m[1];
};

function luminance(hex: string): number {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const [r, g, b] = c.map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
export function contrast(a: string, b: string): number {
  const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

const TEXT = ["text", "text-2", "accent", "safe", "suspicious", "blocked"];
const BACKGROUNDS = ["bg", "card", "card-raised"];

describe("colour contrast", () => {
  for (const fg of TEXT)
    for (const bg of BACKGROUNDS)
      it(`--${fg} on --${bg} >= 4.5`, () => expect(contrast(token(fg), token(bg))).toBeGreaterThanOrEqual(4.5));

  it("placeholder pill text >= 4.5", () => expect(contrast(token("pill-text"), token("pill-bg"))).toBeGreaterThanOrEqual(4.5));
  it("text on the accent button >= 4.5", () => expect(contrast(token("accent-ink"), token("accent"))).toBeGreaterThanOrEqual(4.5));
});
