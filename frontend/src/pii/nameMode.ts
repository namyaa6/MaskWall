/**
 * Which name-finding method a message uses.
 *   standard  the cased NER model on the text as typed (default; what was measured in Day 3.2)
 *   strict    approach B2 (docs/decisions.md Day 3.13): the same model on a copy where unknown lowercase words get a capital letter,
 *             with the independent common-word list. OPT-IN: it finds most lowercase names but also hides some ordinary words
 *             (round 2 on a sealed set: 48 of 60 lowercase names found, 16 of 60 harmless sentences wrongly flagged).
 *             It was added AFTER seeing those results, as a user choice, not as a default.
 */
import type { PiiSpan } from "./detectors";
import { detectNames, type CommonWords, type NameModel } from "./nameApproaches";
import { dropNonNames } from "./nonNames";

export type NameMode = "standard" | "strict";

export async function detectNamesForMode(mode: NameMode, text: string, model: NameModel, loadWords: () => Promise<CommonWords>): Promise<PiiSpan[]> {
  const spans = mode === "strict" ? await detectNames("B", text, model, await loadWords()) : await model.detect(text);
  return dropNonNames(text, spans);      // Day 4.24: "Aadhaar", "PAN", "UPI" ... are never a person's name
}

/**
 * True when a message is mostly lowercase (at most 5% of its cased letters are capitals) and long enough to matter (12+ cased letters).
 * The standard name model learned from news text where names are capitalised, so names in such a message are often not found.
 */
export function isMostlyLowercase(text: string): boolean {
  let lower = 0;
  let upper = 0;
  for (const ch of text) {
    if (ch === ch.toLowerCase() && ch !== ch.toUpperCase()) lower++;
    else if (ch === ch.toUpperCase() && ch !== ch.toLowerCase()) upper++;
  }
  const cased = lower + upper;
  return cased >= 12 && upper / cased <= 0.05;
}
