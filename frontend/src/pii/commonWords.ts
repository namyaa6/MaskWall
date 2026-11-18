/**
 * Common-word set for re-casing (docs/decisions.md Day 3.13). Built ONLY from independent corpora; no word is chosen by hand:
 *   English  (WikiText-103 raw, first train shard; Wikipedia text, CC BY-SA 3.0 / GFDL): a word is common if its lowercase count >= tEn and its
 *            lowercase share (lower / (lower + Capitalised)) >= share. The share is the name filter: "rose" and "mark" are mostly
 *            written lowercase in running text, "priya" and "john" almost never are.
 *   Hinglish (Hinglish-TOP TRAIN split, Apache-2.0): a token is common if it occurs >= kH times in the code-switched queries and
 *            fewer than half of its occurrences are inside a person/contact slot of the parse (names sit in such slots nearly every time).
 * The NER model's own lowercase vocabulary (Apache-2.0) is the third source; the approaches read it from the model.
 */
export interface CommonWordStats {
  /** [word, lowercase count, lowercase share] */
  english: [string, number, number][];
  /** [word, count in the TOP train queries, share of its occurrences inside a person/contact slot of the parse] */
  hinglish: [string, number, number][];
}

/** A token that sits inside a person/contact slot at least this often is a name, not a common word (plan, amended before any evaluation). */
export const MAX_PERSON_SHARE = 0.5;

export interface CommonWordConfig {
  tEn: number;
  kH: number;
  share: number;
}

/** The eight settings fixed in the plan (T_en x K_h x s); chosen on the round-1 development set, never on the round-2 test set. */
export const GRID: CommonWordConfig[] = [50, 500].flatMap((tEn) => [2, 5].flatMap((kH) => [0.5, 0.8].map((share) => ({ tEn, kH, share }))));

export function buildCommonWords(stats: CommonWordStats, cfg: CommonWordConfig): Set<string> {
  const out = new Set<string>();
  for (const [w, lower, share] of stats.english) if (lower >= cfg.tEn && share >= cfg.share) out.add(w);
  for (const [w, count, personShare] of stats.hinglish) if (count >= cfg.kH && personShare < MAX_PERSON_SHARE) out.add(w);
  return out;
}
