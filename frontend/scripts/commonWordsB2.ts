/**
 * Builds frontend/src/data/commonWordsB2.json: the common-word list used by the opt-in "Strict name hiding" (approach B2).
 * It is generated from ml/cache/common_words_stats.json (made by ml/data/build_common_words.py) with the setting that was FROZEN
 * before the round-2 evaluation (frontend/src/data/common_words_config.json; docs/decisions.md Day 3.13). No word is chosen by hand.
 *
 *   cd frontend; npm run export-common-words
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { buildCommonWords, MAX_PERSON_SHARE, type CommonWordConfig, type CommonWordStats } from "../src/pii/commonWords";

export const B2_PATH = (repo: string) => join(repo, "frontend", "src", "data", "commonWordsB2.json");
export const STATS_PATH = (repo: string) => join(repo, "ml", "cache", "common_words_stats.json");

export function buildB2File(repo: string) {
  const cfg = JSON.parse(readFileSync(join(repo, "frontend", "src", "data", "common_words_config.json"), "utf8")) as CommonWordConfig;
  const stats = JSON.parse(readFileSync(STATS_PATH(repo), "utf8")) as CommonWordStats & { meta: unknown };
  const config = { tEn: cfg.tEn, kH: cfg.kH, share: cfg.share, maxPersonShare: MAX_PERSON_SHARE };
  const words = [...buildCommonWords(stats, config)].sort();
  return {
    _generated_by: "frontend/scripts/commonWordsB2.ts from ml/cache/common_words_stats.json (do not edit by hand)",
    purpose: "Words that Strict name hiding never capitalises before running the name model (docs/decisions.md Day 3.13). The model's own lowercase vocabulary is added in the browser.",
    config,
    sources: [
      { part: "Hinglish words", source: "Hinglish-TOP, train split (WillHeld/hinglish_top, human code-switched assistant queries)", license: "Apache-2.0" },
      { part: "English words", source: "WikiText-103 raw, first train shard (Salesforce/wikitext; Wikipedia text), lowercase counts and shares", license: "CC BY-SA 3.0 and GFDL" },
    ],
    notice: "This list is derived from Wikipedia text (WikiText-103) and Hinglish-TOP. Attribution: Wikipedia contributors (CC BY-SA 3.0 / GFDL); Hinglish-TOP, Google Research (Apache-2.0). The list is shared under CC BY-SA 3.0.",
    words,
  };
}

export function writeB2File(repo = resolve(__dirname, "..", "..")): string {
  writeFileSync(B2_PATH(repo), JSON.stringify(buildB2File(repo)) + "\n");
  return B2_PATH(repo);
}
