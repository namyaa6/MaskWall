/**
 * Model-independent part of name detection (no Transformers.js import, so it is unit-testable in Node):
 *  - alignTokens: map WordPiece tokens back to character offsets in the original text
 *    (Transformers.js tokenizers do not return an offset mapping)
 *  - personSpans: argmax labels per token -> PER entity spans ("first sub-token" word labelling, B-PER starts an entity,
 *    I-PER continues it). No confidence threshold: fixed in docs/decisions.md Day 3.2 before measuring.
 *  - chunkText: split long text at whitespace so each chunk fits the model's 512-token limit.
 */
import type { PiiSpan } from "./detectors";

const SPECIAL = new Set(["[CLS]", "[SEP]", "[PAD]", "[MASK]", "<s>", "</s>", "<pad>"]);
const isSpace = (c: string) => /\s/u.test(c);
const isPunct = (c: string) => /[\p{P}\p{S}]/u.test(c);

export type Offset = [number, number] | null;

export function alignTokens(text: string, tokens: string[]): Offset[] {
  const out: Offset[] = [];
  let cur = 0;
  for (const tok of tokens) {
    if (SPECIAL.has(tok)) {
      out.push(null);
      continue;
    }
    const cont = tok.startsWith("##");
    if (!cont) while (cur < text.length && isSpace(text[cur])) cur++;
    if (tok === "[UNK]") {
      // BERT emits [UNK] for a whole basic-tokenizer word: up to the next space or punctuation (or one punctuation char)
      const start = cur;
      if (cur < text.length && isPunct(text[cur])) cur++;
      else while (cur < text.length && !isSpace(text[cur]) && !isPunct(text[cur])) cur++;
      out.push(cur > start ? [start, cur] : null);
      continue;
    }
    const piece = cont ? tok.slice(2) : tok;
    if (text.startsWith(piece, cur)) {
      out.push([cur, cur + piece.length]);
      cur += piece.length;
      continue;
    }
    // tolerate small normalisation differences (case, accents) by a case-insensitive search close to the cursor
    const idx = text.toLowerCase().indexOf(piece.toLowerCase(), cur);
    if (idx >= 0 && idx - cur <= 8) {
      out.push([idx, idx + piece.length]);
      cur = idx + piece.length;
    } else {
      out.push(null);   // could not align: this token cannot be masked (reported, not silently guessed)
    }
  }
  return out;
}

/** labels[i] is the argmax label of tokens[i] (e.g. "B-PER", "I-PER", "O", "B-LOC"). */
export function personSpans(text: string, tokens: string[], labels: string[], offset = 0): PiiSpan[] {
  const offs = alignTokens(text, tokens);
  // words: a non-## token plus its ## continuations; the word takes its first sub-token's label
  const words: { start: number; end: number; label: string }[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const o = offs[i];
    if (!o) continue;
    if (tokens[i].startsWith("##") && words.length) {
      words[words.length - 1].end = o[1];
      continue;
    }
    words.push({ start: o[0], end: o[1], label: labels[i] });
  }
  const spans: { start: number; end: number }[] = [];
  let open: { start: number; end: number } | null = null;
  for (const w of words) {
    const tag = w.label.replace(/^[BI]-/, "");
    const isB = w.label.startsWith("B-");
    if (tag !== "PER") {
      open = null;
      continue;
    }
    if (open && !isB) open.end = w.end;
    else {
      open = { start: w.start, end: w.end };
      spans.push(open);
    }
  }
  return spans.map((s) => ({
    start: s.start + offset,
    end: s.end + offset,
    type: "NAME" as const,
    key: text.slice(s.start, s.end).toLowerCase().replace(/\s+/g, " "),
  }));
}

/** Split at whitespace into chunks of at most maxChars (a single longer "word" becomes its own chunk). */
export function chunkText(text: string, maxChars = 600): { text: string; offset: number }[] {
  const chunks: { text: string; offset: number }[] = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(text.length, start + maxChars);
    if (end < text.length) {
      const cut = text.lastIndexOf(" ", end);
      const nl = text.lastIndexOf("\n", end);
      const best = Math.max(cut, nl);
      if (best > start) end = best;
    }
    chunks.push({ text: text.slice(start, end), offset: start });
    start = end;
  }
  return chunks;
}

// ---------------------------------------------------------------- evaluation helpers (docs/decisions.md Day 3.2)
export interface NameGold {
  id: number;
  category: string;
  text: string;
  names: string[];
}

export interface NameEval {
  gold: number;
  strict: number;           // every non-space character of the gold name covered
  overlap: number;          // any character covered
  falsePositives: number;   // predicted spans overlapping no gold name
  fpOnNoNameSentences: number;
  missed: { id: number; name: string }[];
  fps: { id: number; text: string }[];
  byCategory: Record<string, { gold: number; strict: number; fp: number }>;
}

/** Every occurrence of each gold name (case-sensitive) in the sentence. */
export function goldSpans(row: NameGold): { start: number; end: number; name: string }[] {
  const out: { start: number; end: number; name: string }[] = [];
  for (const name of new Set(row.names)) {
    let i = row.text.indexOf(name);
    while (i >= 0) {
      out.push({ start: i, end: i + name.length, name });
      i = row.text.indexOf(name, i + name.length);
    }
  }
  return out;
}

export function evaluateNames(rows: NameGold[], predicted: { start: number; end: number }[][]): NameEval {
  const r: NameEval = { gold: 0, strict: 0, overlap: 0, falsePositives: 0, fpOnNoNameSentences: 0, missed: [], fps: [], byCategory: {} };
  rows.forEach((row, k) => {
    const pred = predicted[k];
    const gold = goldSpans(row);
    const cat = (r.byCategory[row.category] ??= { gold: 0, strict: 0, fp: 0 });
    for (const g of gold) {
      r.gold++;
      cat.gold++;
      const covered = (i: number) => pred.some((p) => p.start <= i && i < p.end);
      let all = true;
      let any = false;
      for (let i = g.start; i < g.end; i++) {
        if (isSpace(row.text[i])) continue;
        if (covered(i)) any = true;
        else all = false;
      }
      if (all) {
        r.strict++;
        cat.strict++;
      } else r.missed.push({ id: row.id, name: g.name });
      if (any) r.overlap++;
    }
    for (const p of pred) {
      if (!gold.some((g) => p.start < g.end && g.start < p.end)) {
        r.falsePositives++;
        cat.fp++;
        if (row.names.length === 0) r.fpOnNoNameSentences++;
        r.fps.push({ id: row.id, text: row.text.slice(p.start, p.end) });
      }
    }
  });
  return r;
}
