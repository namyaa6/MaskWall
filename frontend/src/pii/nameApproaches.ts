/**
 * Ways to find lowercase / casual names ("pls tell priya") that the cased NER model misses.
 * Pre-registered in docs/decisions.md Day 3.12 and FROZEN before the lowercase evaluation set was written:
 * no word, cue or rule below may be changed after the first measurement.
 *
 *   A  plain NER on the text as typed (the baseline, Day 3.2)
 *   B  NER on a length-preserving copy in which every all-lowercase word that is not a common word gets a capital letter
 *   C  context cues ("tell X", "X ne", "my name is X"...) in addition to A
 *   E  B and C together
 *
 * "Common word" = the lowercase word is a whole token in the NER model's own cased vocabulary (so "rose", "deep", "joy" are
 * common while "priya", "rahul", "john" are not), or it is in HINGLISH_COMMON below, or it is chat shorthand.
 * Pure functions only: the model is passed in, so everything here is unit-testable without loading it.
 */
import type { PiiSpan } from "./detectors";

export type NameApproach = "A" | "B" | "C" | "E";

export interface NameModel {
  /** NER spans for the text exactly as given */
  detect(text: string): Promise<PiiSpan[]>;
  /** is this exact (lowercase) string a whole token of the model's vocabulary? */
  hasToken(word: string): boolean;
}

// ---------------------------------------------------------------- word lists (written once, generic, frozen)
/** Hinglish function words, common verbs/nouns and chat shorthand: never treated as names. */
export const HINGLISH_COMMON = new Set(
  (
    "hai hain ho hu hoon hota hoti hote hoga hogi honge tha thi the raha rahi rahe rahega rahegi " +
    "ka ki ke ko ne se me mein mai main par pe pey tak liye liya lie sath saath bina baad pehle pahle " +
    "aur ya lekin par magar toh to bhi hi nahi nahin na mat kya kyun kyon kaise kaisa kaisi kahan kab kaun kitna kitni kitne " +
    "yeh ye woh wo yahan wahan jab tab abhi ab phir fir kal aaj parso subah shaam raat dopahar " +
    "mera meri mere mujhe mujhko hum humko humein tum tumhe tumhara tumhari aap aapko aapka aapki aapke uska uski uske unka unki unke " +
    "iska iski iske apna apni apne sab sabhi koi kuch kuchh bahut bohot thoda thodi zyada jyada kam bas sirf " +
    "karo karna karta karti karte kar kiya kiye kare karenge karunga karungi kardo karde " +
    "bolo bol bola boli bole bolna bataya batao bata batana bhej bhejo bheja bheji bhejna likh likho likha likhna " +
    "dekh dekho dekha dekhna sun suno suna sunna chal chalo chala chalna ja jao jana gaya gayi gaye aa aao aana aaya aayi aaye " +
    "de do dena diya di diye le lo lena liya mil milo milna mila ruk ruko rukna dhundh dhoondh " +
    "chahiye chahta chahti chahte pata samajh samjha samjhao padh padho padhna khao khana peena " +
    "ghar kaam paisa paise paani chai kaafi accha acha achha theek thik sahi galat jaldi dhire " +
    "haan han ji haanji arre arey are oye yaar yar bhai bhaiya bhaiyya didi dii bhabhi uncle aunty aunt sir madam mam sahab saab " +
    "namaste namaskar shukriya dhanyavaad shukriya sorry thanks thank " +
    "pls plz thx tmrw tomo btw asap tbh imo idk lol omg haha hehe hmm ok okay k kk gonna wanna gotta " +
    "im ive id dont cant wont didnt doesnt isnt arent wasnt werent couldnt wouldnt shouldnt youre theyre " +
    "msg sms dm pm am"
  ).split(/\s+/),
);

/** Words that are cues, not names, even when they are not in the vocabulary. */
const TITLES = new Set(["mr", "mrs", "ms", "miss", "dr", "shri", "shree", "smt", "prof", "mister", "madam"]);
const HONORIFICS_AFTER = new Set(["ji", "bhai", "bhaiya", "bhaiyya", "didi", "di", "bhabhi", "uncle", "aunty", "auntie", "aunt", "sir", "sahab", "saab", "madam", "mam"]);
const VERB_BEFORE = new Set([
  "tell", "told", "call", "called", "ask", "asked", "text", "texted", "email", "emailed", "mail", "mailed", "message", "messaged", "msg",
  "meet", "met", "thank", "thanks", "remind", "inform", "informed", "invite", "invited", "ping", "pinged", "whatsapp", "forward", "bolo", "batao",
]);
const TO_AFTER = new Set(["talk", "speak", "reply", "write", "wrote", "send", "sent", "forward", "forwarded", "give", "gave", "say", "tell", "explain", "message", "mail", "msg"]);
const GREETING_BEFORE = new Set(["hi", "hey", "hello", "dear", "namaste", "namaskar", "hii", "heyy", "yo"]);
const POST_PARTICLES = new Set(["ne", "ko", "ka", "ki", "ke", "se", "ji", ...HONORIFICS_AFTER]);
const POST_ENGLISH = new Set(["said", "says", "asked", "wants", "needs", "replied", "called", "texted", "mentioned", "thinks", "told", "sent", "wrote"]);
const CONNECTORS = new Set(["and", "aur", "or", "&", ",", "with"]);

// ---------------------------------------------------------------- tokens
interface Tok {
  w: string;       // as written
  lw: string;      // lowercase
  start: number;
  end: number;
}

function tokenize(text: string): Tok[] {
  const out: Tok[] = [];
  for (const m of text.matchAll(/[A-Za-z]+/g)) {
    const start = m.index!;
    out.push({ w: m[0], lw: m[0].toLowerCase(), start, end: start + m[0].length });
  }
  return out;
}

/**
 * Words that are never read as names: the model's lowercase vocabulary plus EITHER the round-1 hand-written list (when `words` is
 * omitted: reproduces Day 3.12) OR `words`, a set built only from independent corpora (round 2, Day 3.13: see commonWords.ts).
 */
export type CommonWords = ReadonlySet<string>;
export function isCommonWord(lw: string, model: Pick<NameModel, "hasToken">, words?: CommonWords): boolean {
  return (words ? words.has(lw) : HINGLISH_COMMON.has(lw)) || model.hasToken(lw);
}

/** Characters around a token that mean it is part of an email address, URL, handle or number: never a name. */
function insideIdentifier(text: string, t: Tok): boolean {
  const before = text[t.start - 1];
  const after = text[t.end];
  return before === "@" || after === "@" || (before !== undefined && /[0-9_/]/.test(before)) || (after !== undefined && /[0-9_/]/.test(after));
}

function candidate(text: string, t: Tok, model: Pick<NameModel, "hasToken">, words?: CommonWords): boolean {
  if (t.w.length < 3) return false;
  if (insideIdentifier(text, t)) return false;
  const lw = t.lw;
  if (isCommonWord(lw, model, words)) return false;
  return !(TITLES.has(lw) || VERB_BEFORE.has(lw) || TO_AFTER.has(lw) || GREETING_BEFORE.has(lw) || POST_PARTICLES.has(lw) || POST_ENGLISH.has(lw) || CONNECTORS.has(lw));
}

// ---------------------------------------------------------------- B: selective re-casing
/**
 * Same length as the input (only ASCII letters change case), so a span found in the copy is valid in the original.
 * Words inside emails, URLs and handles are left alone. Words already capitalised are left alone.
 */
export function recase(text: string, model: Pick<NameModel, "hasToken">, words?: CommonWords): string {
  let out = "";
  let pos = 0;
  for (const chunk of text.matchAll(/\S+/g)) {
    const c = chunk[0];
    const base = chunk.index!;
    if (/@|:\/\/|^www\./i.test(c)) continue;
    for (const m of c.matchAll(/(?<![A-Za-z])[a-z]{3,}(?![A-Za-z])/g)) {
      const start = base + m.index!;
      if (isCommonWord(m[0], model, words)) continue;
      out += text.slice(pos, start) + m[0][0].toUpperCase() + m[0].slice(1);
      pos = start + m[0].length;
    }
  }
  return out + text.slice(pos);
}

// ---------------------------------------------------------------- C: context cues
function span(text: string, from: Tok, to: Tok): PiiSpan {
  return { start: from.start, end: to.end, type: "NAME", key: text.slice(from.start, to.end).toLowerCase().replace(/\s+/g, " ") };
}

/** Up to two candidate tokens in a row starting at index i, separated by plain spaces. */
function chainForward(text: string, toks: Tok[], i: number, model: Pick<NameModel, "hasToken">, words?: CommonWords): [number, number] | null {
  if (i >= toks.length || !candidate(text, toks[i], model, words)) return null;
  let j = i;
  if (j + 1 < toks.length && /^ +$/.test(text.slice(toks[j].end, toks[j + 1].start)) && candidate(text, toks[j + 1], model, words)) j++;
  return [i, j];
}

function chainBackward(text: string, toks: Tok[], i: number, model: Pick<NameModel, "hasToken">, words?: CommonWords): [number, number] | null {
  if (i < 0 || !candidate(text, toks[i], model, words)) return null;
  let j = i;
  if (j - 1 >= 0 && /^ +$/.test(text.slice(toks[j - 1].end, toks[j].start)) && candidate(text, toks[j - 1], model, words)) j--;
  return [j, i];
}

export function cueSpans(text: string, model: Pick<NameModel, "hasToken">, words?: CommonWords): PiiSpan[] {
  const toks = tokenize(text);
  const found: PiiSpan[] = [];
  const add = (r: [number, number] | null) => {
    if (r) found.push(span(text, toks[r[0]], toks[r[1]]));
    return r;
  };
  for (let i = 0; i < toks.length; i++) {
    const lw = toks[i].lw;
    const prev = i > 0 ? toks[i - 1].lw : "";
    // cue BEFORE the name: "tell X", "call X", "hi X", "mr X", "my name is X", "naam X", "talk to X"
    if (VERB_BEFORE.has(lw) || GREETING_BEFORE.has(lw) || TITLES.has(lw) || lw === "naam") add(chainForward(text, toks, i + 1, model, words));
    if (lw === "to" && TO_AFTER.has(prev)) add(chainForward(text, toks, i + 1, model, words));
    if (lw === "is" && prev === "name") add(chainForward(text, toks, i + 1, model, words));
    if ((lw === "am" && prev === "i") || (lw === "is" && prev === "this") || lw === "myself") add(chainForward(text, toks, i + 1, model, words));
    // cue AFTER the name: "X ne", "X ko", "X ji", "X bhaiya", "X said", "X's"
    if (POST_PARTICLES.has(lw) || POST_ENGLISH.has(lw)) add(chainBackward(text, toks, i - 1, model, words));
  }
  // lists: "X and Y", "X aur Y", "X, Y": a candidate right after a connector that follows an already-found name
  const spansByEnd = new Map(found.map((s) => [s.end, s]));
  for (let i = 1; i < toks.length - 1; i++) {
    const gap = text.slice(toks[i - 1].end, toks[i].start);
    if (CONNECTORS.has(toks[i].lw) && spansByEnd.has(toks[i - 1].end) && /^ *$/.test(gap)) add(chainForward(text, toks, i + 1, model, words));
  }
  return found;
}

// ---------------------------------------------------------------- combining
/** Merge overlapping spans and spans separated by a single space ("priya" + "sharma"), recomputing the key from the text. */
export function mergeNameSpans(text: string, spans: PiiSpan[]): PiiSpan[] {
  const sorted = [...spans].sort((a, b) => a.start - b.start || b.end - a.end);
  const out: PiiSpan[] = [];
  for (const s of sorted) {
    const last = out[out.length - 1];
    if (last && (s.start < last.end || (s.start === last.end + 1 && text[last.end] === " "))) {
      last.end = Math.max(last.end, s.end);
      continue;
    }
    out.push({ ...s });
  }
  return out.map((s) => ({ ...s, key: text.slice(s.start, s.end).toLowerCase().replace(/\s+/g, " ") }));
}

/** Name spans for `text` with the chosen approach. Offsets always refer to `text`. */
export async function detectNames(approach: NameApproach, text: string, model: NameModel, words?: CommonWords): Promise<PiiSpan[]> {
  const useRecase = approach === "B" || approach === "E";
  const useCues = approach === "C" || approach === "E";
  const ner = await model.detect(useRecase ? recase(text, model, words) : text);
  if (!useCues) return useRecase ? ner.map((s) => ({ ...s, key: text.slice(s.start, s.end).toLowerCase().replace(/\s+/g, " ") })) : ner;
  return mergeNameSpans(text, [...ner, ...cueSpans(text, model, words)]);
}
