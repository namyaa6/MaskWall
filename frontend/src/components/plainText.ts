/**
 * Markdown -> readable plain text, for the Chat "Copy" button.
 *
 * The input is the reply AFTER placeholders were swapped back (on the device), i.e. exactly what Reply.tsx renders. The output keeps
 * line breaks, list items and code lines readable and drops the Markdown syntax. It deliberately mirrors what Reply.tsx shows:
 * - images are never loaded, so they become "[image not loaded: alt]" (as on screen)
 * - a link whose address contains one of the user's own values is shown on screen as text plus a "link removed" note, so the address
 *   is NOT copied either (see linkSafety.ts)
 * - raw HTML is shown as text on screen (react-markdown escapes it), so it is left as typed
 * A small hand-written converter instead of a Markdown parser: it only has to produce readable text, and it adds no dependency.
 */
import { hrefContainsValue } from "./linkSafety";

const OPEN = "";
const CLOSE = "";

export function markdownToPlainText(markdown: string, sensitive: readonly string[] = []): string {
  const stash: string[] = [];
  const keep = (s: string) => `${OPEN}${stash.push(s) - 1}${CLOSE}`;

  // reference-style link definitions "[id]: url" are collected and removed from the text
  const defs = new Map<string, string>();
  const lines = markdown
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .filter((line) => {
      const m = /^ {0,3}\[([^\]]+)\]:\s*<?(\S+?)>?(?:\s+["'(].*)?\s*$/.exec(line);
      if (!m) return true;
      defs.set(m[1].toLowerCase(), m[2]);
      return false;
    });

  const link = (label: string, href: string | undefined): string => {
    if (hrefContainsValue(href, sensitive)) return `${label} [link removed: its address contained your personal data]`;
    if (!href || href === label) return label;
    return `${label} (${href})`;
  };

  const inline = (s: string): string => {
    s = s.replace(/\\([\\`*_{}[\]()#+\-.!|>~<])/g, (_, c: string) => keep(c));                 // escapes
    s = s.replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, (_, __, code: string) => keep(code.trim()));   // inline code
    s = s.replace(/!\[([^\]]*)\]\([^)]*\)/g, (_, alt: string) => keep(`[image not loaded${alt ? `: ${alt}` : ""}]`));
    s = s.replace(/!\[([^\]]*)\]\[[^\]]*\]/g, (_, alt: string) => keep(`[image not loaded${alt ? `: ${alt}` : ""}]`));
    s = s.replace(/\[([^\]]+)\]\(\s*<?([^)\s>]*)>?(?:\s+"[^"]*")?\s*\)/g, (_, label: string, href: string) => keep(link(label, href)));
    s = s.replace(/\[([^\]]+)\]\[([^\]]*)\]/g, (whole, label: string, id: string) => {
      const href = defs.get((id || label).toLowerCase());
      return href === undefined ? whole : keep(link(label, href));
    });
    s = s.replace(/<((?:https?:\/\/|mailto:)[^>\s]+)>/gi, (_, url: string) => keep(link(url, url)));
    s = s.replace(/\*\*([^\s*](?:[^*]*[^\s*])?)\*\*/g, "$1");
    s = s.replace(/__([^\s_](?:[^_]*[^\s_])?)__(?!\w)/g, "$1");
    s = s.replace(/\*([^\s*](?:[^*]*[^\s*])?)\*/g, "$1");
    s = s.replace(/(^|[^\w])_([^\s_](?:[^_]*[^\s_])?)_(?!\w)/g, "$1$2");                      // snake_case_words stay as they are
    s = s.replace(/~~([^\s~](?:[^~]*[^\s~])?)~~/g, "$1");
    return s;
  };

  const out: string[] = [];
  let fence: string | null = null;
  for (const raw of lines) {
    const f = /^ {0,3}(`{3,}|~{3,})/.exec(raw);
    if (fence !== null) {
      if (f && f[1][0] === fence[0] && f[1].length >= fence.length && /^\s*$/.test(raw.slice(f[0].length))) fence = null;
      else out.push(raw);                                                                      // code lines are copied as written
      continue;
    }
    if (f) {
      fence = f[1];
      continue;
    }
    let line = raw;
    if (/^ {0,3}([-*_])( ?\1){2,}\s*$/.test(line)) {                                             // horizontal rule
      out.push("");
      continue;
    }
    if (/^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line) && line.includes("-") && /[|]/.test(line)) continue;   // table separator row
    line = line.replace(/^ {0,3}#{1,6}\s+(.*?)\s*#*\s*$/, "$1");                                // headings
    line = line.replace(/^(\s*)(?:>\s?)+/, "$1");                                                // block quotes
    line = line.replace(/^(\s*)[*+-]\s+(?:\[[ xX]\]\s+)?/, "$1- ");                              // bullets (task boxes dropped), one style
    if (/^\s*\|.*\|\s*$/.test(line)) line = line.trim().slice(1, -1).split("|").map((c) => c.trim()).join(" | ");   // table row
    out.push(inline(line));
  }

  return out
    .join("\n")
    .replace(new RegExp(`${OPEN}(\\d+)${CLOSE}`, "g"), (_, i: string) => stash[Number(i)])
    .replace(/[ \t]+$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}
