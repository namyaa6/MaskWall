/**
 * Links in the AI's reply are checked AFTER the placeholders were filled back in on the device: if a link's address contains one of the
 * user's real values, the browser would send that value to whoever owns the link when it is clicked (the same exfiltration trick as an
 * image, which is already blocked). Such a link is shown as plain text instead.
 *
 * Matching ignores URL-encoding, case and separators, so `?d=rahul.v%40example.in` and `?d=98765-43210` are caught. Values of 4 or more
 * letters/digits match as a substring; shorter ones (a 3-letter name) only as a whole token, to avoid removing harmless links.
 */
const squash = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
const tokens = (s: string) => s.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean);

function decodeAll(s: string): string[] {
  const out = [s];
  let cur = s;
  for (let i = 0; i < 2; i++) {
    try {
      const d = decodeURIComponent(cur.replace(/\+/g, " "));
      if (d === cur) break;
      out.push(d);
      cur = d;
    } catch {
      break;
    }
  }
  return out;
}

export function hrefContainsValue(href: string | undefined, values: readonly string[]): boolean {
  if (!href || values.length === 0) return false;
  const forms = decodeAll(href);
  const squashed = forms.map(squash);
  const toks = new Set(forms.flatMap(tokens));
  for (const v of values) {
    const sv = squash(v);
    if (!sv) continue;
    if (sv.length >= 4 ? squashed.some((f) => f.includes(sv)) : tokens(v).every((t) => toks.has(t))) return true;
  }
  return false;
}
