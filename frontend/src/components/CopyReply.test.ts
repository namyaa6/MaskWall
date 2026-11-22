/**
 * The Chat "Copy" button: Markdown -> plain text, that restored (unmasked) values are what gets copied, the clipboard helper with its
 * fallback, and the button's accessibility markup.
 */
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PlaceholderVault } from "../pii/vault";
import { copyText } from "./clipboard";
import { CopyReplyButton, FAILED_TEXT } from "./CopyReplyButton";
import { markdownToPlainText as plain } from "./plainText";

describe("markdownToPlainText", () => {
  it("removes emphasis, headings and inline code marks but keeps the words", () => {
    expect(plain("# Title\n\nThis is **bold**, *italic*, _also italic_, ~~gone~~ and `code`.")).toBe("Title\n\nThis is bold, italic, also italic, gone and code.");
  });

  it("keeps list items and line breaks readable, with one bullet style and numbers kept", () => {
    const md = "Steps:\n\n* one\n+ two\n  - nested\n\n1. first\n2. second";
    expect(plain(md)).toBe("Steps:\n\n- one\n- two\n  - nested\n\n1. first\n2. second");
  });

  it("copies code blocks as written, without fences or inline processing", () => {
    expect(plain("Run:\n\n```bash\nls *.txt\nmy_var_name = 2 * 3 * 4\n```\n\nDone.")).toBe("Run:\n\nls *.txt\nmy_var_name = 2 * 3 * 4\n\nDone.");
  });

  it("does not damage snake_case words or spaced asterisks", () => {
    expect(plain("Use snake_case_names and 2 * 3 * 4.")).toBe("Use snake_case_names and 2 * 3 * 4.");
  });

  it("shows a link as 'text (address)', and a bare link once", () => {
    expect(plain("See [Paris](https://en.wikipedia.org/wiki/Paris) or <https://example.com>.")).toBe(
      "See Paris (https://en.wikipedia.org/wiki/Paris) or https://example.com.",
    );
    expect(plain("[x][1]\n\n[1]: https://example.com/a")).toBe("x (https://example.com/a)");
  });

  it("copies what the screen shows for images and for links that carry the user's values", () => {
    expect(plain("![chart](https://evil.example/p.png)")).toBe("[image not loaded: chart]");
    const out = plain("[click](https://evil.example/?d=rahul.v@example.in)", ["rahul.v@example.in"]);
    expect(out).toContain("click");
    expect(out).toContain("link removed");
    expect(out).not.toContain("evil.example");
  });

  it("turns quotes and tables into plain lines, and drops rules", () => {
    expect(plain("> quoted\n\n---\n\n| a | b |\n|---|---|\n| 1 | 2 |")).toBe("quoted\n\na | b\n1 | 2");
  });

  it("unescapes escaped characters", () => {
    expect(plain("\\*not italic\\* and 1\\. not a list")).toBe("*not italic* and 1. not a list");
  });
});

describe("what is copied is the unmasked reply", () => {
  it("has no [EMAIL_1]-style placeholders left once the vault has unmasked the reply", () => {
    const vault = new PlaceholderVault();
    vault.mask("Mail rahul.v@example.in or call 98765 43210", []);
    const shown = vault.unmask("I will write to **[EMAIL_1]** and call [PHONE_1].\n\n- Mail: [EMAIL_1]\n- Phone: [PHONE_1]");
    const copied = plain(shown, vault.values());
    expect(copied).toBe("I will write to rahul.v@example.in and call 98765 43210.\n\n- Mail: rahul.v@example.in\n- Phone: 98765 43210");
    expect(copied).not.toMatch(/\[(EMAIL|PHONE|NAME|AADHAAR|PAN)_\d+\]/);
  });
});

describe("copyText", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("uses the Clipboard API when it works", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    expect(await copyText("hello")).toBe(true);
    expect(writeText).toHaveBeenCalledWith("hello");
  });

  it("falls back to a hidden textarea + execCommand when the Clipboard API is refused, then removes the textarea", async () => {
    vi.stubGlobal("navigator", { clipboard: { writeText: vi.fn().mockRejectedValue(new Error("denied")) } });
    const box = { value: "", style: {} as Record<string, string>, setAttribute: vi.fn(), select: vi.fn(), setSelectionRange: vi.fn() };
    const body = { appendChild: vi.fn(), removeChild: vi.fn() };
    const execCommand = vi.fn().mockReturnValue(true);
    vi.stubGlobal("document", { body, activeElement: null, createElement: () => box, execCommand });
    expect(await copyText("hello")).toBe(true);
    expect(box.value).toBe("hello");
    expect(execCommand).toHaveBeenCalledWith("copy");
    expect(body.removeChild).toHaveBeenCalledWith(box);
  });

  it("returns false (does not throw) when both ways fail", async () => {
    vi.stubGlobal("navigator", {});
    const box = { value: "", style: {} as Record<string, string>, setAttribute: vi.fn(), select: vi.fn(), setSelectionRange: vi.fn() };
    vi.stubGlobal("document", { body: { appendChild: vi.fn(), removeChild: vi.fn() }, activeElement: null, createElement: () => box, execCommand: () => false });
    expect(await copyText("hello")).toBe(false);
  });
});

describe("CopyReplyButton markup", () => {
  const html = renderToStaticMarkup(createElement(CopyReplyButton, { text: "hi" }));

  it("is a real button named 'Copy reply' with a polite status region", () => {
    expect(html).toMatch(/<button type="button"[^>]*aria-label="Copy reply"/);
    expect(html).toContain('role="status"');
    expect(html).toContain(">Copy</button>");
  });

  it("uses the agreed failure wording", () => {
    expect(FAILED_TEXT).toBe("Couldn’t copy, select the text instead");
  });
});
