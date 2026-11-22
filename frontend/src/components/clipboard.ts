/**
 * Copy text to the clipboard. Returns true on success, false if the browser refused (never throws).
 * First choice: the async Clipboard API (needs a secure context, i.e. https or localhost, and a user gesture).
 * Fallback: a hidden textarea + document.execCommand("copy"), which older browsers and some embedded views still allow.
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* permission denied or not a secure context: try the fallback */
  }
  return legacyCopy(text);
}

function legacyCopy(text: string): boolean {
  if (typeof document === "undefined" || !document.body) return false;
  const active = document.activeElement as HTMLElement | null;
  const box = document.createElement("textarea");
  box.value = text;
  box.setAttribute("readonly", "");
  box.setAttribute("aria-hidden", "true");
  box.style.position = "fixed";
  box.style.top = "0";
  box.style.left = "-9999px";
  document.body.appendChild(box);
  try {
    box.select();
    box.setSelectionRange(0, text.length);
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    document.body.removeChild(box);
    active?.focus?.();                       // give keyboard focus back to the Copy button
  }
}
