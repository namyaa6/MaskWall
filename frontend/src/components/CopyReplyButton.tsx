import { useEffect, useRef, useState } from "react";
import { copyText } from "./clipboard";
import { markdownToPlainText } from "./plainText";

export const COPIED_MS = 2000;
export const FAILED_MS = 5000;
export const FAILED_TEXT = "Couldn’t copy, select the text instead";

type State = "idle" | "copied" | "failed";

/**
 * "Copy" for one answered reply. Copies the reply as shown (placeholders already restored on the device) as plain text.
 * The button's name stays "Copy reply"; the result is announced through a status region (polite live region).
 */
export function CopyReplyButton({ text, sensitive = [] }: { text: string; sensitive?: readonly string[] }) {
  const [state, setState] = useState<State>("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);

  async function copy() {
    clearTimeout(timer.current);
    const ok = await copyText(markdownToPlainText(text, sensitive));
    setState(ok ? "copied" : "failed");
    timer.current = setTimeout(() => setState("idle"), ok ? COPIED_MS : FAILED_MS);
  }

  return (
    <div className="reply-actions">
      <span role="status" className={state === "failed" ? "reply-copy-note" : "sr-only"}>
        {state === "copied" ? "Reply copied to clipboard" : state === "failed" ? FAILED_TEXT : ""}
      </span>
      <button type="button" className="btn btn-small reply-copy" aria-label="Copy reply" data-state={state} onClick={copy}>
        {state === "copied" ? (
          <>
            Copied <span aria-hidden="true">✓</span>
          </>
        ) : (
          "Copy"
        )}
      </button>
    </div>
  );
}
