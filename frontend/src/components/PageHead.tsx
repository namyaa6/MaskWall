import type { ReactNode } from "react";

/** Every screen starts with a title and one plain-English sentence, so a first-time visitor knows what they are looking at. */
export function PageHead({ title, children }: { title: string; children: ReactNode }) {
  return (
    <header className="page-head">
      <h1>{title}</h1>
      <p>{children}</p>
    </header>
  );
}
