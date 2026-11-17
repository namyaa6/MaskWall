import type { ModelStatus } from "../workers/useModelWorker";

const mb = (bytes: number) => Math.round(bytes / 1e6);

/**
 * Download/loading state of an in-browser model. The first use downloads the model once (the browser caches it);
 * afterwards loading takes about a second.
 */
export function ModelStatusLine({ status, name, approxMB, readyText }: { status: ModelStatus; name: string; approxMB: number; readyText: string }) {
  if (status.state === "idle") return null;
  if (status.state === "error")
    return (
      <p className="model-status model-error" role="alert">
        The {name} did not load: {status.message}
      </p>
    );
  if (status.state === "ready")
    return (
      <p className="model-status model-ready" role="status">
        {readyText}
      </p>
    );
  const pct = status.total > 0 ? Math.min(100, Math.round((status.loaded / status.total) * 100)) : null;
  if (pct === 100)
    return (
      <p className="model-status" role="status">
        Downloaded. Starting the {name} on your device…
      </p>
    );
  return (
    <div className="model-status" role="status" aria-live="polite">
      <span>
        Downloading {name} (about {approxMB} MB, once){pct !== null ? `: ${mb(status.loaded)} of ${mb(status.total)} MB` : "…"}
      </span>
      <progress max={100} value={pct ?? undefined} aria-label={`Downloading ${name}`} />
    </div>
  );
}
