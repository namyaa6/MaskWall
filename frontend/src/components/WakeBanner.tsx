import { useEffect, useState, useSyncExternalStore } from "react";
import { getServerState, subscribeServerState, wakeUp } from "../serverState";

export const WAKING_TEXT = "Waking up the firewall. This can take up to a minute or two.";

/** Shown under the top bar while the (free, sleeping) server is starting; replaced by a plain message if it never answers. */
export function WakeBanner() {
  const s = useSyncExternalStore(subscribeServerState, getServerState, getServerState);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (s.status !== "waking") return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [s.status]);

  if (s.status === "waking") {
    const secs = s.since ? Math.max(0, Math.round((now - s.since) / 1000)) : 0;
    return (
      <div className="wake-banner" role="status" aria-live="polite" data-testid="wake-banner">
        <p>
          <strong>{WAKING_TEXT}</strong> <span className="muted">({secs} s)</span>
        </p>
        <p className="muted small">
          The server is on free hosting that sleeps when nobody is using it. Anything you send now is held and goes out as soon as it is awake. Nothing is lost.
        </p>
      </div>
    );
  }
  if (s.status === "down") {
    return (
      <div className="wake-banner down" role="alert" data-testid="wake-banner-down">
        <p>
          <strong>The firewall is not answering.</strong> Free hosting sometimes takes longer than usual to start.
        </p>
        <p>
          <button type="button" className="btn" onClick={() => void wakeUp().catch(() => undefined)}>Try again</button>
        </p>
      </div>
    );
  }
  return null;
}
