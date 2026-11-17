import { useEffect, useState } from "react";
import { getHealth, type Health } from "./api";

export type HealthState = { state: "checking" } | { state: "online"; health: Health } | { state: "offline" };

/** Polls /health every 30 s. Used for the "Firewall online" indicator and the live thresholds. */
export function useHealth(): HealthState {
  const [h, setH] = useState<HealthState>({ state: "checking" });
  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    const check = () =>
      getHealth(ctrl.signal)
        .then((health) => alive && setH(health.model_loaded ? { state: "online", health } : { state: "offline" }))
        .catch(() => alive && setH({ state: "offline" }));
    check();
    const id = window.setInterval(check, 30_000);
    return () => {
      alive = false;
      ctrl.abort();
      window.clearInterval(id);
    };
  }, []);
  return h;
}
