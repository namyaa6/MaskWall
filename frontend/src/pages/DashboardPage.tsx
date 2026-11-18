import { useEffect, useState } from "react";
import { describeError, getStats, type Stats } from "../api";
import { PageHead } from "../components/PageHead";
import { ActionPill } from "../components/Pills";
import { actionLabel, fmtMs, fmtScore, mainReason, plural } from "../explain";
import { useHealth } from "../useHealth";

const REFRESH_MS = 10_000;

function ago(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="card tile">
      <p className="tile-label">{label}</p>
      <p className="tile-value">{value}</p>
      {note && <p className="muted small tile-note">{note}</p>}
    </div>
  );
}

export function DashboardPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updated, setUpdated] = useState<Date | null>(null);

  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    const load = () => {
      if (document.visibilityState !== "visible") return;     // no polling in a background tab
      getStats(ctrl.signal)
        .then((s) => {
          if (!alive) return;
          setStats(s);
          setError(null);
          setUpdated(new Date());
        })
        .catch((e) => alive && setError(describeError(e)));
    };
    load();
    const id = window.setInterval(load, REFRESH_MS);
    document.addEventListener("visibilitychange", load);
    return () => {
      alive = false;
      ctrl.abort();
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", load);
    };
  }, []);

  const health = useHealth();
  const thr = health.state === "online" ? health.health.thresholds : undefined;
  const by = stats?.by_action ?? {};
  const added = stats?.latency_ms.added_ms;

  return (
    <>
      <PageHead title="Dashboard">
        What the firewall has done so far: how many messages it checked, blocked and let through, and how much time it added.
      </PageHead>
      <p className="card privacy-note">
        <strong>Message text is never saved.</strong> The firewall only records the result, the risk score, the reason, counts of personal data and
        timings, so this page can show what happened without anyone being able to read what was said.
      </p>

      {error && (
        <p className="card error-card" role="alert">
          {error}
        </p>
      )}

      {stats && (
        <>
          <div className="tiles">
            <Tile label="Messages checked" value={String(stats.total_requests)} note="Chat messages and Playground checks" />
            <Tile
              label="Blocked"
              value={String(stats.attacks_blocked + stats.outputs_blocked)}
              note={stats.outputs_blocked ? "Stopped before reaching the AI, plus replies withheld after it" : "Stopped before reaching the AI"}
            />
            <Tile
              label="Safe"
              value={String(by.allow ?? 0)}
              note={`Let through with no warning (chat messages were answered). ${plural(by.warn ?? 0, "more was", "more were")} suspicious.`}
            />
            <Tile
              label="Typical time added on the server"
              value={fmtMs(added?.p50)}
              note={added?.p95 != null ? `Median; 95% of messages under ${fmtMs(added.p95)}. The AI's own time and the work done on your device are not counted.` : "No messages yet."}
            />
          </div>

          <section className="card recent" aria-labelledby="recent-title">
            <div className="recent-head">
              <h2 id="recent-title">Recent messages</h2>
              {updated && <p className="muted small">Updated {updated.toLocaleTimeString()} · refreshes every 10 s</p>}
            </div>
            <p className="muted">
              Risk score: 0 = clearly harmless, 1 = clearly an attack.
              {thr && ` From ${thr.warn.toFixed(2)} a message is suspicious; from ${thr.block.toFixed(2)} it is blocked.`}
            </p>
            {stats.recent.length === 0 ? (
              <p className="muted">No messages yet. Send one from Chat or check one in the Attack playground.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <caption className="sr-only">The 20 most recent messages, newest first</caption>
                  <thead>
                    <tr>
                      <th scope="col">Result</th>
                      <th scope="col">Risk score</th>
                      <th scope="col">Reason</th>
                      <th scope="col">From</th>
                      <th scope="col">Time added (server)</th>
                      <th scope="col">When</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.recent.map((r, i) => (
                      <tr key={`${r.ts}-${i}`}>
                        <td>
                          <ActionPill action={r.action} label={actionLabel(r.action)} />
                        </td>
                        <td className="num">{r.score == null ? "–" : fmtScore(r.score)}</td>
                        <td>
                          {mainReason(r.action, r.reasons)}
                          {r.pii_server_side > 0 && <span className="block muted small">The server's backup hid {plural(r.pii_server_side, "personal detail")}.</span>}
                        </td>
                        <td>{r.endpoint === "/chat" ? "Chat" : r.endpoint === "/scan" ? "Playground" : r.endpoint}</td>
                        <td className="num">{fmtMs(r.total_ms == null ? null : Math.max(0, r.total_ms - (r.llm_ms ?? 0)))}</td>
                        <td>{ago(r.ts)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
      {!stats && !error && <p className="muted">Loading…</p>}
    </>
  );
}
