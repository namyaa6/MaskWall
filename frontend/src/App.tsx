import { useEffect, useSyncExternalStore } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { WakeBanner } from "./components/WakeBanner";
import { getServerState, subscribeServerState, wakeUp } from "./serverState";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { useHealth } from "./useHealth";
import { ChatPage } from "./pages/ChatPage";
import { PlaygroundPage } from "./pages/PlaygroundPage";
import { DashboardPage } from "./pages/DashboardPage";
import { HowItWorksPage } from "./pages/HowItWorksPage";
import { BenchmarksPage } from "./pages/BenchmarksPage";

const LINKS = [
  { to: "/chat", label: "Chat" },
  { to: "/playground", label: "Attack playground" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/how-it-works", label: "How it works" },
  { to: "/benchmarks", label: "Benchmarks" },
];

function FirewallStatus() {
  const h = useHealth();
  const waking = useSyncExternalStore(subscribeServerState, getServerState, getServerState).status === "waking";
  const cls = waking ? "" : h.state === "online" ? "online" : h.state === "offline" ? "offline" : "";
  const label = waking ? "Waking up…" : h.state === "online" ? "Firewall online" : h.state === "offline" ? "Firewall offline" : "Checking firewall…";
  return (
    <span className={`status ${cls}`} role="status">
      <span className="status-dot" aria-hidden="true" />
      {label}
    </span>
  );
}

/** Dev-only (not in production builds): a page that always crashes, so the e2e test can see the error boundary. */
function Crash(): never {
  throw new Error("deliberate test crash");
}

export function App() {
  const { pathname } = useLocation();
  // wake the (free, sleeping) server as soon as the site opens, so it is usually ready by the time someone has typed a message
  useEffect(() => {
    wakeUp().catch(() => undefined);
  }, []);
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <div className="topbar">
        <div className="topbar-inner">
          <NavLink to="/chat" className="brand">MaskWall</NavLink>
          <nav className="nav" aria-label="Main">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to}>{l.label}</NavLink>
            ))}
          </nav>
          <FirewallStatus />
        </div>
      </div>
      <WakeBanner />
      <main id="main">
        <ErrorBoundary resetKey={pathname}>
          <Routes>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/playground" element={<PlaygroundPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/how-it-works" element={<HowItWorksPage />} />
            <Route path="/benchmarks" element={<BenchmarksPage />} />
            {import.meta.env.DEV && <Route path="/__crash" element={<Crash />} />}
          <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </>
  );
}
