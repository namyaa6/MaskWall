import { NavLink, Navigate, Route, Routes } from "react-router-dom";
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
  const cls = h.state === "online" ? "online" : h.state === "offline" ? "offline" : "";
  const label = h.state === "online" ? "Firewall online" : h.state === "offline" ? "Firewall offline" : "Checking firewall…";
  return (
    <span className={`status ${cls}`} role="status">
      <span className="status-dot" aria-hidden="true" />
      {label}
    </span>
  );
}

export function App() {
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
      <main id="main">
        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/playground" element={<PlaygroundPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/how-it-works" element={<HowItWorksPage />} />
          <Route path="/benchmarks" element={<BenchmarksPage />} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </main>
    </>
  );
}
