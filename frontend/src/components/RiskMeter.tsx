import { fmtScore } from "../explain";

/**
 * Risk score on a 0-1 bar with three zones whose boundaries are the server's live thresholds.
 * Zones are labelled in words (colour is never the only signal); the marker shows the score.
 */
export function RiskMeter({ score, warn, block }: { score: number; warn: number; block: number }) {
  const pct = (x: number) => `${Math.max(0, Math.min(1, x)) * 100}%`;
  const zone = score >= block ? "Attack · blocked" : score >= warn ? "Suspicious · sent with a warning" : "Safe · sent to the AI";
  return (
    <div className="meter">
      <div
        className="meter-bar"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={Number(score.toFixed(3))}
        aria-valuetext={`${fmtScore(score)} out of 1: ${zone}`}
        aria-label="Risk score"
      >
        <span className="zone zone-safe" style={{ width: pct(warn) }} />
        <span className="zone zone-suspicious" style={{ width: pct(block - warn) }} />
        <span className="zone zone-blocked" style={{ width: pct(1 - block) }} />
        <span className="meter-marker" style={{ left: pct(score) }} aria-hidden="true" />
      </div>
      <div className="meter-labels" aria-hidden="true">
        <span style={{ width: pct(warn) }}>Safe · sent to the AI</span>
        <span style={{ width: pct(block - warn) }}>Suspicious · sent with a warning</span>
        <span style={{ width: pct(1 - block) }}>Attack · blocked</span>
      </div>
      <p className="muted small meter-scale">
        0 = clearly harmless, 1 = clearly an attack. Boundaries at {warn.toFixed(2)} and {block.toFixed(2)}, tuned on validation data.
      </p>
    </div>
  );
}
