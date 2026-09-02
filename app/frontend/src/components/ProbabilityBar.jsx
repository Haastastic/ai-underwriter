import { pct } from "../format.js";

// Places P(serious delinquency) against the two policy cutoffs so an officer
// working a referred case can see at a glance how close it sits to each line.
export default function ProbabilityBar({ probability, thresholds }) {
  const approveBelow = thresholds?.approve_below ?? 0.08;
  const denyAtOrAbove = thresholds?.deny_at_or_above ?? 0.28;

  const maxScale = Math.min(
    1,
    Math.max(0.5, probability * 1.3, denyAtOrAbove * 1.5),
  );
  const x = (v) => `${Math.min(100, (v / maxScale) * 100)}%`;

  return (
    <div className="probbar">
      <div className="track" role="img" aria-label={`Probability of default ${pct(probability, 2)}`}>
        <div className="zone good" style={{ width: x(approveBelow) }} />
        <div
          className="zone warning"
          style={{ width: `calc(${x(denyAtOrAbove)} - ${x(approveBelow)})` }}
        />
        <div className="zone critical" style={{ flex: 1 }} />
        <div className="marker" style={{ left: x(probability) }} />
      </div>
      <div className="scale">
        <span>0%</span>
        <span>{pct(maxScale, 0)}</span>
      </div>
      <div className="legend">
        <span>
          Approve <b>&lt; {pct(approveBelow, 0)}</b>
        </span>
        <span>
          Refer <b>{pct(approveBelow, 0)}–{pct(denyAtOrAbove, 0)}</b>
        </span>
        <span>
          Deny <b>≥ {pct(denyAtOrAbove, 0)}</b>
        </span>
        <span>
          This application <b>{pct(probability, 2)}</b>
        </span>
      </div>
    </div>
  );
}
