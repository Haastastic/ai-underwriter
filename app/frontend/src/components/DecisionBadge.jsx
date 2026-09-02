import { decisionMeta } from "../format.js";

const GLYPH = { good: "✓", warning: "◑", critical: "✕", muted: "•" };

export default function DecisionBadge({ decision }) {
  const { label, tone } = decisionMeta(decision);
  return (
    <span className={`badge ${tone}`}>
      <span className="glyph" aria-hidden="true">
        {GLYPH[tone]}
      </span>
      {label}
    </span>
  );
}
