import { useState } from "react";

import { featureLabel, fmtValue, pct, signed } from "../format.js";

// Local ("force plot"-style) SHAP explanation as a diverging bar chart:
// polarity is the data's job, so two hues + a neutral centre axis. Red bars
// push the model toward higher risk, blue toward lower. Bars are pre-sorted
// by magnitude upstream; we show the most influential `topN`.
const W = 720;
const LABEL_W = 210;
const PAD_RIGHT = 52;
const ROW_H = 30;
const BAR_H = 16;
const TOP_PAD = 8;
const BOTTOM_PAD = 8;

const PLOT_W = W - LABEL_W - PAD_RIGHT;
const CENTER_X = LABEL_W + PLOT_W / 2;
const HALF_W = PLOT_W / 2 - 6;

function endRoundedPath(cx, y, len, h, side) {
  const r = Math.max(0, Math.min(4, len, h / 2));
  if (side === "right") {
    const xe = cx + len;
    return `M${cx},${y} H${xe - r} A${r},${r} 0 0 1 ${xe},${y + r} V${y + h - r} A${r},${r} 0 0 1 ${xe - r},${y + h} H${cx} Z`;
  }
  const xe = cx - len;
  return `M${cx},${y} H${xe + r} A${r},${r} 0 0 0 ${xe},${y + r} V${y + h - r} A${r},${r} 0 0 0 ${xe + r},${y + h} H${cx} Z`;
}

export default function ContributionsChart({ contributions = [], baseRate, topN = 12 }) {
  const [hover, setHover] = useState(null);

  if (!contributions.length) {
    return <p className="shap-caption">No contributions to display.</p>;
  }

  const rows = contributions.slice(0, topN);
  const maxAbs = Math.max(...rows.map((c) => Math.abs(c.shap_value))) || 1;
  const height = TOP_PAD + rows.length * ROW_H + BOTTOM_PAD;

  return (
    <div>
      <div className="shap-legend">
        <span className="key">
          <span className="swatch up" /> Increases risk
        </span>
        <span className="key">
          <span className="swatch down" /> Decreases risk
        </span>
      </div>

      <div className="shap-wrap">
        <svg
          className="shap-svg"
          viewBox={`0 0 ${W} ${height}`}
          role="img"
          aria-label="Per-feature SHAP contributions for this application"
          onMouseLeave={() => setHover(null)}
        >
          <line
            className="axis"
            x1={CENTER_X}
            x2={CENTER_X}
            y1={TOP_PAD}
            y2={height - BOTTOM_PAD}
          />

          {rows.map((c, i) => {
            const up = c.shap_value >= 0;
            const len = (Math.abs(c.shap_value) / maxAbs) * HALF_W;
            const barY = TOP_PAD + i * ROW_H + (ROW_H - BAR_H) / 2;
            const midY = TOP_PAD + i * ROW_H + ROW_H / 2;
            const valX = up ? CENTER_X + len + 6 : CENTER_X - len - 6;
            return (
              <g
                key={c.feature}
                className={`row${hover?.i === i ? " hovered" : ""}`}
                onMouseMove={(e) => {
                  const box = e.currentTarget.ownerSVGElement.getBoundingClientRect();
                  setHover({
                    i,
                    x: e.clientX - box.left,
                    y: e.clientY - box.top,
                    c,
                  });
                }}
              >
                <text className="row-label" x={LABEL_W - 12} y={midY + 4} textAnchor="end">
                  {featureLabel(c.feature)}
                </text>
                <path
                  className="barmark"
                  d={endRoundedPath(CENTER_X, barY, Math.max(len, 0.5), BAR_H, up ? "right" : "left")}
                  fill={up ? "var(--risk-up)" : "var(--risk-down)"}
                />
                <text
                  className="row-value"
                  x={valX}
                  y={midY + 4}
                  textAnchor={up ? "start" : "end"}
                >
                  {signed(c.shap_value)}
                </text>
                <rect className="hit" x={0} y={TOP_PAD + i * ROW_H} width={W} height={ROW_H} />
              </g>
            );
          })}
        </svg>

        {hover && (
          <div
            className="chart-tooltip"
            style={{
              left: Math.min(hover.x + 14, W - 180),
              top: Math.max(hover.y - 10, 0),
            }}
          >
            <div className="t-feat">{featureLabel(hover.c.feature)}</div>
            <div className="t-row">
              <span>Applicant value</span>
              <span>{fmtValue(hover.c.value)}</span>
            </div>
            <div className="t-row">
              <span>Contribution</span>
              <span>{signed(hover.c.shap_value, 3)} log-odds</span>
            </div>
            <div className="t-row">
              <span>Effect</span>
              <span>{hover.c.shap_value >= 0 ? "increases risk" : "decreases risk"}</span>
            </div>
          </div>
        )}
      </div>

      <p className="shap-caption">
        Top {rows.length} of {contributions.length} features by influence. Each bar is that
        factor&rsquo;s contribution to the model&rsquo;s log-odds score, relative to the population
        base rate of {pct(baseRate)}. The score is the sum of all {contributions.length}{" "}
        contributions — a longer bar moved this decision more.
      </p>
    </div>
  );
}
