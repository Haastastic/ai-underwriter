import AdverseActionNotice from "./AdverseActionNotice.jsx";
import ContributionsChart from "./ContributionsChart.jsx";
import DecisionBadge from "./DecisionBadge.jsx";
import ProbabilityBar from "./ProbabilityBar.jsx";
import { featureLabel, fmtDateTime, fmtValue, pct, signed } from "../format.js";

const APPLICATION_FIELDS = [
  "RevolvingUtilizationOfUnsecuredLines",
  "age",
  "NumberOfTime30-59DaysPastDueNotWorse",
  "DebtRatio",
  "MonthlyIncome",
  "NumberOfOpenCreditLinesAndLoans",
  "NumberOfTimes90DaysLate",
  "NumberRealEstateLoansOrLines",
  "NumberOfTime60-89DaysPastDueNotWorse",
  "NumberOfDependents",
];

export default function ReviewResult({ review }) {
  const { decision, explanation, adverse_action: adverseAction, application } = review;
  const p = decision.probability;

  return (
    <div>
      <div className="card">
        <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
          <DecisionBadge decision={decision.decision} />
          <strong style={{ fontSize: 18, fontVariantNumeric: "tabular-nums" }}>
            {pct(p, 2)}
          </strong>
          <span style={{ color: "var(--muted)", fontSize: 13 }}>P(serious delinquency)</span>
          <span className="spacer" style={{ flex: 1 }} />
          <span style={{ color: "var(--muted)", fontSize: 12.5 }}>
            model {review.model_version}
            {review.id != null && ` · record #${review.id}`}
            {review.created_at && ` · ${fmtDateTime(review.created_at)}`}
          </span>
        </div>

        <div className="stats" style={{ marginTop: 16 }}>
          <div className="stat">
            <div className="k">Probability of default</div>
            <div className="v hero">{pct(p, 2)}</div>
          </div>
          <div className="stat">
            <div className="k">Population base rate</div>
            <div className="v">{pct(explanation.base_rate)}</div>
            <div className="note">SHAP expected value</div>
          </div>
          <div className="stat">
            <div className="k">Model log-odds margin</div>
            <div className="v">{signed(explanation.logodds_margin)}</div>
            <div className="note">base {signed(explanation.base_value)} + contributions</div>
          </div>
          <div className="stat">
            <div className="k">Decision band</div>
            <div className="v" style={{ fontSize: 18 }}>
              <DecisionBadge decision={decision.decision} />
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Where this sits</h2>
        <ProbabilityBar probability={p} thresholds={decision.thresholds} />
      </div>

      <div className="card">
        <h2>Why — per-feature contributions</h2>
        <ContributionsChart
          contributions={explanation.contributions}
          baseRate={explanation.base_rate}
        />
      </div>

      {adverseAction && (
        <div className="card">
          <h2>Adverse-action notice</h2>
          <AdverseActionNotice adverseAction={adverseAction} />
        </div>
      )}

      {application && (
        <div className="card">
          <h2>Application as submitted</h2>
          <dl className="kv">
            {APPLICATION_FIELDS.map((f) => (
              <div key={f} style={{ display: "contents" }}>
                <dt>{featureLabel(f)}</dt>
                <dd>{fmtValue(application[f])}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  );
}
