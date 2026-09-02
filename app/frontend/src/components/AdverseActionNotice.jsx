import { featureLabel, signed } from "../format.js";

export default function AdverseActionNotice({ adverseAction }) {
  const aa = adverseAction;
  if (!aa) return null;

  const isDraft = typeof aa.notice_text === "string" && aa.notice_text.startsWith("[DRAFT");
  const statements = aa.reason_statements || [];
  const features = aa.reason_features || [];
  const shap = aa.reason_shap_values || [];

  return (
    <div className="notice">
      {isDraft && (
        <span className="draft-flag">
          Draft — generated without a language model ({aa.llm_provider})
        </span>
      )}
      <pre>{aa.notice_text}</pre>

      {statements.length > 0 && (
        <>
          <ol className="reason-list">
            {statements.map((s, i) => (
              <li key={features[i] ?? i}>
                {s}
                {features[i] && (
                  <>
                    {" "}
                    <span style={{ color: "var(--muted)" }}>
                      · {featureLabel(features[i])}
                      {shap[i] != null && ` (${signed(shap[i])})`}
                    </span>
                  </>
                )}
              </li>
            ))}
          </ol>
          <p className="shap-caption" style={{ marginTop: 8 }}>
            Reasons are selected deterministically from SHAP rank in{" "}
            <code>src/llm/reasons.py</code>; age-derived factors are excluded (Regulation B).
            The model{aa.model ? ` (${aa.model})` : ""} only renders them as prose.
          </p>
        </>
      )}
    </div>
  );
}
