import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../api.js";
import DecisionBadge from "../components/DecisionBadge.jsx";
import { fmtDateTime, pct } from "../format.js";

const FILTERS = [
  { key: "", label: "All" },
  { key: "approved", label: "Approved" },
  { key: "referred", label: "Referred" },
  { key: "denied", label: "Denied" },
];

export default function Queue() {
  const navigate = useNavigate();
  const [filter, setFilter] = useState("");
  const [state, setState] = useState({ status: "loading", rows: [] });

  useEffect(() => {
    const ctrl = new AbortController();
    setState((s) => ({ ...s, status: "loading" }));
    api
      .listApplications({ decision: filter || undefined }, { signal: ctrl.signal })
      .then((rows) => setState({ status: "ready", rows }))
      .catch((err) => {
        if (err.name !== "AbortError") setState({ status: "error", rows: [], error: err.message });
      });
    return () => ctrl.abort();
  }, [filter]);

  return (
    <div>
      <h1 className="page-title">Review queue</h1>
      <p className="page-sub">
        Every application scored through <code>/review</code>, newest first. The{" "}
        <strong>referred</strong> band is the one that needs a loan officer&rsquo;s judgement.
      </p>

      <div className="filters">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            className={filter === f.key ? "active" : ""}
            onClick={() => setFilter(f.key)}
          >
            {f.label}
          </button>
        ))}
      </div>

      {state.status === "error" && <div className="msg error">{state.error}</div>}
      {state.status === "loading" && <div className="loading">Loading…</div>}

      {state.status === "ready" && state.rows.length === 0 && (
        <div className="msg empty">
          No records{filter ? ` in the ${filter} band` : ""} yet. Run one from{" "}
          <strong>New review</strong>.
        </div>
      )}

      {state.status === "ready" && state.rows.length > 0 && (
        <div className="card" style={{ padding: 0, overflowX: "auto" }}>
          <table className="queue">
            <thead>
              <tr>
                <th>#</th>
                <th>Reviewed</th>
                <th>Decision</th>
                <th>P(default)</th>
                <th>Reasons</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {state.rows.map((r) => (
                <tr key={r.id} onClick={() => navigate(`/applications/${r.id}`)}>
                  <td className="num">{r.id}</td>
                  <td>{fmtDateTime(r.created_at)}</td>
                  <td>
                    <DecisionBadge decision={r.decision.decision} />
                  </td>
                  <td className="num">{pct(r.decision.probability, 2)}</td>
                  <td className="num">
                    {r.adverse_action ? r.adverse_action.reason_features.length : "—"}
                  </td>
                  <td>{r.model_version}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
