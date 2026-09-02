import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api.js";
import ReviewResult from "../components/ReviewResult.jsx";

export default function ReviewDetail() {
  const { id } = useParams();
  const [state, setState] = useState({ status: "loading" });

  useEffect(() => {
    const ctrl = new AbortController();
    setState({ status: "loading" });
    api
      .getApplication(id, { signal: ctrl.signal })
      .then((review) => setState({ status: "ready", review }))
      .catch((err) => {
        if (err.name !== "AbortError") setState({ status: "error", error: err.message });
      });
    return () => ctrl.abort();
  }, [id]);

  return (
    <div>
      <Link to="/queue" className="back-link">
        ← Back to queue
      </Link>
      <h1 className="page-title">Review #{id}</h1>

      {state.status === "loading" && <div className="loading">Loading…</div>}
      {state.status === "error" && <div className="msg error">{state.error}</div>}
      {state.status === "ready" && <ReviewResult review={state.review} />}
    </div>
  );
}
