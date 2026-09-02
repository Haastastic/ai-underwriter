import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { api } from "./api.js";
import Queue from "./pages/Queue.jsx";
import NewReview from "./pages/NewReview.jsx";
import ReviewDetail from "./pages/ReviewDetail.jsx";

function HealthBadge() {
  const [state, setState] = useState({ status: "loading" });

  useEffect(() => {
    const ctrl = new AbortController();
    api
      .health({ signal: ctrl.signal })
      .then((h) => setState({ status: "ok", ...h }))
      .catch((err) => {
        if (err.name !== "AbortError") setState({ status: "down", error: err.message });
      });
    return () => ctrl.abort();
  }, []);

  if (state.status === "loading") {
    return <span className="health"><span className="dot" /> checking API…</span>;
  }
  if (state.status === "down") {
    return (
      <span className="health" title={state.error}>
        <span className="dot down" /> API unreachable
      </span>
    );
  }
  const stub = state.llm_provider === "stub";
  return (
    <span className="health">
      <span className="dot ok" /> model {state.model_version}
      <span className={`pill${stub ? " warn" : ""}`}>
        LLM: {state.llm_provider}
        {stub ? " · notices are drafts" : ""}
      </span>
    </span>
  );
}

export default function App() {
  return (
    <>
      <header className="app-header">
        <h1>
          AI Underwriter <span>· loan-officer review</span>
        </h1>
        <nav className="app-nav">
          <NavLink to="/queue" className={({ isActive }) => (isActive ? "active" : "")}>
            Queue
          </NavLink>
          <NavLink to="/new" className={({ isActive }) => (isActive ? "active" : "")}>
            New review
          </NavLink>
        </nav>
        <span className="spacer" />
        <HealthBadge />
      </header>

      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/queue" replace />} />
          <Route path="/queue" element={<Queue />} />
          <Route path="/new" element={<NewReview />} />
          <Route path="/applications/:id" element={<ReviewDetail />} />
          <Route path="*" element={<Navigate to="/queue" replace />} />
        </Routes>
      </main>
    </>
  );
}
