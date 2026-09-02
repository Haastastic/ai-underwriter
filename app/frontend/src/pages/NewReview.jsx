import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../api.js";
import ApplicationForm from "../components/ApplicationForm.jsx";

export default function NewReview() {
  const navigate = useNavigate();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (payload) => {
    setSubmitting(true);
    setError(null);
    try {
      const review = await api.review(payload);
      navigate(`/applications/${review.id}`);
    } catch (err) {
      setError(err.message);
      setSubmitting(false);
    }
  };

  return (
    <div>
      <h1 className="page-title">New review</h1>
      <p className="page-sub">
        Enter the ten raw application fields. The gradient-boosted model produces the score and
        decision; SHAP explains it; a denial also gets an ECOA-style notice.
      </p>

      {error && <div className="msg error">{error}</div>}

      <div className="card">
        <ApplicationForm onSubmit={handleSubmit} submitting={submitting} />
      </div>
    </div>
  );
}
