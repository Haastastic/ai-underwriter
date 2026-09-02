# Loan-officer review UI

React + Vite frontend for the underwriting pipeline (Phase 7). It is a thin
presentation layer over the FastAPI backend — it never scores anything itself.

## What it does

- **New review** — enter the ten raw application fields (or load a preset),
  `POST /review`, and land on the saved record.
- **Queue** — every reviewed application (`GET /applications`), filterable by
  decision band. The `referred` band is the one meant for human judgement.
- **Review detail** (`GET /applications/{id}`) —
  - the decision band and `P(serious delinquency)`,
  - a probability bar placing the score against the two policy cutoffs,
  - a diverging SHAP bar chart of the per-feature contributions (red pushes
    toward higher risk, blue toward lower),
  - the ECOA-style adverse-action notice for denials, with the deterministic
    reason/feature audit trail behind it,
  - the application as submitted.

## Run it

```bash
# 1. backend (from the repo root)
uvicorn app.backend.main:app --reload      # serves on :8000

# 2. frontend (from app/frontend/)
npm install
npm run dev                                # serves on :5173
```

The dev server proxies `/api/*` to `http://localhost:8000` (override with
`VITE_API_TARGET`), so no CORS round-trip is needed in development.

Without an `ANTHROPIC_API_KEY` the backend uses its offline stub; the header
shows `LLM: stub` and notices render with a **Draft** flag.

## Production build

```bash
VITE_API_BASE_URL=https://api.example.com npm run build   # -> dist/
```

A separately hosted build calls the API cross-origin, so add its origin to the
backend's `AIU_CORS_ORIGINS`.

## Layout

```
src/
  api.js                 fetch client + ApiError
  format.js              feature labels, number/date formatting
  App.jsx                header, health badge, routes
  pages/
    Queue.jsx            GET /applications + band filter
    NewReview.jsx        application form -> POST /review
    ReviewDetail.jsx     GET /applications/:id
  components/
    ApplicationForm.jsx  the ten fields, presets, client-side validation
    ReviewResult.jsx     full review presentation (shared)
    DecisionBadge.jsx    approved / referred / denied pill
    ProbabilityBar.jsx   score vs the approve/deny cutoffs
    ContributionsChart.jsx  diverging SHAP bar chart (hand-rolled SVG)
    AdverseActionNotice.jsx notice text + reason audit trail
```

Chart colours follow the repo's data-viz palette: a validated blue↔red
diverging pair for risk polarity, the status palette for the decision bands.

## Not in this phase

Recording the officer's own approve/decline decision on a referred case needs
a backend write endpoint and a store column; it is a natural follow-up.
