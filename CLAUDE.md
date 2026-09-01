# Project: Credit Underwriting Demo

## Purpose
A portfolio project demonstrating a real-world-adjacent fintech underwriting
pipeline: classical ML for the risk decision, an LLM for translating model
output into plain-language adverse-action rationale, and a loan-officer-facing
review UI. Built on public credit datasets (Kaggle "Give Me Some Credit" or
Lending Club). Target audience: ML engineering / fintech hiring reviewers, so
architecture and reasoning matter as much as the code working.

## Core architectural rule
**The LLM never makes or influences the credit risk decision.** The gradient-
boosted model produces the score/decision. SHAP explains *why* the model
produced it. The LLM's only job is to turn structured SHAP output into
readable adverse-action language. If a change would let the LLM affect the
score, stop and flag it — that breaks the whole point of the project.

## Architecture (in build order — see phase notes below)
1. **Data layer** (`src/data/`, `src/features/`) — load public dataset(s),
   clean, engineer features (ratios, binning, missing-value handling,
   encoding).
2. **Model layer** (`src/model/`) — train a gradient-boosted model
   (XGBoost or LightGBM), evaluate, persist artifacts to `models/`.
3. **Explainability layer** (`src/explain/`) — SHAP values per prediction,
   returned as structured data (not prose).
4. **LLM layer** (`src/llm/`) — pure function: structured SHAP output +
   application context in, plain-language ECOA-style adverse-action text
   out. No access to raw model internals, only the SHAP summary.
5. **App layer** (`app/backend/`, `app/frontend/`) — FastAPI (or Flask)
   backend exposing predict + explain endpoints; Streamlit or React frontend
   for loan officers to review flagged applications.

## Tech stack
- Data/ML: pandas, scikit-learn, XGBoost or LightGBM, SHAP
- LLM: Claude or GPT API (see `src/llm/`)
- Backend: FastAPI (preferred) or Flask
- Frontend: Streamlit (fastest path) or React (if UI polish matters more)
- Storage: SQLite
- Eval metrics: AUC-ROC, KS-statistic, probability calibration (Brier score
  or calibration curve) — every model version should report all three.

## Conventions
- Each layer (`src/data`, `src/features`, `src/model`, `src/explain`,
  `src/llm`) should be independently importable and independently testable.
  Don't let layers reach into each other's internals — pass plain dicts /
  dataframes across boundaries, not framework-specific objects, so any layer
  can be swapped (e.g. XGBoost → LightGBM, Claude → GPT) without touching
  the others.
- Model artifacts are versioned under `models/v1/`, `models/v2/`, etc., each
  with its own eval report (metrics + calibration plot) saved alongside the
  model file. Never overwrite a previous version's directory.
- Tests live in `tests/`, mirroring the `src/` structure
  (`tests/test_features.py`, `tests/test_model.py`, etc.). Data/feature and
  SHAP→LLM boundary tests are the highest priority — they're what let later
  phases (fairness audit, model swaps) proceed without silently breaking
  upstream work.
- Keep feature engineering interpretable. Prefer a smaller, explainable
  feature set over broad automated feature combinatorics — this demo's
  value is in the SHAP/explanation story, and an uninterpretable feature
  soup undermines that.

## Build phases (work one at a time; commit at the end of each)
1. Repo scaffold + this file (done).
2. Data layer: loading, cleaning, feature engineering, with unit tests on a
   small fixture of the dataset.
3. Model layer: train/val split, training script, eval report script
   (AUC/KS/calibration), artifact persistence.
4. SHAP explainability module: model + single application row → structured
   per-feature contributions.
5. LLM adverse-action layer: SHAP dict → plain-language ECOA-style text,
   as a pure, independently testable function.
6. Backend API wrapping the pipeline end-to-end.
7. Frontend for loan officers to review flagged applications.
8. Fairness audit extension (disparate impact ratios) + README section
   connecting design choices to ECOA adverse-action requirements.

Do not jump ahead to backend/frontend work until the pipeline runs correctly
end-to-end from the command line.

## Regulatory context (for reference, not implementation detail)
The adverse-action language should reflect the spirit of ECOA's requirement
that declined applicants receive specific, accurate reasons for the
decision — this is the framing that makes the LLM layer's design choice
(explanation only, never decisioning) meaningful rather than arbitrary.

## Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Train model (once src/model/train.py exists)
python -m src.model.train

# Run backend (once app/backend exists)
uvicorn app.backend.main:app --reload
```

## Things to ask before assuming
- Which dataset for this session: "Give Me Some Credit" or Lending Club?
  (They have different schemas — don't mix assumptions from one into code
  meant for the other.)
- Which GBM library: XGBoost or LightGBM? Pick one early and stay
  consistent — the model layer's interface should not change based on this
  choice.
- Streamlit vs Flask/React for the frontend — Streamlit is faster to demo,
  React is more portfolio-impressive. Confirm before scaffolding `app/`.
