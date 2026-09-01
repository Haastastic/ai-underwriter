# Credit Underwriting Demo

A demonstration credit-risk underwriting pipeline: a gradient-boosted model
makes the risk decision, SHAP explains it, and an LLM translates that
explanation into plain-language, ECOA-style adverse-action rationale for a
loan-officer-facing review UI.

See `CLAUDE.md` for full architecture, conventions, and build-phase plan.

## Status
🚧 Data layer complete (loading, cleaning, feature engineering). Model, SHAP,
LLM, and app layers not started yet.

## Layout
```
data/            raw and processed datasets (gitignored)
src/data/        dataset loading
src/features/    feature engineering
src/model/       training, evaluation, artifact persistence
src/explain/     SHAP explainability
src/llm/         SHAP -> plain-language adverse-action text
app/backend/     API layer
app/frontend/    loan-officer review UI
models/          versioned trained model artifacts (gitignored)
notebooks/       exploratory work
tests/           unit tests, mirrors src/ structure
```

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
Download Kaggle's "Give Me Some Credit" (or Lending Club) data and place it
under `data/raw/`. Not committed to the repo.
