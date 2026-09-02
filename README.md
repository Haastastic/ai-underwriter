# Credit Underwriting Demo

A demonstration credit-risk underwriting pipeline: a gradient-boosted model
makes the risk decision, SHAP explains it, and an LLM translates that
explanation into plain-language, ECOA-style adverse-action rationale for a
loan-officer-facing review UI.

See `CLAUDE.md` for full architecture, conventions, and build-phase plan.

## Status
Pipeline runs end to end from the command line and over HTTP.

- ✅ Data layer — loading, cleaning, feature engineering
- ✅ Model layer — XGBoost training, eval report (AUC / KS / calibration), versioned artifacts
- ✅ Explainability — per-application SHAP contributions as structured data
- ✅ LLM layer — SHAP dict → ECOA-style adverse-action text (Claude, `claude-haiku-4-5` by default)
- ✅ Backend — FastAPI wrapping the whole pipeline, SQLite audit log
- ✅ Frontend — React + Vite loan-officer review UI (`app/frontend/`)
- ✅ Fairness audit — age-band selection/denial rates, disparate-impact ratios, four-fifths rule (`src/fairness/`)
- ✅ Model v2 (**default**) — less-discriminatory alternative: age and every age-derived feature removed, re-tuned, re-audited ([comparison](#model-v2--the-less-discriminatory-alternative))

## Architecture

```mermaid
flowchart TB
    CSV[("cs-training.csv")]
    REQ["application<br/>10 raw fields"]

    subgraph TRAIN["training · offline"]
      D1["src/data + src/features<br/>clean · engineer"]
      D2["src/model/train<br/>XGBoost + early stopping"]
      D1 --> D2 --> ART[("models/v2/<br/>model · clean stats<br/>eval report · calibration")]
    end
    CSV --> D1

    subgraph SERVE["serving · per application"]
      PREP["src/features/prepare<br/>single-row clean + align to model features"]

      subgraph DEC["risk decision — no LLM"]
        M["src/model<br/>predict_proba → P(default)"]
        P["src/model/decision<br/>two cutoffs → approved / referred / denied"]
        M --> P
      end

      subgraph EXP["explanation only"]
        S["src/explain<br/>SHAP → per-feature contributions"]
        R["src/llm/reasons<br/>pick top factors · exclude age (Reg B)"]
        PR["src/llm/prompt<br/>build ECOA-style prompt"]
        L["Claude · claude-haiku-4-5<br/>render adverse-action notice"]
        S --> R --> PR --> L
      end

      PREP --> M
      PREP --> S
      P -->|"denied only"| R
    end

    REQ --> PREP
    ART -. loaded at startup .-> M
    ART -. loaded at startup .-> PREP

    subgraph API["app/backend · FastAPI"]
      O["service orchestration"] --> DB[("SQLite<br/>review audit log")]
    end

    P --> O
    S --> O
    L --> O
    O --> OUT["JSON response<br/>decision + explanation + notice"]
    OUT --> FE["app/frontend<br/>React + Vite · loan-officer review UI"]

    subgraph AUDIT["fairness audit · offline · measures outcomes only"]
      FA["src/fairness/report<br/>score dataset · band by age<br/>approval/denial rates · disparate-impact ratios<br/>four-fifths rule"]
      FA --> FR[("models/v2/<br/>fairness_audit.json")]
    end
    CSV --> FA
    ART -. loaded .-> FA

    style DEC stroke:#2e7d32,stroke-width:2px
    style EXP stroke:#1565c0,stroke-width:2px
    style AUDIT stroke:#6a1b9a,stroke-width:2px
```

The gradient-boosted model produces the score **and** the decision; SHAP
explains it; the LLM only turns the SHAP output into adverse-action prose,
and only for denials. Nothing on the explanation path feeds back into the
decision. The fairness audit sits entirely downstream: it re-scores a
dataset through the *same* decision policy and measures how the three bands
fall across age groups — it never scores or re-decides an application.

## Layout
```
data/            raw and processed datasets (gitignored)
src/data/        dataset loading + cleaning
src/features/    feature engineering + single-row inference prep
src/model/       training, evaluation, decision policy, artifact persistence
src/explain/     SHAP explainability
src/llm/         SHAP -> plain-language adverse-action text
src/fairness/    post-hoc group fairness audit (age-band disparate impact)
app/backend/     FastAPI service
app/frontend/    loan-officer review UI
models/          versioned trained model artifacts (gitignored)
scripts/         end-to-end command-line pipeline run
tests/           unit + integration tests, mirrors src/ structure
```

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
Download Kaggle's "Give Me Some Credit" data and place `cs-training.csv`
under `data/raw/`. Not committed to the repo.

## Train a model
```bash
python -m src.model.train                 # default (v2) config -> next free models/vN/ (model, eval report, calibration plot)
python -m src.model.train --config v1     # the original baseline: every feature, age included
python -m src.model.report v2             # re-print a version's AUC / KS / Brier
python -m scripts.tune_hparams --config v2 --out /tmp/v2_grid.csv   # the CV grid behind v2's params
```

A model version is a **config** (`src/model/config.py`: which engineered
columns the model may see, plus XGBoost hyperparameters), not a code fork.
Every version shares the same seed and stratified train/validation split,
so eval reports are directly comparable. Each version's `metadata.json`
also records the decision cutoffs recommended for its calibration (see
[Model v2](#model-v2--the-less-discriminatory-alternative)).

## Command-line pipeline
```bash
python -m scripts.run_pipeline --row 5                    # data -> model -> SHAP -> notice
python -m scripts.run_pipeline --row 5 --print-prompt-only  # no LLM call
```

## Run the API
```bash
cp .env.example .env          # then set ANTHROPIC_API_KEY (optional; falls back to an offline stub)
uvicorn app.backend.main:app --reload
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/health` | model version + which LLM (`anthropic` / `stub`) |
| `POST` | `/predict` | application → probability + three-band decision |
| `POST` | `/explain` | application → structured SHAP contributions |
| `POST` | `/adverse-action` | denials only → statement of specific reasons (`409` otherwise) |
| `POST` | `/review` | full pipeline, persisted; returns a record id |
| `GET`  | `/applications`, `/applications/{id}` | stored review records |

### Decision policy

The model outputs `P(serious delinquency)`; `src/model/decision.py` applies
two cutoffs. The defaults are v2's recommended values (derived on its
validation split to keep the same band sizes as the original v1 policy,
which was `0.08` / `0.30`); override with `AIU_APPROVE_BELOW` /
`AIU_DENY_AT_OR_ABOVE`, and always set them together with
`AIU_MODEL_VERSION` — a version's probabilities are not automatically on
the default scale:

| Band | Rule | Adverse-action notice |
| --- | --- | --- |
| `approved` | `P < 0.08` | none |
| `referred` | `0.08 ≤ P < 0.28` | none (routed to a loan officer) |
| `denied`   | `P ≥ 0.28` | generated |

### `POST /review` — one example per band

Run against `models/v1` (the original baseline, under its `0.08 / 0.30`
cutoffs) with a real API key (`llm_provider: anthropic`,
`model: claude-haiku-4-5`). Each call also persists a record. v2 produces
the same shape of response; only the probabilities, the `deny_at_or_above`
threshold, and the absence of age-derived features differ.

**Approved** — `P(default) = 0.0219`
```json
{
  "id": 1,
  "decision": {"decision": "approved", "probability": 0.0219,
               "thresholds": {"approve_below": 0.08, "deny_at_or_above": 0.3}},
  "adverse_action": null
}
```

**Referred** — `P(default) = 0.0838`
```json
{
  "id": 2,
  "decision": {"decision": "referred", "probability": 0.0838,
               "thresholds": {"approve_below": 0.08, "deny_at_or_above": 0.3}},
  "adverse_action": null
}
```

**Denied** — `P(default) = 0.3809`
```json
{
  "id": 3,
  "decision": {"decision": "denied", "probability": 0.3809,
               "thresholds": {"approve_below": 0.08, "deny_at_or_above": 0.3}},
  "adverse_action": {
    "llm_provider": "anthropic",
    "model": "claude-haiku-4-5",
    "reason_features": [
      "total_past_due_count",
      "NumberRealEstateLoansOrLines",
      "RevolvingUtilizationOfUnsecuredLines",
      "DebtRatio"
    ],
    "notice_text": "Your application for credit has been denied for the following specific reasons:\n\n• Number of separate periods with past-due payments in your history\n• Number of real-estate-secured loans or lines of credit on your file\n• Proportion of your available revolving credit that is currently in use\n• Level of your monthly debt payments relative to your monthly income\n\nYou have the right to request a copy of any credit report used in this decision and to dispute the accuracy of information in your credit file."
  }
}
```

Age and every age-derived feature are excluded from `reason_features`
regardless of SHAP rank (Regulation B, 12 CFR 1002.6(b)(2)).

## Run the frontend

A React + Vite loan-officer UI over the API: a form to run new applications
through `/review`, a filterable queue of stored records, and a per-record view
with the probability placed against the two cutoffs, a diverging SHAP chart of
the per-feature contributions, and the adverse-action notice for denials.

```bash
uvicorn app.backend.main:app --reload      # backend on :8000 (as above)

cd app/frontend
npm install
npm run dev                                # http://localhost:5173
```

The dev server proxies `/api/*` to the backend, so no CORS setup is needed
locally. A separately hosted build calls the API cross-origin; list its origin
in `AIU_CORS_ORIGINS`. See `app/frontend/README.md` for details.

## Fairness audit (`src/fairness/`)

An independent, post-hoc layer that re-scores a dataset through the exact
serving decision policy and measures how the three bands fall across
demographic groups. It runs **after** the model decides and feeds nothing
back — it cannot change a score or a decision (the core architectural rule).

```bash
python -m src.fairness.report v2                 # audit models/v2 on the validation split
python -m src.fairness.report v2 --split all     # audit on the whole dataset
python -m src.fairness.report v2 --print-only    # print, write no artifact
python -m src.fairness.report v1                 # audit the v1 baseline under the cutoffs in its metadata.json (or the code defaults)
```

Cutoffs come from `--approve-below` / `--deny-at-or-above` if given, else
from the version's `metadata.json` (`recommended_cutoffs`), else from the
code defaults; the summary line says which.

The JSON report is written to `models/<version>/fairness_audit.json`,
alongside the model it audits; the model's own immutable files are never
touched, and an existing report is only replaced with `--force`.

**What it computes** (all as plain dicts / DataFrames, per the cross-layer
convention):

| Metric | Definition |
| --- | --- |
| approval / referral / denial rate per group | share of each group placed in that band |
| adverse-impact ratio (AIR) | group approval rate ÷ approval rate of the most-approved group |
| four-fifths rule | AIR `< 0.80` for any group is flagged (`passes: false`) |
| denial-rate ratio | group denial rate ÷ denial rate of the least-denied group; flagged `> 1.25` |

An "acceptance" AIR (approved **or** referred, i.e. "not denied") is
reported alongside the strict approval AIR, since a referral is not itself
an adverse action.

### Protected attribute — and its limitation

"Give Me Some Credit" carries no race, sex, ethnicity, national-origin, or
marital-status fields. `age` is the only demographic attribute in it, and
age is an ECOA-protected basis, so the audit groups applicants by the **same
age bands** the feature layer already bins to (`src/features/engineer.py`).
This is a dataset limitation, not a modelling choice: a production
fair-lending audit would repeat every metric here for race, sex, national
origin, marital status, age (≥ 62), and receipt of public assistance —
typically via a proxy method such as BISG where those attributes are not
collected directly.

### How the design connects to ECOA adverse-action requirements

ECOA and Regulation B require that a declined applicant receive the
**specific, accurate principal reasons** for the decision, and separately
that a creditor's policies not produce an unjustified **disparate impact**
on a protected basis. The pipeline is built around both halves:

- **Reasons are accurate by construction.** The gradient-boosted model
  makes the decision; SHAP identifies the factors that actually moved *that*
  application; the LLM only renders those factors into sentences. It has no
  access to the model or the score and cannot introduce, drop, or re-rank a
  reason — so what the applicant is told always matches why the model
  decided.
- **Age is never given as a reason.** Age and every age-derived feature are
  removed from the adverse-action reason list regardless of SHAP rank
  (Reg B, 12 CFR 1002.6(b)(2)). An applicant is never told they were denied
  because of their age.
- **The three bands keep denials narrow and reviewable.** Only `denied`
  triggers a notice; the ambiguous middle band is `referred` to a human
  rather than auto-declined, which shrinks the set of decisions that must
  carry a statement of specific reasons.
- **The audit closes the loop.** Adverse-action notices explain individual
  decisions; this audit checks the *aggregate* pattern of those same
  decisions for disparate impact on a protected basis.

### Reading the v1 result

On the shipped v1 model the audit flags **REVIEW**: `age` is a model
feature and older applicants are genuinely lower-risk in this dataset, so
the youngest bands sit well below the 0.80 approval AIR and above the 1.25
denial-rate ratio. That is exactly the signal the audit exists to surface.
A production response would be a less-discriminatory-alternative search —
e.g. removing `age` and the age-derived features from the model, or
constraining/monitoring their effect — followed by re-running this audit;
the point of keeping the audit as its own layer is that such a change is
measured here without touching the decision, explanation, or LLM code.
That search is what model v2, below, does.

## Model v2 — the less-discriminatory alternative

v2 is the first step of that search: **`age` and every age-derived feature
(`age_bin_*`, `credit_lines_per_year_of_age`) are removed from the model**,
the hyperparameters are re-tuned by a small 3-fold CV grid on the training
split (`scripts/tune_hparams.py`), and the audit is re-run. Nothing else
changes: same data pipeline, same seed and split, same SHAP → reasons →
LLM path, same three-band decision code. **v2 is the default model** — it
is what `python -m src.model.train` builds and what the API serves unless
told otherwise. Serving the v1 baseline is configuration (version and its
cutoffs together):

```bash
AIU_MODEL_VERSION=v1 AIU_APPROVE_BELOW=0.08 AIU_DENY_AT_OR_ABOVE=0.30 uvicorn app.backend.main:app
```

The serving path aligns each application to the loaded version's
`feature_names.json`, so v2 needs no backend change; the audit still groups
by age because it reads `age` from the cleaned data, not from the model's
features. One property v2 has that v1 does not: two applications that
differ only in age get the *identical* score (`tests/test_model_config.py`
checks this).

**Feature set and hyperparameters.** 23 features → 15. Five candidate
interpretable replacement features (monthly debt payment, has-real-estate
flag, unsecured-line count, utilization > 100% flag, 90-day share of
past-dues) were each tried and none moved validation AUC by more than
±0.0005, so no new feature was added. The CV grid was flat (0.0006 AUC
across the whole grid, fold std ≈ 0.004); v2 keeps v1's depth 4 and takes a
slower, more regularised setting (`learning_rate` 0.05 → 0.03,
`min_child_weight` 5 → 20, `reg_lambda` 1 → 5, early-stopped at 337 rounds
vs 209).

**Cutoffs.** v2's probabilities sit on almost the same scale as v1's, so
the same policy reasoning (keep ≈ 80 % auto-approved and ≈ 6 % auto-denied,
then read off the default rate each band carries) gives `approve_below =
0.08`, `deny_at_or_above = 0.28`. They are recorded in
`models/v2/metadata.json` as `recommended_cutoffs` and, since v2 is the
default model, they are also the code defaults in `src/model/decision.py`.
Under v1's `0.30` the fairness picture below is the same to two decimals.

### v1 vs v2 — validation split, n = 30 000

| | v1 | v2 | Δ |
| --- | --- | --- | --- |
| features | 23 | 15 (no age-derived) | −8 |
| AUC-ROC | 0.8697 | 0.8655 | −0.0042 |
| KS statistic | 0.585 | 0.572 | −0.014 |
| Brier score | 0.04887 | 0.04900 | +0.00014 |
| cutoffs | 0.08 / 0.30 | 0.08 / 0.28 | |
| approved — share / observed default rate | 79.5 % / 2.1 % | 79.5 % / 2.2 % | |
| referred — share / observed default rate | 14.8 % / 15.7 % | 14.4 % / 14.8 % | |
| denied — share / observed default rate | 5.7 % / 46.7 % | 6.1 % / 45.5 % | |

| age band | n | observed default rate | v1 approval | v1 approval AIR | v1 denial ratio | v2 approval | v2 approval AIR | v2 denial ratio |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| 18-24 | 572 | 11.9 % | 57.7 % | **0.611** | **8.64** | 61.9 % | **0.677** | **5.13** |
| 25-34 | 3 737 | 12.1 % | 62.9 % | **0.667** | **10.12** | 68.6 % | **0.750** | **6.97** |
| 35-44 | 5 944 | 8.8 % | 72.2 % | **0.765** | **7.21** | 74.5 % | 0.815 | **5.03** |
| 45-54 | 7 361 | 7.5 % | 76.4 % | 0.809 | **5.97** | 77.1 % | 0.843 | **4.30** |
| 55-64 | 6 637 | 4.2 % | 87.8 % | 0.930 | **2.71** | 84.1 % | 0.920 | **2.36** |
| 65+ | 5 749 | 2.2 % | 94.4 % | 1.000 | 1.00 | 91.4 % | 1.000 | 1.00 |
| **four-fifths rule** | | | | **REVIEW** (min AIR 0.611, max ratio 10.1) | | | **REVIEW** (min AIR 0.677, max ratio 7.0) | |

(Bold = fails the 0.80 AIR / 1.25 denial-ratio flag. The acceptance AIR —
approved *or* referred — passes for every band under both models: min
0.897 for v1, 0.901 for v2. `models/v1/fairness_audit.json` in the repo was
written with `--split all`; the numbers above are all on the held-out
validation split so the two models are compared on the same rows.)

### The honest tradeoff

- **What it cost:** −0.004 AUC and −0.014 KS. Brier is unchanged to four
  decimals; the band shares and per-band default rates are essentially the
  same, so the portfolio-level risk of the policy did not move. The tuning
  recovered about +0.0005 AUC of the −0.0047 lost by dropping the features —
  the search surface is flat and there is little to recover.
- **What it bought:** the youngest band's approval AIR rose from 0.61 to
  0.68, 35-44 now clears 0.80, and the worst denial-rate ratio fell from
  10.1× to 7.0×. Those are real reductions, and v2 is age-blind at the
  individual level.
- **What it did not buy:** v2 **still fails the four-fifths rule**, on both
  the approval AIR (18-24, 25-34) and the denial-rate ratio (every band but
  65+). The reason is visible in the "observed default rate" column: the
  outcome itself runs from 12 % for the youngest applicants to 2 % for the
  oldest, and the remaining features (utilization, past-due counts,
  real-estate lines, income) carry a good part of that age signal. Any
  accurate model of this outcome will approve older applicants more often;
  removing `age` removes the *direct* channel, not the correlation. It also
  slightly mis-calibrates by group — v2's mean predicted probability is now
  below the observed default rate for 18-24 (9.6 % vs 11.9 %) and above it
  for 65+ (3.2 % vs 2.2 %), which is the cost of forbidding the model a true
  signal.

So v2 is a *less* discriminatory alternative, not a non-discriminatory one.
The next steps in a real search would be to measure and constrain the
strongest age proxies among the remaining features, to decide whether the
strict approval AIR or the acceptance AIR is the right test for a policy
whose middle band is reviewed by a person, and to weigh the (small) AUC
cost against the (partial) disparity reduction as a business and legal
judgement rather than a modelling one. For reference, Regulation B does
permit age in an "empirically derived, demonstrably and statistically
sound" credit-scoring system provided applicants 62 and older are not
assigned a negative factor (12 CFR 1002.6(b)(2)(ii)) — v1's use of age is
the permitted kind, and v2 is the stricter alternative that a
disparate-impact review would ask to see evaluated.

## Tests
```bash
pytest -q
```
