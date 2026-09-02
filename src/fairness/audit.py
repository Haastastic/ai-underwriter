"""Assemble the fairness audit: score a dataset, group it, measure outcomes.

Two layers:

* :func:`audit_decisions` is pure -- a DataFrame of already-decided,
  already-grouped applications in, a structured audit dict out. It touches
  no model.
* :func:`score_frame` / :func:`build_audit_frame` / :func:`run_audit` are the
  adapter that turns a saved model version + a raw dataset into the frame
  :func:`audit_decisions` expects. They call ``model.predict_proba`` and the
  decision-policy cutoffs, exactly as the serving path does, and then stop --
  no value computed here is ever written back into the model or its inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.fairness.groups import (
    AGE_BAND_LIMITATION,
    assign_age_band,
    assign_decision_band,
)
from src.fairness.metrics import (
    DECISION_BANDS,
    denial_rate_disparity,
    disparate_impact,
    group_rates,
)
from src.model.artifacts import load_model
from src.model.config import RANDOM_SEED, VAL_FRACTION, align_features
from src.model.dataset import build_model_frame, split_xy, train_val_split
from src.model.decision import APPROVE_BELOW, DENY_AT_OR_ABOVE

DEFAULT_DATA_PATH = Path("data/raw/cs-training.csv")


def score_frame(
    model,
    feature_names: list[str],
    features: pd.DataFrame,
    *,
    approve_below: float = APPROVE_BELOW,
    deny_at_or_above: float = DENY_AT_OR_ABOVE,
    y_true: pd.Series | None = None,
) -> pd.DataFrame:
    """Model + engineered frame -> a per-applicant frame ready for the audit.

    ``features`` is the engineered pipeline output (``split_xy``'s ``X``). It
    must carry the raw ``age`` column -- that is what the audit groups on --
    and every column in ``feature_names``. The model is scored on exactly
    ``feature_names``; any other column (for a version like v2 that excludes
    age from the model, that is ``age`` itself and the age-derived features)
    is used only for grouping, never for scoring. Returns a DataFrame with
    columns::

        age  age_band  probability  decision  [defaulted]

    ``defaulted`` is included only when ``y_true`` is passed.
    """
    if "age" not in features.columns:
        raise ValueError("features frame has no 'age' column to group on")
    model_input = align_features(features, feature_names)

    probability = pd.Series(
        model.predict_proba(model_input)[:, 1], index=features.index, name="probability"
    )
    decision = assign_decision_band(
        probability.to_numpy(), approve_below, deny_at_or_above
    )
    out = pd.DataFrame(
        {
            "age": features["age"].to_numpy(),
            "age_band": assign_age_band(features["age"].to_numpy()),
            "probability": probability.to_numpy(),
            "decision": decision.to_numpy(),
        }
    )
    if y_true is not None:
        out["defaulted"] = pd.Series(y_true).to_numpy().astype(int)
    return out


def build_audit_frame(
    version: str = "v1",
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    models_root: str | Path = "models",
    split: str = "val",
    approve_below: float = APPROVE_BELOW,
    deny_at_or_above: float = DENY_AT_OR_ABOVE,
) -> pd.DataFrame:
    """Load model ``version`` and score ``data_path`` into an audit frame.

    ``split`` is ``"val"`` (default -- the same stratified validation split
    ``src.model.report`` scores, so the audit is not read off rows the model
    trained on) or ``"all"`` (every row in the dataset).
    """
    if split not in {"val", "all"}:
        raise ValueError(f"split must be 'val' or 'all', got {split!r}")

    model, feature_names = load_model(Path(models_root) / version)
    df = build_model_frame(data_path)
    X, y = split_xy(df)
    # Fail early if the pipeline no longer produces something the model
    # expects; extra columns (age, for a version that excludes it) are kept
    # for grouping and dropped from the model input in score_frame.
    align_features(X, feature_names)

    if split == "val":
        _, X, _, y = train_val_split(X, y, VAL_FRACTION, RANDOM_SEED)

    return score_frame(
        model,
        feature_names,
        X,
        approve_below=approve_below,
        deny_at_or_above=deny_at_or_above,
        y_true=y,
    )


def audit_decisions(
    frame: pd.DataFrame,
    *,
    group_col: str = "age_band",
    decision_col: str = "decision",
    approve_below: float | None = None,
    deny_at_or_above: float | None = None,
) -> dict[str, Any]:
    """Structured group-fairness audit over an already-decided frame.

    ``frame`` needs a group-label column and a decision-band column; nothing
    else is read (an ``age`` / ``probability`` / ``defaulted`` column, if
    present, is ignored here). ``approve_below`` / ``deny_at_or_above`` are
    recorded in the report for provenance only -- they do not re-decide
    anything.

    Returns a JSON-serialisable dict::

        {
          "protected_attribute", "decision_bands", "thresholds",
          "n", "n_excluded_missing_group",
          "groups":  [ {group, n, approved, referred, denied,
                        approval_rate, referred_rate, denial_rate,
                        acceptance_rate}, ... ],
          "disparate_impact": {"approval": {...}, "acceptance": {...}},
          "denial_rate_disparity": {...},
          "four_fifths_rule": {"passes", "approval_air_min",
                               "approval_air_min_group",
                               "denial_ratio_max", "denial_ratio_max_group"},
          "limitations": str,
        }
    """
    if group_col not in frame.columns:
        raise ValueError(f"frame has no group column {group_col!r}")

    total = len(frame)
    rates = group_rates(frame, group_col=group_col, decision_col=decision_col)
    n_kept = int(rates["n"].sum())

    di_approval = disparate_impact(rates, favorable="approval_rate")
    di_acceptance = disparate_impact(rates, favorable="acceptance_rate")
    denial = denial_rate_disparity(rates)

    groups = [
        {
            "group": str(group),
            "n": int(row["n"]),
            "approved": int(row["approved"]),
            "referred": int(row["referred"]),
            "denied": int(row["denied"]),
            "approval_rate": float(row["approval_rate"]),
            "referred_rate": float(row["referred_rate"]),
            "denial_rate": float(row["denial_rate"]),
            "acceptance_rate": float(row["acceptance_rate"]),
        }
        for group, row in rates.iterrows()
    ]

    return {
        "protected_attribute": group_col,
        "decision_bands": list(DECISION_BANDS),
        "thresholds": {
            "approve_below": approve_below,
            "deny_at_or_above": deny_at_or_above,
        },
        "n": int(total),
        "n_excluded_missing_group": int(total - n_kept),
        "groups": groups,
        "disparate_impact": {"approval": di_approval, "acceptance": di_acceptance},
        "denial_rate_disparity": denial,
        "four_fifths_rule": {
            "passes": bool(
                di_approval["passes_four_fifths"] and denial["passes"]
            ),
            "approval_air_min": di_approval["min_ratio"],
            "approval_air_min_group": di_approval["min_ratio_group"],
            "denial_ratio_max": denial["max_ratio"],
            "denial_ratio_max_group": denial["max_ratio_group"],
        },
        "limitations": AGE_BAND_LIMITATION,
    }


def run_audit(
    version: str = "v1",
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    models_root: str | Path = "models",
    split: str = "val",
    approve_below: float = APPROVE_BELOW,
    deny_at_or_above: float = DENY_AT_OR_ABOVE,
) -> dict[str, Any]:
    """:func:`build_audit_frame` then :func:`audit_decisions`, with provenance."""
    frame = build_audit_frame(
        version,
        data_path,
        models_root=models_root,
        split=split,
        approve_below=approve_below,
        deny_at_or_above=deny_at_or_above,
    )
    report = audit_decisions(
        frame,
        approve_below=approve_below,
        deny_at_or_above=deny_at_or_above,
    )
    report["model_version"] = version
    report["dataset"] = str(data_path)
    report["split"] = split
    if "defaulted" in frame.columns:
        report["observed_default_rate"] = float(frame["defaulted"].mean())
    return report


__all__ = [
    "score_frame",
    "build_audit_frame",
    "audit_decisions",
    "run_audit",
    "DEFAULT_DATA_PATH",
]
