"""Explainability layer: per-application SHAP contributions as plain data."""

from src.explain.explainer import (
    DECREASES_RISK,
    INCREASES_RISK,
    build_explainer,
    explain_row,
    top_contributors,
)

__all__ = [
    "build_explainer",
    "explain_row",
    "top_contributors",
    "INCREASES_RISK",
    "DECREASES_RISK",
]
