"""Fairness-audit layer: measure decision outcomes across protected groups.

This layer runs **after** the model has decided. It never scores an
application, never sees a SHAP value, and nothing it computes is fed back
into ``src.model`` -- it only reads the three-band decisions the model layer
already produced and reports how they are distributed across groups. Keeping
it downstream and side-effect-free is what lets it sit in the repo without
touching the core rule that the model (not the LLM, and certainly not an
audit script) makes the credit decision.

Public surface
--------------
``assign_age_band``      raw age -> the same band label the feature layer bins to
``assign_decision_band`` probability + cutoffs -> "approved" / "referred" / "denied"
``group_rates``          (group, decision) table -> per-group selection/denial rates
``disparate_impact``     per-group rates -> adverse-impact ratios + four-fifths flag
``denial_rate_disparity``per-group rates -> denial-rate ratios vs the best group
``audit_decisions``      a scored+grouped frame -> the full structured audit dict
``build_audit_frame``    model version + dataset -> that scored+grouped frame
``run_audit``            build_audit_frame + audit_decisions in one call
"""

from src.fairness.audit import (
    audit_decisions,
    build_audit_frame,
    run_audit,
    score_frame,
)
from src.fairness.groups import (
    AGE_BAND_LIMITATION,
    assign_age_band,
    assign_decision_band,
)
from src.fairness.metrics import (
    DENIAL_RATIO_FLAG,
    FOUR_FIFTHS,
    denial_rate_disparity,
    disparate_impact,
    group_rates,
)

__all__ = [
    "assign_age_band",
    "assign_decision_band",
    "AGE_BAND_LIMITATION",
    "group_rates",
    "disparate_impact",
    "denial_rate_disparity",
    "FOUR_FIFTHS",
    "DENIAL_RATIO_FLAG",
    "audit_decisions",
    "score_frame",
    "build_audit_frame",
    "run_audit",
]
