"""Deterministic mapping from model features to adverse-action reason text.

This is the compliance-critical half of the LLM layer, and it is plain code
with no model in it. Given the SHAP explanation for a denied application, it
picks the principal risk-increasing factors and turns each into a short,
vetted plain-language statement. The LLM only ever sees these statements --
it never names a reason itself -- so the reasons given to an applicant are
auditable and cannot be hallucinated.

Two rules are enforced here:

1. Age is off limits. ECOA / Regulation B (12 CFR 1002.6(b)(2)) sharply
   restricts using age in a credit decision and, in particular, an
   adverse-action notice must not tell an applicant they were denied
   because of their age. Every age-derived feature is dropped from the
   reason list regardless of how strongly SHAP weighted it. (Whether the
   *model* should see age at all is a separate question for the Phase 8
   fairness audit; this module only governs what the applicant is told.)

2. No silent gaps. If a risk-increasing feature has no entry in
   ``FEATURE_REASONS`` we raise rather than omit it or pass a raw feature
   name through to the applicant.
"""

from __future__ import annotations

from typing import Any

# Feature name -> the reason statement a declined applicant will see.
#
# Phrased as factor labels ("Level of ...", "Number of ...") rather than as
# assertions about the applicant's specific values. SHAP measures how much a
# feature moved *this* decision relative to the population baseline, which is
# not the same as a claim like "you have missed payments"; a factor label is
# accurate whatever the underlying value, and it matches how real
# adverse-action reason codes are written.
FEATURE_REASONS: dict[str, str] = {
    "RevolvingUtilizationOfUnsecuredLines": (
        "Proportion of your available revolving credit that is currently in use"
    ),
    "DebtRatio": (
        "Level of your monthly debt payments relative to your monthly income"
    ),
    "MonthlyIncome": (
        "Level of income relative to the obligations on your credit file"
    ),
    "NumberOfOpenCreditLinesAndLoans": (
        "Number of open credit lines and loans on your credit file"
    ),
    "NumberRealEstateLoansOrLines": (
        "Number of real-estate-secured loans or lines of credit on your file"
    ),
    "NumberOfDependents": (
        "Level of income relative to the number of dependents you support"
    ),
    "NumberOfTimes90DaysLate": (
        "Number of payments 90 or more days past due on your credit file"
    ),
    "NumberOfTime60-89DaysPastDueNotWorse": (
        "Number of payments 60 to 89 days past due on your credit file"
    ),
    "NumberOfTime30-59DaysPastDueNotWorse": (
        "Number of payments 30 to 59 days past due on your credit file"
    ),
    "income_missing": (
        "Income could not be verified from the information provided"
    ),
    "dependents_missing": (
        "Information needed to assess the application was incomplete"
    ),
    "total_past_due_count": (
        "Number of separate periods with past-due payments in your history"
    ),
    "has_past_due": (
        "Presence of past-due payments in your recent credit history"
    ),
    "income_per_dependent": (
        "Level of income relative to the number of dependents you support"
    ),
    "credit_lines_per_year_of_age": (
        "Number of credit lines opened relative to the length of your credit "
        "history"
    ),
    "has_dependents": (
        "Level of your household obligations relative to your income"
    ),
}

# Age itself plus every one-hot age bucket produced by the feature layer.
AGE_DERIVED_FEATURES: frozenset[str] = frozenset(
    {
        "age",
        "age_bin_18-24",
        "age_bin_25-34",
        "age_bin_35-44",
        "age_bin_45-54",
        "age_bin_55-64",
        "age_bin_65+",
    }
)

INCREASES_RISK = "increases_risk"


def select_reasons(
    explanation: dict[str, Any], max_reasons: int = 4
) -> list[dict[str, Any]]:
    """Return the principal adverse-action reasons for a denied application.

    Input is the dict produced by :func:`src.explain.explainer.explain_row`.
    Output is an ordered list (most influential first) of::

        {"feature": str, "statement": str, "shap_value": float}

    Age-derived features are excluded. Raises ``ValueError`` if a
    risk-increasing feature has no reason template, or if nothing is left to
    report.
    """
    if max_reasons < 1:
        raise ValueError("max_reasons must be at least 1")

    candidates = [
        c
        for c in explanation["contributions"]
        if c["direction"] == INCREASES_RISK
        and c["feature"] not in AGE_DERIVED_FEATURES
    ]

    reasons: list[dict[str, Any]] = []
    for contribution in candidates:
        feature = contribution["feature"]
        try:
            statement = FEATURE_REASONS[feature]
        except KeyError:
            raise ValueError(
                f"No adverse-action reason template for feature {feature!r}; "
                "add one to FEATURE_REASONS rather than exposing a raw feature "
                "name to the applicant."
            ) from None
        reasons.append(
            {
                "feature": feature,
                "statement": statement,
                "shap_value": contribution["shap_value"],
            }
        )
        if len(reasons) == max_reasons:
            break

    if not reasons:
        raise ValueError(
            "No non-age risk-increasing factors were found to explain this "
            "denial; a statement of specific reasons cannot be produced."
        )
    return reasons
