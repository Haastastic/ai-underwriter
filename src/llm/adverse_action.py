"""Turn a structured SHAP explanation into ECOA-style adverse-action text.

This is the LLM layer's public entry point and it is a pure function of its
arguments plus the injected :class:`~src.llm.client.LLMClient`:

    structured SHAP explanation + decision  ->  plain-language notice

The layer never sees the model, its probability, or any score, and it never
decides anything. ``decision`` is passed in by the caller (it comes from the
model layer) and this function only *renders language* for it -- it will
refuse to run for anything other than a denial, and it does not look at the
predicted probability when choosing or wording reasons. The principal
reasons are selected deterministically in :mod:`src.llm.reasons`; the model
only assembles them into prose.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from src.llm.client import DEFAULT_MODEL, LLMClient, build_default_client
from src.llm.prompt import build_system_prompt, build_user_prompt
from src.llm.reasons import select_reasons

DENIED = "denied"


@dataclass(frozen=True)
class AdverseActionResult:
    """The generated notice plus the audit trail behind it."""

    notice_text: str
    decision: str
    model: str
    reason_statements: tuple[str, ...]
    reason_features: tuple[str, ...]
    reason_shap_values: tuple[float, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_adverse_action(
    *,
    explanation: dict[str, Any],
    decision: str,
    max_reasons: int = 4,
    applicant_reference: str | None = None,
    client: LLMClient | None = None,
    model: str = DEFAULT_MODEL,
) -> AdverseActionResult:
    """Generate the statement of specific reasons for a denied application.

    Parameters
    ----------
    explanation:
        The dict returned by :func:`src.explain.explainer.explain_row`.
    decision:
        The model layer's decision. Must be ``"denied"`` -- an
        adverse-action notice is not produced for any other outcome, and
        this function will not be the thing that turns a score into a
        decision.
    max_reasons:
        Cap on the number of principal reasons (ECOA notices typically list
        up to four).
    applicant_reference:
        Optional opaque identifier to echo in the notice (never a name).
    client:
        Any object satisfying :class:`~src.llm.client.LLMClient`. Defaults
        to a real Anthropic client.
    model:
        LLM model id used for prose only.

    Raises
    ------
    ValueError
        If ``decision`` is not ``"denied"``, or if no reportable reasons
        remain after excluding age-derived features.
    """
    if decision != DENIED:
        raise ValueError(
            f"adverse-action text is only generated for a {DENIED!r} decision, "
            f"got {decision!r}"
        )

    reasons = select_reasons(explanation, max_reasons=max_reasons)
    statements = tuple(r["statement"] for r in reasons)

    client = client or build_default_client()
    notice_text = client.complete(
        system=build_system_prompt(),
        user=build_user_prompt(
            statements, decision=decision, applicant_reference=applicant_reference
        ),
        model=model,
    )

    return AdverseActionResult(
        notice_text=notice_text,
        decision=decision,
        model=model,
        reason_statements=statements,
        reason_features=tuple(r["feature"] for r in reasons),
        reason_shap_values=tuple(float(r["shap_value"]) for r in reasons),
    )


def result_to_json(result: AdverseActionResult) -> str:
    return json.dumps(result.to_dict(), indent=2)
