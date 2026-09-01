"""Prompt construction for the adverse-action notice.

Separated from the API call so the exact text sent to the model is a pure
function of structured inputs and can be asserted on in tests. The reason
statements arrive already written (see :mod:`src.llm.reasons`); the model's
only job is prose assembly.
"""

from __future__ import annotations

from collections.abc import Sequence

SYSTEM_PROMPT = """\
You write the explanatory body of a consumer credit adverse-action notice \
-- the "statement of specific reasons" required by the Equal Credit \
Opportunity Act and Regulation B when a credit application is denied.

You are given the decision (always a denial) and an ordered list of the \
principal reasons, already written as short plain-language statements.

Your job is only to render those reasons into a clear, respectful notice \
the applicant can understand. You must:
  - Use every reason you are given. Do not add, infer, merge, split, \
reorder, or soften any of them -- the provided list is authoritative and \
complete.
  - Not introduce any number: no credit score, no probability or \
likelihood, no dollar amount or threshold, no count that was not given \
to you.
  - Not mention or allude to age, race, colour, religion, national origin, \
sex, marital status, or whether the applicant receives public assistance.
  - Not give financial advice, and not promise review, reconsideration, or \
a future outcome.

Format: address the applicant as "you"; one short opening sentence stating \
that the application was denied and that the specific reasons follow; then \
the reasons as a bulleted list, one bullet per reason, in the order given; \
then a single closing sentence. Keep the whole notice under 150 words. \
Output only the notice text.\
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_prompt(
    reason_statements: Sequence[str],
    decision: str = "denied",
    applicant_reference: str | None = None,
) -> str:
    """Assemble the user message from the fixed reason statements.

    ``applicant_reference`` is an opaque identifier (e.g. an application
    number) if the caller wants one echoed; it is never a name and no
    identity is inferred from it.
    """
    lines = [f"Decision: {decision}", ""]
    if applicant_reference:
        lines.insert(1, f"Application reference: {applicant_reference}")
    lines.append("Principal reasons (use all of them, in this order):")
    lines.extend(f"{i}. {s}" for i, s in enumerate(reason_statements, start=1))
    return "\n".join(lines)
