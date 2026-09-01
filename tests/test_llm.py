"""LLM adverse-action layer tests.

The SHAP -> text boundary is the priority: these lock that the layer
consumes only the structured explanation dict, renders prose through an
injectable client (no network, no API key), never leaks a score or age into
the prompt, and never turns a score into a decision.
"""

import json
from types import SimpleNamespace

import pytest

from src.llm.adverse_action import AdverseActionResult, generate_adverse_action
from src.llm.client import DEFAULT_MODEL, AnthropicClient
from src.llm.prompt import build_user_prompt
from src.llm.reasons import AGE_DERIVED_FEATURES, select_reasons

NOTICE = "Your application was denied. Reasons follow:\n- a\n- b\nThank you."


class FakeClient:
    """Records the prompts it was called with; returns a canned notice."""

    def __init__(self, reply=NOTICE):
        self.reply = reply
        self.calls = []

    def complete(self, *, system, user, model):
        self.calls.append({"system": system, "user": user, "model": model})
        return self.reply


def make_explanation(contributions, predicted_probability=0.91, base_value=-2.66):
    return {
        "predicted_probability": predicted_probability,
        "base_value": base_value,
        "base_rate": 0.065,
        "logodds_margin": 2.4,
        "contributions": contributions,
    }


def c(feature, shap_value, value=1.0):
    direction = "increases_risk" if shap_value > 0 else "decreases_risk"
    return {
        "feature": feature,
        "value": value,
        "shap_value": shap_value,
        "direction": direction,
    }


@pytest.fixture
def explanation():
    # Deliberately unsorted; select_reasons must rank by |shap| itself is not
    # required (explain_row pre-sorts), but order here is the intended output.
    return make_explanation(
        [
            c("RevolvingUtilizationOfUnsecuredLines", 0.51, 0.95),
            c("NumberOfTimes90DaysLate", 0.33, 2),
            c("DebtRatio", 0.21, 0.8),
            c("MonthlyIncome", 0.12, 1500.0),
            c("age", 0.40, 24),          # large, but must be excluded
            c("NumberOfDependents", -0.30, 3),  # protective, must be excluded
        ]
    )


# --- select_reasons ------------------------------------------------------


def test_select_reasons_takes_top_k_risk_increasing_in_order(explanation):
    reasons = select_reasons(explanation, max_reasons=3)
    assert [r["feature"] for r in reasons] == [
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTimes90DaysLate",
        "DebtRatio",
    ]
    assert all(isinstance(r["statement"], str) and r["statement"] for r in reasons)


def test_select_reasons_excludes_age_even_when_it_dominates(explanation):
    reasons = select_reasons(explanation, max_reasons=10)
    features = {r["feature"] for r in reasons}
    assert features.isdisjoint(AGE_DERIVED_FEATURES)
    assert "age" not in features


def test_select_reasons_excludes_risk_reducing_features(explanation):
    reasons = select_reasons(explanation, max_reasons=10)
    assert "NumberOfDependents" not in {r["feature"] for r in reasons}


def test_select_reasons_raises_on_unmapped_feature():
    exp = make_explanation([c("some_new_unmapped_feature", 0.9)])
    with pytest.raises(ValueError, match="no adverse-action reason template|No adverse-action reason template"):
        select_reasons(exp)


def test_select_reasons_raises_when_nothing_reportable():
    exp = make_explanation([c("age", 0.9), c("MonthlyIncome", -0.4)])
    with pytest.raises(ValueError, match="cannot be produced"):
        select_reasons(exp)


# --- generate_adverse_action: decision independence --------------------


def test_only_generated_for_a_denial(explanation):
    fake = FakeClient()
    for decision in ("approved", "referred", "APPROVED", "", "denied "):
        with pytest.raises(ValueError, match="only generated for"):
            generate_adverse_action(
                explanation=explanation, decision=decision, client=fake
            )
    assert fake.calls == []  # never reached the model


def test_decision_is_authoritative_not_re_derived_from_probability():
    # A low-risk-looking explanation still yields a denial notice, because
    # the decision is an input, not something this layer computes.
    low_risk = make_explanation(
        [c("DebtRatio", 0.05), c("MonthlyIncome", 0.04)],
        predicted_probability=0.02,
    )
    fake = FakeClient()
    result = generate_adverse_action(
        explanation=low_risk, decision="denied", client=fake
    )
    assert result.decision == "denied"
    assert fake.calls, "client should have been asked to render the notice"


# --- generate_adverse_action: result + prompt content -----------------


def test_result_carries_notice_and_audit_trail(explanation):
    fake = FakeClient()
    result = generate_adverse_action(
        explanation=explanation, decision="denied", max_reasons=3, client=fake
    )
    assert isinstance(result, AdverseActionResult)
    assert result.notice_text == NOTICE
    assert result.model == DEFAULT_MODEL
    assert result.reason_features == (
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTimes90DaysLate",
        "DebtRatio",
    )
    assert len(result.reason_statements) == 3
    assert len(result.reason_shap_values) == 3


def test_prompt_contains_every_reason_statement_in_order(explanation):
    fake = FakeClient()
    generate_adverse_action(
        explanation=explanation, decision="denied", max_reasons=3, client=fake
    )
    user_prompt = fake.calls[0]["user"]
    reasons = select_reasons(explanation, max_reasons=3)
    positions = [user_prompt.find(r["statement"]) for r in reasons]
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions)


def test_prompt_never_leaks_score_probability_or_age(explanation):
    fake = FakeClient()
    generate_adverse_action(
        explanation=explanation, decision="denied", client=fake
    )
    blob = fake.calls[0]["system"] + "\n" + fake.calls[0]["user"]
    assert "0.91" not in blob          # predicted_probability
    assert "-2.66" not in blob         # base_value
    assert "2.4" not in blob           # logodds_margin
    assert "age" not in fake.calls[0]["user"].lower()
    assert "24" not in fake.calls[0]["user"]  # the age value


def test_age_feature_never_appears_in_audit_trail(explanation):
    result = generate_adverse_action(
        explanation=explanation, decision="denied", max_reasons=10, client=FakeClient()
    )
    assert "age" not in result.reason_features


# --- purity / serialisation ------------------------------------------


def test_same_inputs_same_output(explanation):
    a = generate_adverse_action(
        explanation=explanation, decision="denied", client=FakeClient()
    )
    b = generate_adverse_action(
        explanation=explanation, decision="denied", client=FakeClient()
    )
    assert a == b


def test_result_is_frozen_and_json_serialisable(explanation):
    result = generate_adverse_action(
        explanation=explanation, decision="denied", client=FakeClient()
    )
    with pytest.raises(Exception):
        result.notice_text = "tampered"  # frozen dataclass
    round_tripped = json.loads(json.dumps(result.to_dict()))
    assert round_tripped["notice_text"] == NOTICE


def test_max_reasons_is_respected(explanation):
    result = generate_adverse_action(
        explanation=explanation, decision="denied", max_reasons=2, client=FakeClient()
    )
    assert len(result.reason_statements) == 2


def test_model_override_is_passed_through(explanation):
    fake = FakeClient()
    generate_adverse_action(
        explanation=explanation, decision="denied", client=fake, model="claude-haiku-4-5"
    )
    assert fake.calls[0]["model"] == "claude-haiku-4-5"


# --- prompt builder detail ------------------------------------------


def test_build_user_prompt_includes_optional_reference():
    prompt = build_user_prompt(["reason one"], applicant_reference="APP-123")
    assert "APP-123" in prompt
    assert "reason one" in prompt
    assert "Decision: denied" in prompt


# --- AnthropicClient adapter (no real network) --------------------


def test_anthropic_client_calls_messages_create_with_expected_shape():
    captured = {}

    class StubMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="  hello notice  "),
                    SimpleNamespace(type="thinking", text="ignore me"),
                ]
            )

    stub = SimpleNamespace(messages=StubMessages())
    adapter = AnthropicClient(client=stub, max_tokens=512)

    out = adapter.complete(system="SYS", user="USR", model="claude-opus-5")

    assert out == "hello notice"  # joined text blocks only, stripped
    assert captured["model"] == "claude-opus-5"
    assert captured["max_tokens"] == 512
    assert captured["system"] == "SYS"
    assert captured["messages"] == [{"role": "user", "content": "USR"}]
