"""Request/response models for the API.

`ApplicationIn` is the only real input model: the ten raw "Give Me Some
Credit" fields, with the two that have genuine missingness left optional.
Responses are typed just enough for the OpenAPI docs; the nested
`explanation` / `adverse_action` payloads are the same dicts the service
layer already produces.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Several response fields start with "model_" (a pydantic-protected prefix)
# or are literally named "model"; opt out of the namespace guard.
_ALLOW_MODEL = ConfigDict(protected_namespaces=())


class ApplicationIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    RevolvingUtilizationOfUnsecuredLines: float = Field(ge=0)
    age: int = Field(ge=18, le=120)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse"
    )
    DebtRatio: float = Field(ge=0)
    MonthlyIncome: float | None = Field(default=None, ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(ge=0)
    NumberOfTimes90DaysLate: int = Field(ge=0)
    NumberRealEstateLoansOrLines: int = Field(ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse"
    )
    NumberOfDependents: int | None = Field(default=None, ge=0)

    def to_raw_dict(self) -> dict[str, Any]:
        """Return the dict keyed by the dataset's real column names."""
        return {
            "RevolvingUtilizationOfUnsecuredLines": self.RevolvingUtilizationOfUnsecuredLines,
            "age": self.age,
            "NumberOfTime30-59DaysPastDueNotWorse": self.NumberOfTime30_59DaysPastDueNotWorse,
            "DebtRatio": self.DebtRatio,
            "MonthlyIncome": self.MonthlyIncome,
            "NumberOfOpenCreditLinesAndLoans": self.NumberOfOpenCreditLinesAndLoans,
            "NumberOfTimes90DaysLate": self.NumberOfTimes90DaysLate,
            "NumberRealEstateLoansOrLines": self.NumberRealEstateLoansOrLines,
            "NumberOfTime60-89DaysPastDueNotWorse": self.NumberOfTime60_89DaysPastDueNotWorse,
            "NumberOfDependents": self.NumberOfDependents,
        }


class HealthOut(BaseModel):
    model_config = _ALLOW_MODEL

    status: str
    model_version: str
    llm_provider: str


class DecisionOut(BaseModel):
    decision: str
    probability: float
    thresholds: dict[str, float]


class ExplanationOut(BaseModel):
    predicted_probability: float
    base_value: float
    base_rate: float
    logodds_margin: float
    contributions: list[dict[str, Any]]


class AdverseActionOut(BaseModel):
    model_config = _ALLOW_MODEL

    notice_text: str
    decision: str
    model: str
    reason_statements: list[str]
    reason_features: list[str]
    reason_shap_values: list[float]
    llm_provider: str


class ReviewOut(BaseModel):
    model_config = _ALLOW_MODEL

    id: int | None
    model_version: str
    decision: DecisionOut
    explanation: ExplanationOut
    adverse_action: AdverseActionOut | None


class ReviewRecordOut(ReviewOut):
    created_at: str
    application: dict[str, Any]
