"""API tests: the full data -> model -> SHAP -> notice pipeline over HTTP.

A small real model is trained into a tmp directory once per module (no
committed artifact needed), the LLM is the offline stub, and SQLite writes
go to a tmp file. These are end-to-end tests of `create_app`, not unit
tests of the route functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.backend.config import Settings
from app.backend.llm_client import STUB_PREFIX, StubLLMClient
from app.backend.main import create_app
from app.backend.service import UnderwritingService
from app.backend.store import ReviewStore
from src.data.schema import RAW_COLUMNS
from src.model.train import run_training

LOW_RISK_APPLICATION = {
    "RevolvingUtilizationOfUnsecuredLines": 0.05,
    "age": 52,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.2,
    "MonthlyIncome": 12000,
    "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0,
}
HIGH_RISK_APPLICATION = {
    "RevolvingUtilizationOfUnsecuredLines": 0.99,
    "age": 30,
    "NumberOfTime30-59DaysPastDueNotWorse": 4,
    "DebtRatio": 0.9,
    "MonthlyIncome": 1500,
    "NumberOfOpenCreditLinesAndLoans": 3,
    "NumberOfTimes90DaysLate": 3,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 2,
    "NumberOfDependents": 4,
}


def _write_synthetic_raw_csv(path, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    util = rng.beta(2, 5, n)
    age = rng.integers(21, 80, n)
    past_due_30 = rng.poisson(0.4, n)
    debt_ratio = rng.gamma(2.0, 0.2, n)
    income = rng.lognormal(8.6, 0.6, n)
    open_lines = rng.poisson(8, n)
    late_90 = rng.poisson(0.15, n)
    real_estate = rng.poisson(1.0, n)
    past_due_60 = rng.poisson(0.1, n)
    dependents = rng.poisson(0.8, n)

    # intercept tuned so the synthetic base rate (~5%) and score spread put a
    # genuinely low-risk applicant below 0.08 and a high-risk one above 0.30 --
    # i.e. the shipped production thresholds work on plausible inputs.
    logit = (
        -4.6
        + 3.2 * util
        + 0.9 * past_due_30
        + 1.1 * late_90
        + 0.8 * debt_ratio
        - 1.5e-5 * income
    )
    target = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    df = pd.DataFrame(
        {
            "SeriousDlqin2yrs": target,
            "RevolvingUtilizationOfUnsecuredLines": util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": past_due_30,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": past_due_60,
            "NumberOfDependents": dependents.astype(float),
        }
    )
    assert list(df.columns) == RAW_COLUMNS

    # inject the data-quality artefacts the cleaner is meant to handle
    df.loc[df.sample(frac=0.02, random_state=seed).index, "age"] = 0
    df.loc[df.sample(frac=0.2, random_state=seed + 1).index, "MonthlyIncome"] = np.nan
    df.loc[df.sample(frac=0.03, random_state=seed + 2).index, "NumberOfDependents"] = np.nan
    df.loc[df.sample(frac=0.01, random_state=seed + 3).index,
           "NumberOfTime30-59DaysPastDueNotWorse"] = 98

    df.insert(0, "Unnamed: 0", range(1, n + 1))  # loader should drop this
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("aiu")
    csv_path = root / "cs-training.csv"
    _write_synthetic_raw_csv(csv_path)
    # Pin the v1 config: these tests assert on the full 23-feature set, and
    # the training default is now v2.
    run_training(
        csv_path, models_root=root / "models", version_dir=root / "models" / "v1", config="v1"
    )
    return Settings(
        models_root=root / "models",
        model_version="v1",
        training_data_path=csv_path,
        db_path=root / "reviews.db",
        llm_model="test-model",
        max_reasons=4,
        # cutoffs tuned to this toy model's compressed score range
        approve_below=0.05,
        deny_at_or_above=0.15,
    )


@pytest.fixture
def client(settings, tmp_path):
    # fresh DB per test so /applications assertions are deterministic
    db_path = tmp_path / "reviews.db"
    service = UnderwritingService(
        settings=Settings(**{**settings.__dict__, "db_path": db_path}),
        llm_client=StubLLMClient(),
        llm_provider="stub",
        store=ReviewStore(db_path),
    )
    return TestClient(create_app(service=service))


# --- health -------------------------------------------------------------


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_version": "v1", "llm_provider": "stub"}


# --- CORS (the frontend is a separate origin) -------------------------


def test_cors_allows_configured_frontend_origin(client):
    origin = "http://localhost:5173"
    resp = client.get("/health", headers={"Origin": origin})
    assert resp.headers["access-control-allow-origin"] == origin


def test_cors_omits_header_for_unknown_origin(client):
    resp = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in resp.headers


# --- predict ----------------------------------------------------------


def test_predict_low_and_high_risk(client):
    low = client.post("/predict", json=LOW_RISK_APPLICATION).json()
    high = client.post("/predict", json=HIGH_RISK_APPLICATION).json()

    assert 0.0 <= low["probability"] <= 1.0
    assert low["probability"] < high["probability"]
    assert low["decision"] == "approved"
    assert high["decision"] == "denied"
    assert low["probability"] < high["thresholds"]["approve_below"]
    assert high["probability"] >= high["thresholds"]["deny_at_or_above"]


def test_predict_rejects_unknown_field(client):
    bad = {**LOW_RISK_APPLICATION, "employer": "ACME"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_out_of_range(client):
    bad = {**LOW_RISK_APPLICATION, "age": 9}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_allows_null_income_and_dependents(client):
    app_ = {**LOW_RISK_APPLICATION, "MonthlyIncome": None, "NumberOfDependents": None}
    assert client.post("/predict", json=app_).status_code == 200


# --- explain --------------------------------------------------------


def test_explain_returns_full_contribution_set(client):
    body = client.post("/explain", json=HIGH_RISK_APPLICATION).json()
    assert set(body) == {
        "predicted_probability",
        "base_value",
        "base_rate",
        "logodds_margin",
        "contributions",
    }
    feats = [c["feature"] for c in body["contributions"]]
    assert len(feats) == len(set(feats)) == 23
    mags = [abs(c["shap_value"]) for c in body["contributions"]]
    assert mags == sorted(mags, reverse=True)


# --- adverse action ------------------------------------------------


def test_adverse_action_for_denial(client):
    body = client.post("/adverse-action", json=HIGH_RISK_APPLICATION).json()
    assert body["decision"] == "denied"
    assert body["llm_provider"] == "stub"
    assert body["notice_text"].startswith(STUB_PREFIX)
    assert 1 <= len(body["reason_statements"]) <= 4
    assert "age" not in body["reason_features"]
    assert body["model"] == "test-model"


def test_adverse_action_409_when_not_a_denial(client):
    resp = client.post("/adverse-action", json=LOW_RISK_APPLICATION)
    assert resp.status_code == 409
    assert "denied" in resp.json()["detail"]


# --- review + persistence ---------------------------------------


def test_review_denial_persists_and_is_retrievable(client):
    review = client.post("/review", json=HIGH_RISK_APPLICATION).json()
    assert review["id"] is not None
    assert review["decision"]["decision"] == "denied"
    assert review["adverse_action"]["notice_text"].startswith(STUB_PREFIX)
    # sub-results agree on probability
    assert (
        review["decision"]["probability"]
        == review["explanation"]["predicted_probability"]
    )

    listed = client.get("/applications").json()
    assert [r["id"] for r in listed] == [review["id"]]

    fetched = client.get(f"/applications/{review['id']}").json()
    assert fetched["application"] == HIGH_RISK_APPLICATION
    assert fetched["decision"]["decision"] == "denied"
    assert fetched["adverse_action"]["reason_features"]


def test_stored_record_reports_decision_thresholds(client, settings):
    review = client.post("/review", json=HIGH_RISK_APPLICATION).json()

    fetched = client.get(f"/applications/{review['id']}").json()
    assert fetched["decision"]["thresholds"] == {
        "approve_below": settings.approve_below,
        "deny_at_or_above": settings.deny_at_or_above,
    }


def test_review_approval_has_no_adverse_action(client):
    review = client.post("/review", json=LOW_RISK_APPLICATION).json()
    assert review["decision"]["decision"] == "approved"
    assert review["adverse_action"] is None

    fetched = client.get(f"/applications/{review['id']}").json()
    assert fetched["adverse_action"] is None


def test_applications_filter_by_decision(client):
    client.post("/review", json=LOW_RISK_APPLICATION)
    client.post("/review", json=HIGH_RISK_APPLICATION)

    denied = client.get("/applications", params={"decision": "denied"}).json()
    approved = client.get("/applications", params={"decision": "approved"}).json()
    assert all(r["decision"]["decision"] == "denied" for r in denied)
    assert all(r["decision"]["decision"] == "approved" for r in approved)
    assert len(denied) == len(approved) == 1


def test_get_missing_application_404(client):
    assert client.get("/applications/99999").status_code == 404
