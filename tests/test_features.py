from pathlib import Path

import pytest

from src.data.clean import clean_data
from src.data.load import load_raw_data
from src.features.engineer import engineer_features

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "give_me_some_credit_sample.csv"

_AGE_BIN_COLUMNS = [
    "age_bin_18-24",
    "age_bin_25-34",
    "age_bin_35-44",
    "age_bin_45-54",
    "age_bin_55-64",
    "age_bin_65+",
]


@pytest.fixture
def engineered_df():
    return engineer_features(clean_data(load_raw_data(FIXTURE_PATH)))


def test_engineer_features_adds_expected_columns(engineered_df):
    expected = {
        "total_past_due_count",
        "has_past_due",
        "income_per_dependent",
        "has_dependents",
        "credit_lines_per_year_of_age",
        *_AGE_BIN_COLUMNS,
    }
    assert expected.issubset(engineered_df.columns)


def test_total_past_due_count_sums_the_three_delinquency_columns(engineered_df):
    row = engineered_df.iloc[0]  # id=1: 30-59=0, 60-89=0, 90+=0
    assert row["total_past_due_count"] == 0

    row = engineered_df.iloc[1]  # id=2: 30-59=2, 60-89=1, 90+=1
    assert row["total_past_due_count"] == 4


def test_has_past_due_flag_matches_count(engineered_df):
    assert (
        (engineered_df["has_past_due"] == 1)
        == (engineered_df["total_past_due_count"] > 0)
    ).all()


def test_income_per_dependent_divides_by_dependents_plus_one(engineered_df):
    row = engineered_df.iloc[0]  # id=1: income=5000, dependents=2
    assert row["income_per_dependent"] == pytest.approx(5000 / 3)


def test_has_dependents_flag(engineered_df):
    row = engineered_df.iloc[9]  # id=10: dependents=0
    assert row["has_dependents"] == 0

    row = engineered_df.iloc[0]  # id=1: dependents=2
    assert row["has_dependents"] == 1


def test_age_bin_one_hot_is_mutually_exclusive(engineered_df):
    bin_sums = engineered_df[_AGE_BIN_COLUMNS].sum(axis=1)
    assert (bin_sums == 1).all()


def test_age_bin_assignment_for_known_ages(engineered_df):
    row = engineered_df.iloc[9]  # id=10: age=22
    assert row["age_bin_18-24"] == 1

    row = engineered_df.iloc[6]  # id=7: age=70
    assert row["age_bin_65+"] == 1


def test_no_missing_values_in_engineered_numeric_columns(engineered_df):
    numeric_cols = [
        "total_past_due_count",
        "has_past_due",
        "income_per_dependent",
        "has_dependents",
        "credit_lines_per_year_of_age",
    ]
    assert engineered_df[numeric_cols].isna().sum().sum() == 0
