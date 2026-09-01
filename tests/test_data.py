from pathlib import Path

import pandas as pd
import pytest

from src.data.clean import clean_data
from src.data.load import load_raw_data
from src.data.schema import RAW_COLUMNS

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "give_me_some_credit_sample.csv"


@pytest.fixture
def raw_df():
    return load_raw_data(FIXTURE_PATH)


def test_load_raw_data_returns_expected_columns(raw_df):
    assert list(raw_df.columns) == RAW_COLUMNS


def test_load_raw_data_drops_unnamed_index_column(raw_df):
    assert "Unnamed: 0" not in raw_df.columns


def test_load_raw_data_row_count(raw_df):
    assert len(raw_df) == 12


def test_load_raw_data_raises_on_missing_column(tmp_path):
    df = load_raw_data(FIXTURE_PATH).drop(columns=["age"])
    bad_path = tmp_path / "missing_column.csv"
    df.to_csv(bad_path, index=False)

    with pytest.raises(ValueError, match="age"):
        load_raw_data(bad_path)


def test_clean_data_fixes_invalid_zero_age(raw_df):
    cleaned = clean_data(raw_df)
    assert (cleaned["age"] > 0).all()


def test_clean_data_removes_past_due_sentinel_values(raw_df):
    cleaned = clean_data(raw_df)
    past_due_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for col in past_due_cols:
        assert not cleaned[col].isin([96, 98]).any()


def test_clean_data_flags_and_imputes_missing_income(raw_df):
    cleaned = clean_data(raw_df)
    assert cleaned["MonthlyIncome"].isna().sum() == 0
    # Fixture rows 4 (id=3) and 11 (id=10) have blank MonthlyIncome.
    assert cleaned["income_missing"].sum() == 2


def test_clean_data_flags_and_imputes_missing_dependents(raw_df):
    cleaned = clean_data(raw_df)
    assert cleaned["NumberOfDependents"].isna().sum() == 0
    # Fixture row 5 (id=4) has a blank NumberOfDependents.
    assert cleaned["dependents_missing"].sum() == 1


def test_clean_data_does_not_mutate_input(raw_df):
    original = raw_df.copy(deep=True)
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, original)
