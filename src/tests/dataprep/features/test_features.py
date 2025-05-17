import pytest
import polars as pl
import datetime
from src.dataprep.features import (
    compute_6m_return, 
    compute_volatility, 
    ensure_date_column,
    build_fundamental_features
)

def test_compute_6m_return():
    df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01", "2024-07-01"],
        "close": [100, 120]
    }))
    as_of_date = datetime.date(2024, 7, 1)
    result = compute_6m_return(df, as_of_date=as_of_date)
    print("\n=== test_compute_6m_return ===")
    print(df)
    print(f"As of date: {as_of_date}")
    print(f"Computed 6M return: {result:.4f}")
    assert isinstance(result, float)
    assert pytest.approx(result, rel=1e-2) == 0.2


def test_compute_volatility():
    df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "close": [100, 101, 102]
    }))
    result = compute_volatility(df)
    print("\n=== test_compute_volatility ===")
    print(df)
    print(f"Computed volatility: {result:.6f}")
    assert isinstance(result, float)
    assert result >= 0


def test_build_fundamental_features_returns_expected_metrics():
    balance_df = pl.DataFrame({
        "date": ["2023-12-31"],
        "totalDebt": [1000],
        "cashAndShortTermInvestments": [200]
    })
    income_df = pl.DataFrame({
        "date": ["2023-12-31"],
        "incomeBeforeTax": [400],
        "interestExpense": [100]
    })
    result = build_fundamental_features(balance_df, income_df)
    expected_net_debt_to_ebitda = (1000 - 200) / 400
    expected_ebit_interest_cover = 400 / 100
    print("\n=== test_build_fundamental_features_returns_expected_metrics ===")
    print("Balance DataFrame:")
    print(balance_df)
    print("Income DataFrame:")
    print(income_df)
    print("Resulting Metrics:")
    print(result)
    print(f"Expected net_debt_to_ebitda: {expected_net_debt_to_ebitda:.2f}")
    print(f"Expected ebit_interest_cover: {expected_ebit_interest_cover:.2f}")
    assert result.height == 1
    assert "net_debt_to_ebitda" in result.columns
    assert "ebit_interest_cover" in result.columns
    assert result["net_debt_to_ebitda"][0] == pytest.approx(expected_net_debt_to_ebitda)
    assert result["ebit_interest_cover"][0] == pytest.approx(expected_ebit_interest_cover)
