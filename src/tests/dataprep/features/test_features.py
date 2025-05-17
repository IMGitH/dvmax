import pytest
import polars as pl
import datetime
from src.dataprep.features import (
    compute_6m_return, 
    compute_12m_return,
    compute_volatility,
    compute_max_drawdown, 
    compute_sma_delta,
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

def test_compute_12m_return():
    df = ensure_date_column(pl.DataFrame({
        "date": ["2023-07-01", "2024-07-01"],
        "close": [100, 130]
    }))
    as_of_date = datetime.date(2024, 7, 1)
    result = compute_12m_return(df, as_of_date=as_of_date)
    print("\n=== test_compute_12m_return ===")
    print(df)
    print(f"As of date: {as_of_date}")
    print(f"Computed 12M return: {result:.4f}")
    assert pytest.approx(result, rel=1e-2) == 0.3

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

def test_compute_max_drawdown_basic():
    df = pl.DataFrame({
        "date": ["2023-01-01", "2023-06-01", "2023-12-31"],
        "close": [100, 150, 90]  # drawdown from 150 to 90
    }).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )

    print("\n=== Test Data ===")
    print(df)

    result = compute_max_drawdown(df)

    expected = (150 - 90) / 150
    print(f"Computed Drawdown: {result}, Expected: {expected}")
    assert pytest.approx(result, rel=1e-4) == expected


def test_compute_max_drawdown_larger_range():
    df = pl.DataFrame({
        "date": [
            "2019-01-01", "2020-01-01", "2020-06-01",
            "2021-01-01", "2021-06-01", "2022-01-01",
            "2023-01-01", "2023-06-01", "2023-12-31"
        ],
        "close": [
            50,   # early low
            100,  # rise
            110,  # peak
            95,   # small dip
            120,  # new peak
            80,   # big drop -> drawdown from 120 to 80
            90,   # partial recovery
            130,  # new high
            125   # small drop from recent peak
        ]
    }).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )

    print("\n=== Larger Range Test Data ===")
    print(df)

    result = compute_max_drawdown(df)

    # Expect drawdown from 120 -> 80
    expected = (80 - 120) / 120
    print(f"Computed Drawdown: {result}, Expected: {abs(expected)}")
    assert pytest.approx(result, rel=1e-4) == abs(expected)


def test_compute_sma_delta():
    df = ensure_date_column(pl.DataFrame({
        "date": [f"2024-01-{i+1:02d}" for i in range(200)],
        "close": [100 + i * 0.1 for i in range(200)]  # upward trend
    }))
    result = compute_sma_delta(df, short=50, long=200)
    assert result > 0  # upward SMA crossover
