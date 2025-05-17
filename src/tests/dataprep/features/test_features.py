import pytest
import polars as pl
import datetime
from src.dataprep.features import (
    compute_6m_return, 
    compute_12m_return,
    compute_volatility,
    compute_max_drawdown, 
    ensure_date_column
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

def test_compute_6m_return_precise_date():
    df = pl.DataFrame({
        "date": ["2024-11-17", "2025-05-17"],
        "close": [228.0, 211.26]
    })
    df = ensure_date_column(df)
    result = compute_6m_return(df, as_of_date=datetime.date(2025, 5, 17))
    expected = (211.26 - 228.0) / 228.0
    assert pytest.approx(result, rel=1e-4) == expected

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
