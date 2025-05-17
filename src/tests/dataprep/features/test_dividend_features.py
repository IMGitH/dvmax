from src.dataprep.features.dividend_features import (
    compute_dividend_cagr,
    compute_yield_vs_median
)
import polars as pl
import datetime
import pytest

def test_compute_dividend_cagr_basic():
    df = pl.DataFrame({
        "date": ["2019-01-01", "2020-01-01", "2024-01-01"],
        "dividend": [1.0, 1.1, 2.0]
    })
    result = compute_dividend_cagr(df, years=5)
    expected = (2.0 / 1.0) ** (1 / 5) - 1
    print("\n=== test_compute_dividend_cagr_basic ===")
    print(f"Start dividend: 1.0")
    print(f"End dividend: 2.0")
    print(f"Years: 5")
    print(f"Expected CAGR: {expected:.6f}")
    print(f"Computed CAGR: {result:.6f}")
    assert result == pytest.approx(expected, rel=1e-4)

def test_compute_yield_vs_median():
    df = pl.DataFrame({
        "date": ["2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
        "dividendYield": [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    })
    df = df.sort("date").with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    end_date = df[-1, "date"]
    start_date = end_date - datetime.timedelta(days=5 * 365)
    filtered = df.filter((pl.col("date") >= start_date) & (pl.col("dividendYield") > 0)).sort("date")
    current = filtered[-1, "dividendYield"]
    median_yield = filtered.select(pl.col("dividendYield").median()).item()
    expected = (current - median_yield) / median_yield
    result = compute_yield_vs_median(df, lookback_years=5)
    print("\n=== test_compute_yield_vs_median ===")
    print(f"Lookback start: {start_date}")
    print("Filtered yields:")
    print(filtered.select(["date", "dividendYield"]))
    print(f"Current yield: {current}")
    print(f"Median yield: {median_yield}")
    print(f"Expected relative diff: {expected:.6f}")
    print(f"Computed relative diff: {result:.6f}")
    assert result == pytest.approx(expected, rel=1e-4)