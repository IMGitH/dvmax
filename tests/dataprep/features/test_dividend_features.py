from src.dataprep.features.dividend_features import (
    compute_yield_vs_median
)
import polars as pl
import pytest

def test_compute_yield_vs_median():
    df = pl.DataFrame({
        "date": [
            "2018-01-01", "2019-01-01", "2020-01-01",
            "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"
        ],
        "dividendYield": [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    end_date = df[-1, "date"]  # 2024-01-01
    lookback_years = 6
    start_date = end_date.replace(year=end_date.year - lookback_years)

    filtered = df.filter((pl.col("date") >= start_date) & (pl.col("dividendYield") > 0)).sort("date")

    current = filtered[-1, "dividendYield"]
    median_yield = filtered.select(pl.col("dividendYield").median()).item()
    expected = (current - median_yield) / median_yield

    result = compute_yield_vs_median(df, lookback_years=lookback_years)

    print("\n=== test_compute_yield_vs_median ===")
    print(f"Lookback start: {start_date}")
    print("Filtered yields:")
    print(filtered.select(["date", "dividendYield"]))
    print(f"Current yield: {current}")
    print(f"Median yield: {median_yield}")
    print(f"Expected relative diff: {expected:.6f}")
    print(f"Computed relative diff: {result:.6f}")

    assert result == pytest.approx(expected, rel=1e-4)
    