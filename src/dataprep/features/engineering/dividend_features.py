import polars as pl
import datetime
import numpy as np

def compute_yield_vs_median(df: pl.DataFrame, lookback_years: int, grace_days: int = 90) -> float:
    if df.height < 2 or "dividendYield" not in df.columns or "date" not in df.columns:
        return np.nan

    if df.schema["date"] == pl.Utf8:
        df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    df = df.drop_nulls("date").sort("date")

    end_date = df[-1, "date"]
    raw_start = end_date.replace(year=end_date.year - lookback_years)
    grace = datetime.timedelta(days=grace_days)

    filtered = df.filter(
        (pl.col("date") >= raw_start - grace) &
        (pl.col("date") <= end_date) &
        (pl.col("dividendYield") > 0)
    ).drop_nulls("dividendYield").sort("date")

    if filtered.is_empty():
        print(f"⚠️ No yield data within {lookback_years}Y + {grace_days}d grace before {end_date}")
        return np.nan

    current = filtered[-1, "dividendYield"]
    median = filtered.select(pl.col("dividendYield").median()).item()

    return 0.0 if median == 0 else (current - median) / median
