import polars as pl
import datetime
from dateutil.relativedelta import relativedelta

def compute_dividend_cagr(df: pl.DataFrame, years: int, grace_months: int = 3) -> float:
    if df.schema["date"] != pl.String:
        df = df.with_columns(pl.col("date").cast(pl.String))
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")).sort("date")

    if df.height < 2:
        return 0.0

    end_date = df[-1, "date"]
    end_div = df[-1, "dividend"]

    for y in range(years, 1, -1):
        target_date = end_date - datetime.timedelta(days=365 * y)
        lower_bound = target_date - relativedelta(months=grace_months)
        upper_bound = target_date + relativedelta(months=grace_months)

        start_row = df.filter(
            (pl.col("date") >= lower_bound) &
            (pl.col("date") <= upper_bound)
        )

        if not start_row.is_empty():
            start_div = start_row[-1, "dividend"]
            if start_div > 0 and end_div > 0:
                return (end_div / start_div) ** (1 / y) - 1

        print(f"⚠️ No dividend data found for {y}Y CAGR within ±{grace_months} months of {target_date}")

    return 0.0


def compute_yield_vs_median(df: pl.DataFrame, lookback_years: int, grace_days: int = 45) -> float:
    if df.height < 2 or "dividendYield" not in df.columns or "date" not in df.columns:
        return 0.0

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
        return 0.0

    current = filtered[-1, "dividendYield"]
    median = filtered.select(pl.col("dividendYield").median()).item()

    return 0.0 if median == 0 else (current - median) / median
