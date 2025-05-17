import polars as pl
import datetime

def compute_dividend_cagr(df: pl.DataFrame, years: int) -> float:
    if df.schema["date"] != pl.String:
        df = df.with_columns(pl.col("date").cast(pl.String))
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    df = df.sort("date")
    if df.height < 2:
        return 0.0

    end_date = df[-1, "date"]
    start_date = end_date - datetime.timedelta(days=years * 365)

    # Find dividend closest to start
    start_row = df.filter(pl.col("date") <= start_date)
    if start_row.is_empty():
        return 0.0
    start_div = start_row[-1, "dividend"]
    end_div = df[-1, "dividend"]

    if start_div <= 0 or end_div <= 0:
        return 0.0

    return (end_div / start_div) ** (1 / years) - 1

'''
This function calculates the relative difference between the current dividend yield
and the historical median over a specified lookback period in years.
'''
def compute_yield_vs_median(df: pl.DataFrame, lookback_years: int) -> float:
    if df.height < 2 or "dividendYield" not in df.columns or "date" not in df.columns:
        return 0.0

    # Convert date if it's a string
    if df.schema["date"] == pl.Utf8:
        df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    df = df.drop_nulls("date").sort("date")

    end_date = df[-1, "date"]
    start_date = end_date.replace(year=end_date.year - lookback_years)

    filtered = df.filter(
        (pl.col("date") >= start_date) & (pl.col("dividendYield") > 0)
    ).sort("date")

    filtered = filtered.drop_nulls("dividendYield")
    if filtered.is_empty():
        return 0.0

    current = filtered[-1, "dividendYield"]
    median = filtered.select(pl.col("dividendYield").median()).item()

    if median == 0:
        return 0.0

    return (current - median) / median