from typing import Literal
from datetime import datetime
from dateutil.relativedelta import relativedelta
import polars as pl
import yfinance as yf
from src.dataprep.fetcher.base import FMPClient
from src.dataprep.fetcher.utils import default_date_range
from src.dataprep.fetcher.splits import fetch_splits

def adjust_dividends_with_splits(div_df: pl.DataFrame, split_df: pl.DataFrame) -> pl.DataFrame:
    for row in split_df.iter_rows(named=True):
        date, ratio = row["date"], row["split_ratio"]
        div_df = div_df.with_columns(
            pl.when(pl.col("date") < date)
            .then(pl.col("dividend") / ratio)
            .otherwise(pl.col("dividend"))
            .alias("dividend")
        )
    return div_df

def fetch_dividends(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int | None = None,
    grace_quarters: int = 1,
    mode: Literal["fmp", "yfinance"] = "yfinance"
) -> pl.DataFrame:
    start_date, end_date = default_date_range(lookback_years, start_date, end_date, quarter_mode=True)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    grace_period = relativedelta(months=3 * grace_quarters)
    max_acceptable_start = start_dt + grace_period

    if mode == "yfinance":
        yf_ticker = yf.Ticker(ticker)
        dividends = yf_ticker.dividends
        if dividends.empty:
            raise RuntimeError("No dividend data from yfinance.")
        df_yf = (
            pl.DataFrame({"date": dividends.index, "dividend": dividends.values})
            .with_columns(pl.col("date").cast(pl.Date))
            .filter((pl.col("date") >= pl.lit(start_dt)) & (pl.col("date") <= pl.lit(end_dt)))
        )
        splits = fetch_splits(ticker, mode="yfinance")
        return adjust_dividends_with_splits(df_yf, splits)

    client = FMPClient()
    response = client.fetch(f"historical-price-full/stock_dividend/{ticker}", {"from": start_date, "to": end_date})
    data = response.get("historical", [])
    if not data:
        raise RuntimeError("No dividend data from FMP.")
    df_fmp = pl.DataFrame(data).select(["date", "dividend"]).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )
    splits = fetch_splits(ticker, mode="yfinance")
    return adjust_dividends_with_splits(df_fmp, splits)
