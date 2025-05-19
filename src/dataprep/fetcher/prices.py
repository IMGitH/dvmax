from typing import Literal
import polars as pl
from datetime import datetime, timedelta
import yfinance as yf
from src.dataprep.fetcher.base import FMPClient
from src.dataprep.fetcher.utils import default_date_range

def fetch_prices(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int | None = None,
    grace_days: int = 7,
    mode: Literal["fmp", "yfinance"] = "yfinance"
) -> pl.DataFrame:
    start_date, end_date = default_date_range(lookback_years, start_date, end_date)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    if mode == "yfinance":
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(start=start_date, end=end_date)
        if hist.empty:
            raise RuntimeError(f"No price data from yfinance for {ticker}")
        df = pl.DataFrame({
            "date": hist.index.to_list(),
            "close": hist["Close"].to_list()
        }).with_columns(pl.col("date").cast(pl.Date))
        return df

    client = FMPClient()
    data = client.fetch(f"historical-price-full/{ticker}", {"from": start_date, "to": end_date}).get("historical", [])
    if not data:
        raise RuntimeError(f"No price data from FMP for {ticker}")
    df = pl.DataFrame(data).select(["date", "close"]).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )

    actual_start = df.select(pl.col("date").min()).item()
    actual_end = df.select(pl.col("date").max()).item()

    if actual_start > start_dt + timedelta(days=grace_days):
        raise RuntimeError(f"Data for {ticker} starts at {actual_start}, which is more than {grace_days} days after requested start {start_date}.")
    if actual_end < end_dt - timedelta(days=grace_days):
        raise RuntimeError(f"Data for {ticker} ends at {actual_end}, too far before requested {end_date}.")

    return df
