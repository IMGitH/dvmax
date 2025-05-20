import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Literal

import polars as pl
import yfinance as yf

from src.dataprep.fetcher.splits import fetch_splits
from src.dataprep.fetcher.base import FMPClient
from src.dataprep.fetcher.utils import default_date_range

def _empty_dividends_df() -> pl.DataFrame:
    """
    Return an empty dividends DataFrame with the correct schema,
    so downstream .select()/.project() calls won't fail.
    """
    return pl.DataFrame({
        "date": pl.Series([], dtype=pl.Date),
        "dividend": pl.Series([], dtype=pl.Float64),
    })

def adjust_dividends_with_splits(
    div_df: pl.DataFrame,
    split_df: pl.DataFrame
) -> pl.DataFrame:
    for row in split_df.iter_rows(named=True):
        split_date, ratio = row["date"], row["split_ratio"]
        div_df = div_df.with_columns(
            pl.when(pl.col("date") < split_date)
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
    mode: Literal["fmp", "yfinance"] = "yfinance",
    fallback_to_fmp: bool = True
) -> pl.DataFrame:
    # 1. Compute date window + grace
    start_date, end_date = default_date_range(
        lookback_years, start_date, end_date, quarter_mode=True
    )
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d").date()
    grace    = relativedelta(months=3 * grace_quarters)
    window_start = start_dt - grace
    window_end   = end_dt   + grace

    # 2. YFinance path
    if mode == "yfinance":
        yf_tkr = yf.Ticker(ticker)
        dividends = yf_tkr.dividends
        if dividends.empty:
            logging.warning(f"No dividend data for {ticker} in yfinance.")
            if fallback_to_fmp:
                return fetch_dividends(
                    ticker, start_date, end_date,
                    lookback_years, grace_quarters,
                    mode="fmp", fallback_to_fmp=False
                )
            return _empty_dividends_df()

        df = (
            pl.DataFrame({
                "date":     dividends.index,
                "dividend": dividends.values
            })
            .with_columns(pl.col("date").cast(pl.Date))
            .filter(
                (pl.col("date") >= pl.lit(window_start)) &
                (pl.col("date") <= pl.lit(window_end))
            )
        )
        splits = fetch_splits(ticker, mode="yfinance")
        return adjust_dividends_with_splits(df, splits)

    # 3. FMP path
    if mode == "fmp":
        client = FMPClient()
        resp = client.fetch(
            f"historical-price-full/stock_dividend/{ticker}",
            {"from": start_date, "to": end_date}
        )
        data = resp.get("historical", [])
        if not data:
            logging.warning(f"No dividend data for {ticker} in FMP.")
            return _empty_dividends_df()

        df = (
            pl.DataFrame(data)
              .select(["date", "dividend"])
              .with_columns(
                  pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
              )
              .filter(
                  (pl.col("date") >= pl.lit(window_start)) &
                  (pl.col("date") <= pl.lit(window_end))
              )
        )
        splits = fetch_splits(ticker, mode="yfinance")
        return adjust_dividends_with_splits(df, splits)

    # 4. Bad mode
    raise ValueError(f"Unknown mode '{mode}' in fetch_dividends()")
