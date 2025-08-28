import logging
from datetime import datetime, date
from functools import lru_cache
from dateutil.relativedelta import relativedelta
from typing import Literal

import polars as pl
import yfinance as yf

from src.dataprep.fetcher.ticker_params.splits import fetch_splits
from src.dataprep.fetcher.utils import default_date_range
from src.dataprep.fetcher.client import fmp_client

# ---- warn-once registries ----
_warned_no_dividends: set[str] = set()
_warned_no_yield: set[tuple[str, str]] = set()


def _empty_dividends_df() -> pl.DataFrame:
    return pl.DataFrame({
        "date": pl.Series([], dtype=pl.Date),
        "dividend": pl.Series([], dtype=pl.Float64),
    })


def adjust_dividends_with_splits(div_df: pl.DataFrame, split_df: pl.DataFrame) -> pl.DataFrame:
    for row in split_df.iter_rows(named=True):
        split_date, ratio = row["date"], row["split_ratio"]
        div_df = div_df.with_columns(
            pl.when(pl.col("date") < split_date)
              .then(pl.col("dividend") / ratio)
              .otherwise(pl.col("dividend"))
              .alias("dividend")
        )
    return div_df


def _warn_once(key: str, message: str):
    if key not in _warned_no_dividends:
        logging.warning(message)
        _warned_no_dividends.add(key)


# ---------- CACHES ----------

@lru_cache(maxsize=4096)
def _cached_splits(ticker: str, mode: str = "yfinance") -> pl.DataFrame:
    return fetch_splits(ticker, mode=mode)

@lru_cache(maxsize=4096)
def _cached_dividends_yf_full(ticker: str) -> pl.DataFrame:
    """Fetch full YF dividends once; empty frame if none."""
    yf_tkr = yf.Ticker(ticker)
    dividends = yf_tkr.dividends
    if dividends.empty:
        return _empty_dividends_df()
    return pl.DataFrame({
        "date": dividends.index,
        "dividend": dividends.values
    }).with_columns(pl.col("date").cast(pl.Date))

@lru_cache(maxsize=4096)
def _cached_dividends_fmp_full(ticker: str) -> pl.DataFrame:
    """Fetch a very wide window from FMP once; empty frame if none."""
    # wide window once; slice later
    resp = fmp_client.fetch(
        f"historical-price-full/stock_dividend/{ticker}",
        {"from": "1980-01-01", "to": datetime.today().strftime("%Y-%m-%d")}
    )
    data = resp.get("historical", [])
    if not data:
        return _empty_dividends_df()
    return (
        pl.DataFrame(data)
        .select(["date", "dividend"])
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    )


def _slice(df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    if df.is_empty():
        return df
    return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))


# ---------- PUBLIC API ----------

def fetch_dividends(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int | None = None,
    grace_quarters: int = 1,
    mode: Literal["fmp", "yfinance"] = "yfinance",
    fallback_to_fmp: bool = False
) -> pl.DataFrame:
    """
    Fetch dividends with caching (full-history once per ticker per source),
    window slicing, warn-once logging, and split adjustment.
    """

    # 1) Window (+ grace)
    start_date, end_date = default_date_range(
        lookback_years, start_date, end_date, quarter_mode=True
    )
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d").date()
    grace    = relativedelta(months=3 * grace_quarters)
    window_start = start_dt - grace
    window_end   = end_dt   + grace

    # 2) Primary source
    if mode == "yfinance":
        df_full = _cached_dividends_yf_full(ticker)
        if df_full.is_empty():
            _warn_once(f"yf:{ticker}", f"No dividend data for {ticker} in yfinance.")
            if fallback_to_fmp:
                df_full = _cached_dividends_fmp_full(ticker)
                if df_full.is_empty():
                    _warn_once(f"fmp:{ticker}", f"No dividend data for {ticker} in FMP.")
                    return _empty_dividends_df()
            else:
                return _empty_dividends_df()
        df = _slice(df_full, window_start, window_end)
        splits = _cached_splits(ticker, "yfinance")
        return adjust_dividends_with_splits(df, splits)

    if mode == "fmp":
        df_full = _cached_dividends_fmp_full(ticker)
        if df_full.is_empty():
            _warn_once(f"fmp:{ticker}", f"No dividend data for {ticker} in FMP.")
            return _empty_dividends_df()
        df = _slice(df_full, window_start, window_end)
        splits = _cached_splits(ticker, "yfinance")  # splits via YF is fine
        return adjust_dividends_with_splits(df, splits)

    raise ValueError(f"Unknown mode '{mode}' in fetch_dividends()")
