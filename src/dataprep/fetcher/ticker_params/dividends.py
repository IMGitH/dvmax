import os
import logging
from datetime import datetime, date
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dateutil.relativedelta import relativedelta
from typing import Literal, Optional

import polars as pl
import yfinance as yf

from src.dataprep.fetcher.ticker_params.splits import fetch_splits
from src.dataprep.fetcher.utils import default_date_range
from src.dataprep.fetcher.client import fmp_client

logger = logging.getLogger(__name__)  # module logger (no global config)

# ---------------- CONFIG ----------------
DIV_SOURCE_DEFAULT = os.getenv("DIV_SOURCE", "fmp")          # default to FMP
YF_TIMEOUT_SEC = float(os.getenv("YF_TIMEOUT_SEC", "15"))     # per your ask
FMP_TIMEOUT_SEC = float(os.getenv("FMP_TIMEOUT_SEC", "15"))
CACHE_DIR = os.getenv("DIV_CACHE_DIR", ".cache/dividends")
os.makedirs(CACHE_DIR, exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=8)


# ---------------- HELPERS ----------------
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


def _with_timeout(fn, timeout_s: float):
    fut = _executor.submit(fn)
    return fut.result(timeout=timeout_s)


def _cache_path(ticker: str, source: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_{source}.parquet")


def _load_cache(ticker: str, source: str) -> Optional[pl.DataFrame]:
    p = _cache_path(ticker, source)
    if os.path.exists(p):
        try:
            return pl.read_parquet(p)
        except Exception as e:
            logger.debug("Failed reading cache %s: %s", p, e)
    return None


def _save_cache(ticker: str, source: str, df: pl.DataFrame):
    try:
        if not df.is_empty():
            df.write_parquet(_cache_path(ticker, source))
    except Exception as e:
        logger.debug("Failed writing cache for %s/%s: %s", ticker, source, e)


# ---------------- SPLITS CACHE ----------------
@lru_cache(maxsize=4096)
def _cached_splits(ticker: str, mode: str = "yfinance") -> pl.DataFrame:
    return fetch_splits(ticker, mode=mode)


# ---------------- DIVIDENDS FETCHERS ----------------
def _yf_full_dividends(ticker: str) -> pl.DataFrame:
    cached = _load_cache(ticker, "yf")
    if cached is not None:
        return cached

    def _call():
        tk = yf.Ticker(ticker)
        div = tk.dividends
        if div.empty:
            return _empty_dividends_df()
        return (
            pl.DataFrame({"date": div.index, "dividend": div.values})
            .with_columns(pl.col("date").cast(pl.Date))
        )

    try:
        df = _with_timeout(_call, YF_TIMEOUT_SEC)
    except TimeoutError:
        logger.warning("yfinance dividends timed out for %s after %.1fs.", ticker, YF_TIMEOUT_SEC)
        return _empty_dividends_df()
    except Exception as e:
        logger.warning("yfinance dividends error for %s: %s", ticker, e)
        return _empty_dividends_df()

    _save_cache(ticker, "yf", df)
    return df


def _fmp_full_dividends(ticker: str) -> pl.DataFrame:
    cached = _load_cache(ticker, "fmp")
    if cached is not None:
        return cached

    def _call():
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

    try:
        df = _with_timeout(_call, FMP_TIMEOUT_SEC)
    except TimeoutError:
        logger.warning("FMP dividends timed out for %s after %.1fs.", ticker, FMP_TIMEOUT_SEC)
        return _empty_dividends_df()
    except Exception as e:
        logger.warning("FMP dividends error for %s: %s", ticker, e)
        return _empty_dividends_df()

    _save_cache(ticker, "fmp", df)
    return df


def _slice(df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    if df.is_empty():
        return df
    return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))


# ---------------- PUBLIC API ----------------
def fetch_dividends(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int | None = None,
    grace_quarters: int = 1,
    mode: Literal["fmp", "yfinance"] = DIV_SOURCE_DEFAULT,
    fallback_to_fmp: bool = False,
    profile_last_div: Optional[float] = None,
) -> pl.DataFrame:
    """
    Timeboxed, cached dividends fetcher.
    - Timeout: 15s per source call (configurable via env).
    - No global logging changes; uses module logger only.
    - No warn-once throttling (log volume unchanged).
    - Disk cache for full history; windows slice locally.
    - Optional early skip if profile_last_div <= 0 (won't affect other logs).
    """

    # optional: early skip if you pass it; omit to keep exact behavior
    if profile_last_div is not None and float(profile_last_div or 0.0) <= 0.0:
        return _empty_dividends_df()

    start_date, end_date = default_date_range(
        lookback_years, start_date, end_date, quarter_mode=True
    )
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    grace = relativedelta(months=3 * grace_quarters)
    window_start = start_dt - grace
    window_end = end_dt + grace

    if mode == "yfinance":
        df_full = _yf_full_dividends(ticker)
        if df_full.is_empty():
            logger.warning("No dividend data for %s in yfinance.", ticker)
            if fallback_to_fmp:
                df_full = _fmp_full_dividends(ticker)
                if df_full.is_empty():
                    logger.warning("No dividend data for %s in FMP.", ticker)
                    return _empty_dividends_df()
            else:
                return _empty_dividends_df()
        df = _slice(df_full, window_start, window_end)
        splits = _cached_splits(ticker, "yfinance")
        return adjust_dividends_with_splits(df, splits)

    if mode == "fmp":
        df_full = _fmp_full_dividends(ticker)
        if df_full.is_empty():
            logger.warning("No dividend data for %s in FMP.", ticker)
            return _empty_dividends_df()
        df = _slice(df_full, window_start, window_end)
        splits = _cached_splits(ticker, "yfinance")
        return adjust_dividends_with_splits(df, splits)

    raise ValueError(f"Unknown mode '{mode}' in fetch_dividends()")
