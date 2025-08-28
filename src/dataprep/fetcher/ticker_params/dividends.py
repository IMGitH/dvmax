import os
import logging
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dateutil.relativedelta import relativedelta
from typing import Literal, Optional

import polars as pl
import yfinance as yf

from src.dataprep.fetcher.ticker_params.splits import fetch_splits
from src.dataprep.fetcher.utils import default_date_range
from src.dataprep.fetcher.client import fmp_client

# =========================
# Local, self-contained logging (no global/basicConfig)
# =========================
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()  # stdout
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
# Level controlled by env; defaults to INFO. No propagation to avoid dupes.
logger.setLevel(getattr(logging, os.getenv("DIV_LOG_LEVEL", "INFO").upper(), logging.INFO))
logger.propagate = False

# =========================
# Config (no caches)
# =========================
DIV_SOURCE_DEFAULT = os.getenv("DIV_SOURCE", "fmp")            # "fmp" | "yfinance"
YF_TIMEOUT_SEC = float(os.getenv("YF_TIMEOUT_SEC", "15"))
FMP_TIMEOUT_SEC = float(os.getenv("FMP_TIMEOUT_SEC", "15"))

_executor = ThreadPoolExecutor(max_workers=8)

# =========================
# Helpers
# =========================
def _empty_dividends_df() -> pl.DataFrame:
    return pl.DataFrame({
        "date": pl.Series([], dtype=pl.Date),
        "dividend": pl.Series([], dtype=pl.Float64),
    })


def adjust_dividends_with_splits(div_df: pl.DataFrame, split_df: pl.DataFrame) -> pl.DataFrame:
    """Back-adjust historical dividends for stock splits (pre-split dates are divided by ratio)."""
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


def _slice(df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    if df.is_empty():
        return df
    return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))

# =========================
# Source fetchers (full history, timeboxed)
# =========================
def _yf_full_dividends(ticker: str) -> pl.DataFrame:
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
        return _with_timeout(_call, YF_TIMEOUT_SEC)
    except TimeoutError:
        logger.error("yfinance dividends timed out for %s after %.1fs.", ticker, YF_TIMEOUT_SEC)
        raise
    except Exception as e:
        logger.error("yfinance dividends error for %s: %s", ticker, e)
        raise


def _fmp_full_dividends(ticker: str) -> pl.DataFrame:
    def _call():
        # fmp_client.fetch should raise on non-200 OR return a body we can inspect
        resp = fmp_client.fetch(
            f"historical-price-full/stock_dividend/{ticker}",
            {"from": "1980-01-01", "to": datetime.today().strftime("%Y-%m-%d")}
        )

        # If fetch returns an error envelope instead of raising:
        if isinstance(resp, dict):
            msg = (resp.get("Error Message") or resp.get("Note") or "").strip()
            if "Too Many Requests" in msg or "429" in msg:
                # hard-fail on rate limit (your requirement)
                raise RuntimeError(f"FMP API rate limit hit for {ticker}: {msg}")

        data = resp.get("historical", [])
        if not data:
            return _empty_dividends_df()

        return (
            pl.DataFrame(data)
            .select(["date", "dividend"])
            .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        )

    try:
        return _with_timeout(_call, FMP_TIMEOUT_SEC)
    except TimeoutError:
        logger.error("FMP dividends timed out for %s after %.1fs.", ticker, FMP_TIMEOUT_SEC)
        raise
    except RuntimeError as e:
        # explicit 429 (or similar) -> fail hard
        logger.error("%s", e)
        raise
    except Exception as e:
        logger.error("FMP dividends error for %s: %s", ticker, e)
        raise

# =========================
# Public API
# =========================
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
    Timeboxed dividends fetcher with module-local logging (no caches):
      - Logging configured only for this module; level via DIV_LOG_LEVEL (default INFO).
      - Fails hard on FMP 429 (rate limit) and on unexpected errors/timeouts.
      - No in-memory or disk caches.
      - Optional early skip for non-payers via profile_last_div<=0.
    """
    # Optional fast-skip if you pass it (doesn't affect logs elsewhere)
    if profile_last_div is not None and float(profile_last_div or 0.0) <= 0.0:
        logger.info("Skipping dividends fetch for non-payer %s (profile_last_div<=0).", ticker)
        return _empty_dividends_df()

    # Window (+ grace)
    start_date, end_date = default_date_range(
        lookback_years, start_date, end_date, quarter_mode=True
    )
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    grace = relativedelta(months=3 * grace_quarters)
    window_start = start_dt - grace
    window_end = end_dt + grace

    # Primary source
    if mode == "yfinance":
        try:
            df_full = _yf_full_dividends(ticker)
        except Exception:
            if fallback_to_fmp:
                df_full = _fmp_full_dividends(ticker)
            else:
                raise
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
        splits = fetch_splits(ticker, mode="yfinance")
        return adjust_dividends_with_splits(df, splits)

    if mode == "fmp":
        df_full = _fmp_full_dividends(ticker)
        if df_full.is_empty():
            logger.warning("No dividend data for %s in FMP.", ticker)
            return _empty_dividends_df()
        df = _slice(df_full, window_start, window_end)
        # YF splits are fine here too
        splits = fetch_splits(ticker, mode="yfinance")
        return adjust_dividends_with_splits(df, splits)

    raise ValueError(f"Unknown mode '{mode}' in fetch_dividends()")
    
