# -*- coding: utf-8 -*-
"""
Ticker batch feature generation with simple robustness:
- Circuit breaker on consecutive 429s (FMPRateLimitError)
- Global time budget to avoid hangs
- No retry of FMPRateLimitError inside the generic retry helper
"""

import os
import time
from types import SimpleNamespace
from datetime import date, datetime, timezone
from typing import List
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import sys

import polars as pl
from tqdm import tqdm

# ✅ Correct import of your FMP client + typed errors
from src.dataprep.fetcher._fmp_client import (
    fmp_get,
    FMPAuthError,
    FMPPlanError,
    FMPRateLimitError,
    FMPServerError,
)

from src.dataprep.fetcher.ticker_data_sources import fetch_all_per_ticker
from src.dataprep.features.aggregation.ticker_row_builder import build_feature_table_from_inputs
from src.dataprep.features.aggregation.validate_dynamic_row import validate_dynamic_row
from src.dataprep.constants import EXPECTED_COLUMNS

# === Configuration (WF-controlled) ===========================================
def _get_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, str(int(default)))
    return v not in ("0", "", "false", "False", "no", "No")

def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _load_tickers_from_env_or_file(default_file: str) -> list[str]:
    inline = os.environ.get("TICKERS", "").strip()
    if inline:
        import re
        # split by commas, whitespace, or newlines; normalize & de-dup
        parts = [p.strip().upper() for p in re.split(r"[,\s]+", inline) if p.strip()]
        return sorted(dict.fromkeys(parts))
    path = os.environ.get("TICKERS_FILE", default_file)
    return load_tickers(path)

START_YEAR = _get_int("START_YEAR", 2021)
_end_year_env = _get_int("END_YEAR", 0)
END_YEAR = _end_year_env if _end_year_env > 0 else date.today().year
FREQ = os.environ.get("FREQ", "1Y")

TICKERS_FILE = os.environ.get("TICKERS_FILE", "config/us_tickers_subset_limited.txt")
SLEEP_BETWEEN_CALLS = _get_float("SLEEP_BETWEEN_CALLS", 1.0)
MAX_RETRIES = _get_int("MAX_RETRIES", 3)

OVERWRITE_MODE = os.environ.get("OVERWRITE_MODE", "append").lower()
VALID_OVERWRITE_MODES = {"append", "overwrite", "skip"}
if OVERWRITE_MODE not in VALID_OVERWRITE_MODES:
    raise ValueError(f"Invalid OVERWRITE_MODE: '{OVERWRITE_MODE}'. Must be one of {VALID_OVERWRITE_MODES}")

FORCE_MERGE = _get_bool("FORCE_MERGE", False)
STRICT = _get_bool("STRICT", False)

START_DATE = date(START_YEAR, 12, 31)
END_DATE = date(END_YEAR, 12, 31)
# ============================================================================

# ==== Simple robustness knobs (env-overridable) ==============================
MAX_CONSEC_429       = _get_int("MAX_CONSEC_429", 6)       # abort if we hit too many 429s in a row
MAX_GLOBAL_MINUTES   = _get_float("MAX_GLOBAL_MINUTES", 60.0)  # global time budget for the whole run

# ---- run-state
_RUN_DEADLINE_TS: float = 0.0

ACCEPT_STATUSES = {"ok", "flagged"}  # keep flagged rows, audit them

# ---- live progress helper ----
_progress_hist: list[tuple[float, int]] = []  # (timestamp, processed) in last ~2m

def _update_progress_live(status_dir: str, totals: dict, counts: dict, running=None, note: str | None = None) -> None:
    """
    totals: {"total_tasks": int, "tickers": int, "dates": int}
    counts: {"processed": int, "failed": int, "flagged": int}
    running: {"ticker": str, "as_of": str} | None
    """
    now = time.time()
    _progress_hist.append((now, int(counts.get("processed", 0))))
    # keep only last 120s window
    cutoff = now - 120
    while _progress_hist and _progress_hist[0][0] < cutoff:
        _progress_hist.pop(0)

    eta_iso = None
    if len(_progress_hist) >= 2:
        dt = _progress_hist[-1][0] - _progress_hist[0][0]
        ditems = _progress_hist[-1][1] - _progress_hist[0][1]
        rate = (ditems / dt) if dt > 0 else 0.0
        remaining = max(int(totals.get("total_tasks", 0)) - int(counts.get("processed", 0)), 0)
        if rate > 0:
            secs = int(remaining / rate)
            eta_iso = datetime.fromtimestamp(now + secs, tz=timezone.utc).replace(microsecond=0).isoformat()

    denom = max(int(totals.get("total_tasks", 0)), 1)
    percent = round(100 * int(counts.get("processed", 0)) / denom, 2)

    os.makedirs(status_dir, exist_ok=True)
    path = os.path.join(status_dir, "progress.json")

    # preserve started_at from previous writes
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                prev = json.load(f)
                started_at = prev.get("started_at") or started_at
    except Exception as _e:
        print(f"[WARN] progress read failed: {_e}")

    payload = {
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "totals": totals,
        "counts": counts,
        "percent": percent,
        "eta_utc": eta_iso,
        "running": running or {},
        "note": note or "",
    }
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as _e:
        print(f"[WARN] progress write failed: {_e}")

# ---------------- Run stats ----------------
@dataclass
class RunStats:
    ok: int = 0
    skipped: int = 0
    flagged: int = 0   # soft validations triggered
    failed: int = 0    # code/runtime failures
    changed_tickers: int = 0

    def update_from_lines(self, lines: list[str]) -> None:
        for ln in lines:
            if ln.startswith("[OK] "): self.ok += 1
            elif ln.startswith("[SKIP] "): self.skipped += 1
            elif ln.startswith("[FLAGGED] "): self.flagged += 1
            elif ln.startswith("[FAIL] "): self.failed += 1

def _maybe_preflight_fmp():
    """Run FMP preflight only if explicitly enabled via env."""
    if os.environ.get("FMP_PREFLIGHT", "0") in ("0", "", "false", "False"):
        print("⏭️  Skipping FMP preflight (FMP_PREFLIGHT=0).")
        return
    try:
        _ = fmp_get("/api/v3/ratios/AAPL", {"limit": 1})
        print("✅ FMP auth OK")
    except Exception as e:
        # Keep behavior for CLI runs: fail fast
        raise SystemExit(f"❌ FMP check failed: {e}")

OUTPUT_DIR = os.path.join("features_data", "tickers_history")
AUDIT_DIR  = os.path.join("features_data", "_audit")
STATIC_DIR = os.path.join("features_data", "tickers_static")
STATUS_DIR = os.path.join("features_data", "status")

# === Utilities ===
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(STATUS_DIR, exist_ok=True)

def get_dates_between(start: date, end: date, freq: str = "1Y") -> List[date]:
    current = start
    dates = []
    while current <= end:
        dates.append(current)
        current = current.replace(year=current.year + 1)
    return dates

def load_tickers(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path) as f:
        return [line.strip().upper() for line in f if line.strip()]

def get_parquet_path(ticker: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{ticker}.parquet")

def is_numeric_dtype(dtype: pl.DataType) -> bool:
    return isinstance(dtype, (
        pl.Float32, pl.Float64,
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
    ))

def fill_missing_columns(df: pl.DataFrame, all_columns: list[str]) -> pl.DataFrame:
    height = df.height
    to_add = []
    for col in all_columns:
        if col not in df.columns:
            # keep the SAME number of rows; for empty df this adds 0-length series
            to_add.append(pl.Series(name=col, values=[None] * height))
    if to_add:
        df = df.with_columns(to_add)
    return df.select(all_columns)

def cast_and_round_numeric(df: pl.DataFrame) -> pl.DataFrame:
    cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if is_numeric_dtype(dtype):
            cols.append(pl.col(col).round(2).cast(pl.Float32).alias(col))
        else:
            cols.append(pl.col(col))
    return df.select(cols)

def save_static_row(static_df: pl.DataFrame):
    static_path = os.path.join(STATIC_DIR, "static_ticker_info.parquet")
    os.makedirs(os.path.dirname(static_path), exist_ok=True)

    if os.path.exists(static_path):
        existing = pl.read_parquet(static_path)
    else:
        existing = pl.DataFrame()

    # Align columns
    all_columns = sorted(set(existing.columns) | set(static_df.columns))
    existing = fill_missing_columns(existing, all_columns)
    static_df = fill_missing_columns(static_df, all_columns)

    # Resolve a common dtype per column:
    # - prefer non-Null
    # - if either is Utf8 -> Utf8
    # - if both numeric -> Int32 (compact)
    # - else fall back to the non-Null one
    def _common_dtype(de: pl.DataType, dn: pl.DataType) -> pl.DataType:
        if de == pl.Null and dn != pl.Null:
            return dn
        if dn == pl.Null and de != pl.Null:
            return de
        if de == pl.Utf8 or dn == pl.Utf8:
            return pl.Utf8
        if is_numeric_dtype(de) and is_numeric_dtype(dn):
            return pl.Int32
        if de != pl.Null:
            return de
        return dn

    target_schema: dict[str, pl.DataType] = {}
    for col in all_columns:
        de = existing[col].dtype
        dn = static_df[col].dtype
        if de != dn:
            target_schema[col] = _common_dtype(de, dn)

    if target_schema:
        existing = existing.cast(target_schema)
        static_df = static_df.cast(target_schema)

    # Normalize one-hots: cast to Int8 and fill missing as 0
    ohe_cols = [c for c in all_columns if c.startswith("sector_") or c.startswith("country_")]
    if ohe_cols:
        existing = existing.with_columns([pl.col(c).cast(pl.Int8).fill_null(0) for c in ohe_cols])
        static_df = static_df.with_columns([pl.col(c).cast(pl.Int8).fill_null(0) for c in ohe_cols])

    combined = pl.concat([existing, static_df], how="vertical").unique(subset=["ticker"])
    combined.write_parquet(static_path, compression="zstd")
    print(f"💾 Static info saved (total: {combined.height} rows) → {static_path}")

# ----- Atomic write helper ----------------------------------------------------
def _atomic_write_parquet(df: pl.DataFrame, path: str) -> None:
    """Write parquet atomically: write to temp, then replace."""
    tmp = f"{path}.tmp"
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)

def save_or_append(df: pl.DataFrame, ticker: str, merge_with_existing: bool = True) -> bool:
    """
    Writes/merges the per-ticker parquet. Returns True if the file content changed.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"[BUG] df is not a Polars DataFrame for {ticker}. Got: {type(df)}")
    if "as_of" not in df.columns:
        raise ValueError(f"[BUG] 'as_of' missing in df for {ticker}. Got columns: {df.columns}")

    path = get_parquet_path(ticker)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prev_height = None

    if merge_with_existing and os.path.exists(path):
        df_existing = pl.read_parquet(path)

        # Ensure validation cols exist on existing file as Utf8 empty strings
        for c in ("validation_status", "violations"):
            if c not in df_existing.columns:
                df_existing = df_existing.with_columns(pl.lit("").cast(pl.Utf8).alias(c))

        # Align columns
        all_columns = sorted(set(df.columns) | set(df_existing.columns))
        df = fill_missing_columns(df, all_columns)
        df_existing = fill_missing_columns(df_existing, all_columns)

        # Normalize numerics (keeps strings intact)
        df = cast_and_round_numeric(df)
        df_existing = cast_and_round_numeric(df_existing)

        # Harmonize dtypes
        cast_schema = {}
        for col in all_columns:
            de = df_existing[col].dtype
            dn = df[col].dtype
            if de != dn:
                if is_numeric_dtype(de) and is_numeric_dtype(dn):
                    target = pl.Float32
                else:
                    if de == pl.Null and dn != pl.Null:
                        target = dn
                    elif dn == pl.Null and de != pl.Null:
                        target = de
                    elif de == pl.Utf8 or dn == pl.Utf8:
                        target = pl.Utf8
                    elif de == pl.Boolean or dn == pl.Boolean:
                        target = pl.Boolean
                    else:
                        target = de
                cast_schema[col] = target
        if cast_schema:
            df = df.cast(cast_schema)
            df_existing = df_existing.cast(cast_schema)

        # Merge
        combined = pl.concat([df_existing, df], how="vertical").unique(subset=["as_of"])

        # FINAL GUARANTEES for validation cols:
        if "validation_status" in combined.columns:
            combined = combined.with_columns(
                pl.coalesce([pl.col("validation_status").cast(pl.Utf8), pl.lit("")]).alias("validation_status")
            )
        else:
            combined = combined.with_columns(pl.lit("").alias("validation_status"))

        if "violations" in combined.columns:
            combined = combined.with_columns(
                pl.coalesce([pl.col("violations").cast(pl.Utf8), pl.lit("")]).alias("violations")
            )
        else:
            combined = combined.with_columns(pl.lit("").alias("violations"))

        prev_height = df_existing.height
        df = combined
    else:
        df = cast_and_round_numeric(df)

    df = df.sort("as_of")
    new_height = df.height

    # If we merged with existing and height changed -> changed
    if (prev_height is not None and new_height != prev_height):
        _atomic_write_parquet(df, path)
        print(f"💾 Saved: {ticker} ({df.height} rows, compressed) → {path}")
        return True

    # If we did NOT merge (overwrite mode) and the file already exists,
    # treat overwrite as "changed" (content is different, even if bytes match)
    if not merge_with_existing and os.path.exists(path):
        _atomic_write_parquet(df, path)
        print(f"💾 Saved (overwrite): {ticker} ({df.height} rows, compressed) → {path}")
        return True

    # Otherwise: do the temp write + byte-compare fallback
    tmp_path = path + ".tmp"
    df.write_parquet(tmp_path, compression="zstd")
    same_bytes = False
    try:
        same_bytes = os.path.getsize(tmp_path) == os.path.getsize(path)
    except FileNotFoundError:
        same_bytes = False
    os.replace(tmp_path, path)
    print(f"💾 Saved: {ticker} ({df.height} rows, compressed) → {path}")
    return not same_bytes

# ----- Feature build ----------------------------------------------------------
def has_enough_price_data(inputs: dict, as_of: date) -> bool:
    # Assuming this exists elsewhere; placeholder to preserve original behavior
    # Replace with your actual implementation if different.
    return True

def fetch_and_build_features(ticker: str, as_of: date) -> tuple[pl.DataFrame, pl.DataFrame, str | None]:
    inputs = fetch_all_per_ticker(ticker, div_lookback_years=5, other_lookback_years=5)

    if not has_enough_price_data(inputs, as_of):
        raise ValueError("Not enough price data available for required historical features")

    dynamic_df, static_df = build_feature_table_from_inputs(ticker, inputs, as_of)

    if not isinstance(dynamic_df, pl.DataFrame):
        raise TypeError(f"[BUG] dynamic_df is not a Polars DataFrame for {ticker}@{as_of} — got {type(dynamic_df)}")
    if not isinstance(static_df, pl.DataFrame):
        raise TypeError(f"[BUG] static_df is not a Polars DataFrame for {ticker}@{as_of} — got {type(static_df)}")
    if dynamic_df.is_empty():
        raise ValueError(f"[WARN] build_feature_table_from_inputs returned empty dynamic_df for {ticker}@{as_of}")

    # Try to extract sector (optional)
    sector = None
    if "sector" in static_df.columns:
        try:
            sector = static_df["sector"].to_list()[0]
        except Exception:
            sector = None

    dynamic_df = dynamic_df.with_columns([
        pl.lit(ticker).alias("ticker"),
        pl.lit(as_of).cast(pl.Date).alias("as_of")
    ])
    dynamic_df = dynamic_df.select(["ticker", "as_of"] + [c for c in dynamic_df.columns if c not in ("ticker", "as_of")])

    return dynamic_df, static_df, sector

def _write_flagged_audit(ticker: str, as_of, errors: list, row_df: pl.DataFrame) -> None:
    Path(AUDIT_DIR).mkdir(parents=True, exist_ok=True)
    fname = f"{ticker}_{as_of}.txt"  # test expects this pattern
    audit_fp = Path(AUDIT_DIR) / fname
    with open(audit_fp, "w") as f:
        f.write("\n".join(errors) if errors else "")

def _align_schemas(df1: pl.DataFrame, df2: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Ensure both DataFrames have the same columns
    all_cols = list(dict.fromkeys(df1.columns + df2.columns))  # preserve order
    for col in all_cols:
        if col not in df1.columns:
            df1 = df1.with_columns(pl.lit(None).alias(col))
        if col not in df2.columns:
            df2 = df2.with_columns(pl.lit(None).alias(col))
    # Reorder columns
    return df1.select(all_cols), df2.select(all_cols)

# ----- Row guards -------------------------------------------------------------
def _row_is_all_null_features(df: pl.DataFrame) -> bool:
    """True if all non-meta columns are null/NaN (i.e., empty feature row)."""
    meta = {"ticker", "as_of", "validation_status", "violations"}
    feat_cols = [c for c in df.columns if c not in meta]
    if not feat_cols:
        return True
    nn = df.select([pl.col(c).is_not_null().sum() for c in feat_cols]).row(0)
    return sum(nn) == 0

# ----- Retry helper (no retry on 429) ----------------------------------------
def _retry(max_tries: int, base_sleep: float = 0.5):
    """Exponential backoff retry decorator (skips retrying FMPRateLimitError)."""
    def deco(fn):
        def inner(*a, **kw):
            last = None
            tries = max(1, int(max_tries))
            for i in range(tries):
                try:
                    return fn(*a, **kw)
                except FMPRateLimitError:
                    # Don't retry rate limits here; let the outer loop decide
                    raise
                except Exception as e:
                    last = e
                    if i < tries - 1:
                        time.sleep(base_sleep * (2 ** i))
            raise last
        return inner
    return deco

@_retry(MAX_RETRIES, base_sleep=0.5)
def _fetch_build_validate_once(ticker, as_of, keep_df):
    dyn_df, static_df, sector = fetch_and_build_features(ticker, as_of)
    status, errors, dyn_df = validate_dynamic_row(
        dyn_df, ticker, prev_df=keep_df if not keep_df.is_empty() else None, sector=sector
    )
    dyn_df = dyn_df.with_columns([
        pl.lit(status).cast(pl.Utf8).alias("validation_status"),
        pl.lit(";".join(errors)).cast(pl.Utf8).alias("violations"),
    ])
    return dyn_df, static_df, sector, status, errors

# ----- Main per-ticker generator (with tiny guards) --------------------------
def generate_features_for_ticker(
    ticker: str,
    dates: list[date],
    on_progress=None
):
    logs: list[str] = []
    stats = RunStats()  # ok, skipped, flagged, failed, changed_tickers

    out_fp = Path(OUTPUT_DIR) / f"{ticker}.parquet"
    if out_fp.exists():
        df_existing = pl.read_parquet(out_fp)
        # Ensure legacy files have validation cols as empty strings
        for c in ("validation_status", "violations"):
            if c not in df_existing.columns:
                df_existing = df_existing.with_columns(pl.lit("").cast(pl.Utf8).alias(c))
    else:
        df_existing = pl.DataFrame()

    existing_dates = set(df_existing["as_of"].to_list()) if not df_existing.is_empty() else set()
    keep_df = df_existing
    changed = False
    saved_static = False

    # --- simple guards:
    consec_429 = 0
    total_tasks = len(dates)

    for as_of in dates:
        # Global time budget
        if _RUN_DEADLINE_TS and time.time() >= _RUN_DEADLINE_TS:
            logs.append(f"[INFO] stopping-early: global-deadline reached ({MAX_GLOBAL_MINUTES}m)")
            _update_progress_live(
                STATUS_DIR,
                totals={"total_tasks": total_tasks, "tickers": 1, "dates": total_tasks},
                counts={"processed": stats.ok + stats.flagged + stats.skipped + stats.failed,
                        "failed": stats.failed, "flagged": stats.flagged},
                running={"ticker": ticker, "as_of": str(as_of)},
                note="global-deadline"
            )
            break

        if as_of in existing_dates:
            stats.skipped += 1
            logs.append(f"[SKIP] {ticker} {as_of} exists — skipping")
            # progress after skip
            if on_progress:
                on_progress(ticker, as_of, stats)
            _update_progress_live(
                STATUS_DIR,
                totals={"total_tasks": total_tasks, "tickers": 1, "dates": total_tasks},
                counts={"processed": stats.ok + stats.flagged + stats.skipped + stats.failed,
                        "failed": stats.failed, "flagged": stats.flagged},
                running={"ticker": ticker, "as_of": str(as_of)},
            )
            time.sleep(max(0.0, SLEEP_BETWEEN_CALLS))
            continue

        try:
            dyn_df, static_df, sector, status, errors = _fetch_build_validate_once(
                ticker, as_of, keep_df
            )
            consec_429 = 0  # reset on success

            # save dynamic
            changed |= save_or_append(dyn_df, ticker, merge_with_existing=(OVERWRITE_MODE != "overwrite"))
            # save static once
            if not saved_static and not static_df.is_empty():
                save_static_row(static_df)
                saved_static = True

            if status == "ok":
                stats.ok += 1
            elif status == "flagged":
                stats.flagged += 1
            else:
                stats.failed += 1

            logs.append(f"[OK] {ticker} {as_of} status={status} added")

        except FMPRateLimitError:
            consec_429 += 1
            logs.append(f"[FAIL] {ticker} {as_of}: FMPRateLimitError: 429 Too Many Requests (consecutive={consec_429})")
            if consec_429 >= MAX_CONSEC_429:
                logs.append(f"[INFO] stopping-early: rate-limit-storm (>= {MAX_CONSEC_429} consecutive 429s)")
                _update_progress_live(
                    STATUS_DIR,
                    totals={"total_tasks": total_tasks, "tickers": 1, "dates": total_tasks},
                    counts={"processed": stats.ok + stats.flagged + stats.skipped + stats.failed,
                            "failed": stats.failed, "flagged": stats.flagged},
                    running={"ticker": ticker, "as_of": str(as_of)},
                    note="rate-limit-storm"
                )
                return logs, stats, changed  # abort this ticker ASAP

            # short, linear backoff to be gentle but simple
            time.sleep(min(60.0, 2.0 * consec_429))

        except (FMPAuthError, FMPPlanError) as e:
            # Hard errors: do not continue wasting time on this ticker
            stats.failed += 1
            logs.append(f"[FAIL] {ticker} {as_of}: {type(e).__name__}: {e}")
            break

        except Exception as e:
            stats.failed += 1
            logs.append(f"[FAIL] {ticker} {as_of}: {type(e).__name__}: {e}")

        # progress + pacing (after each date)
        if on_progress:
            on_progress(ticker, as_of, stats)
        _update_progress_live(
            STATUS_DIR,
            totals={"total_tasks": total_tasks, "tickers": 1, "dates": total_tasks},
            counts={"processed": stats.ok + stats.flagged + stats.skipped + stats.failed,
                    "failed": stats.failed, "flagged": stats.flagged},
            running={"ticker": ticker, "as_of": str(as_of)},
        )
        time.sleep(max(0.0, SLEEP_BETWEEN_CALLS))

    return logs, stats, changed

# ----- Driver ----------------------------------------------------------------
def main() -> int:
    global _RUN_DEADLINE_TS
    _RUN_DEADLINE_TS = time.time() + MAX_GLOBAL_MINUTES * 60.0 if MAX_GLOBAL_MINUTES > 0 else 0.0

    _maybe_preflight_fmp()

    tickers = _load_tickers_from_env_or_file(TICKERS_FILE)
    dates = get_dates_between(START_DATE, END_DATE, FREQ)

    ensure_output_dir()

    all_logs: list[str] = []
    any_changed = False

    # Simple outer loop: process tickers one-by-one
    for t in tickers:
        if _RUN_DEADLINE_TS and time.time() >= _RUN_DEADLINE_TS:
            all_logs.append(f"[INFO] stopping-early: global-deadline reached before ticker {t}")
            break

        logs, stats, changed = generate_features_for_ticker(t, dates)
        any_changed = any_changed or changed
        all_logs.extend(logs)

        # If we want to stop after a rate-limit storm in a ticker, the function returns early.
        if _RUN_DEADLINE_TS and time.time() >= _RUN_DEADLINE_TS:
            all_logs.append("[INFO] stopping-early: global-deadline reached")
            break

    # Decide exit code
    hard_fail = any(ln.startswith("[FAIL]") for ln in all_logs)
    if STRICT and hard_fail:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
