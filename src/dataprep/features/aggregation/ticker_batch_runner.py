import os
import time
import traceback
from datetime import date
from typing import List
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import sys

import polars as pl
from tqdm import tqdm

from src.dataprep.fetcher._fmp_client import fmp_get
from src.dataprep.fetcher.ticker_data_sources import fetch_all_per_ticker
from src.dataprep.features.aggregation.ticker_row_builder import build_feature_table_from_inputs
from src.dataprep.features.aggregation.validate_dynamic_row import validate_dynamic_row
from src.dataprep.constants import EXPECTED_COLUMNS


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
    """Run FMP preflight unless disabled via env."""
    if os.environ.get("FMP_PREFLIGHT", "1") in ("0", "false", "False"):
        print("⏭️  Skipping FMP preflight (FMP_PREFLIGHT=0).")
        return
    try:
        _ = fmp_get("/api/v3/ratios/AAPL", {"limit": 1})
        print("✅ FMP auth OK")
    except Exception as e:
        # Keep behavior for CLI runs: fail fast
        raise SystemExit(f"❌ FMP check failed: {e}")


# === Configuration (simplified) ===
START_DATE = date(2021, 12, 31)  # <-- adjust your backtesting start
END_DATE = date.today().replace(month=12, day=31)  # e.g. last full year
FREQ = "1Y"

TICKERS_FILE = "us_tickers_subset_limited.txt"
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3

# Modes:
# - "append":    skip existing rows for each ticker/date; append only new ones
# - "overwrite": ignore existing per-ticker file; rebuild entirely
# - "skip":      if a per-ticker file exists, skip that ticker entirely
OVERWRITE_MODE = os.environ.get("OVERWRITE_MODE", "append").lower()
VALID_OVERWRITE_MODES = {"append", "overwrite", "skip"}
if OVERWRITE_MODE not in VALID_OVERWRITE_MODES:
    raise ValueError(f"Invalid OVERWRITE_MODE: '{OVERWRITE_MODE}'. Must be one of {VALID_OVERWRITE_MODES}")

# Force merge regardless of mtimes; e.g. FORCE_MERGE=1 python script.py
FORCE_MERGE = os.environ.get("FORCE_MERGE", "0") not in ("0", "", "false", "False")
# STRICT now only considers 'failed'; 'flagged' never fails the run
STRICT = os.environ.get("STRICT", "0") not in ("0", "", "false", "False")

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


def fill_missing_columns(df: pl.DataFrame, all_columns: List[str]) -> pl.DataFrame:
    for col in all_columns:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
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
        all_columns = sorted(set(existing.columns) | set(static_df.columns))
        existing = fill_missing_columns(existing, all_columns)
        static_df = fill_missing_columns(static_df, all_columns)

        cast_schema = {}
        for col in all_columns:
            dtype_existing = existing[col].dtype
            dtype_new = static_df[col].dtype
            if dtype_existing != dtype_new:
                if is_numeric_dtype(dtype_existing) and is_numeric_dtype(dtype_new):
                    cast_schema[col] = dtype_existing
                else:
                    cast_schema[col] = dtype_existing

        static_df = static_df.cast(cast_schema)
        existing = existing.cast(cast_schema)
        combined = pl.concat([existing, static_df], how="vertical").unique(subset=["ticker"])
    else:
        combined = cast_and_round_numeric(static_df)

    combined.write_parquet(static_path, compression="zstd")
    print(f"💾 Static info saved (total: {combined.height} rows) → {static_path}")


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
        all_columns = sorted(set(df.columns) | set(df_existing.columns))
        df = fill_missing_columns(df, all_columns)
        df_existing = fill_missing_columns(df_existing, all_columns)

        df = cast_and_round_numeric(df)
        df_existing = cast_and_round_numeric(df_existing)

        cast_schema = {}
        for col in all_columns:
            dtype_existing = df_existing[col].dtype
            dtype_new = df[col].dtype
            if dtype_existing != dtype_new:
                if is_numeric_dtype(dtype_existing) and is_numeric_dtype(dtype_new):
                    cast_schema[col] = pl.Float32
                else:
                    cast_schema[col] = dtype_existing
        if cast_schema:
            df = df.cast(cast_schema)
            df_existing = df_existing.cast(cast_schema)

        combined = pl.concat([df_existing, df], how="vertical").unique(subset=["as_of"])
        prev_height = df_existing.height
        df = combined
    else:
        df = cast_and_round_numeric(df)

    df = df.sort("as_of")
    new_height = df.height

    if (prev_height is None and not os.path.exists(path)) or (prev_height is not None and new_height != prev_height):
        df.write_parquet(path, compression="zstd")
        print(f"💾 Saved: {ticker} ({df.height} rows, compressed) → {path}")
        return True

    # Write fresh file anyway (values may change without height change)
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


def generate_features_for_ticker(ticker: str, all_dates: List[date]) -> tuple[str, bool, RunStats]:
    """
    Returns (log_text, changed_flag, stats).
    changed_flag is True if per-ticker parquet was updated.
    """
    existing_df = None
    existing_dates = set()
    existing_path = get_parquet_path(ticker)

    if OVERWRITE_MODE == "skip" and os.path.exists(existing_path):
        return f"[SKIP TICKER] {ticker} (mode=skip, file exists)", False, RunStats(skipped=1)

    if OVERWRITE_MODE != "overwrite" and os.path.exists(existing_path):
        try:
            existing_df = pl.read_parquet(existing_path)
            existing_dates = set(existing_df["as_of"].cast(pl.Date).to_list())
        except Exception as e:
            print(f"[WARN] Failed to read existing data for {ticker} – treating as new: {e}")

    collected = []
    results = []
    static_written = False

    for as_of in all_dates:
        # append mode → skip dates that already exist
        if OVERWRITE_MODE == "append" and as_of in existing_dates:
            results.append(f"[SKIP] {ticker}@{as_of} already exists (append mode)")
            continue

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                dynamic_df, static_df, sector = fetch_and_build_features(ticker, as_of)

                # Soft validation: status in {'ok','flagged'}; never fatal
                status, violations, dynamic_df = validate_dynamic_row(
                    dynamic_df, ticker, prev_df=existing_df, sector=sector
                )

                # Annotate + audit (for flagged)
                if status == "flagged":
                    audit_path = os.path.join(AUDIT_DIR, f"{ticker}_{as_of}.txt")
                    with open(audit_path, "w") as f:
                        for v in violations:
                            f.write(v + "\n")
                    dynamic_df = dynamic_df.with_columns([
                        pl.lit("flagged").alias("validation_status"),
                        pl.lit("|").join(pl.Series(violations)).alias("violations"),
                    ])
                    results.append(f"[FLAGGED] {ticker}@{as_of}: " + "; ".join(violations))
                else:
                    dynamic_df = dynamic_df.with_columns([
                        pl.lit("ok").alias("validation_status"),
                        pl.lit("").alias("violations"),
                    ])
                    results.append(f"[OK] {ticker}@{as_of}")

                collected.append(dynamic_df)

                # Save static row only once
                if not static_written:
                    save_static_row(static_df)
                    static_written = True

                break  # Done for this date

            except ValueError as ve:
                # Treat as a runtime/inputs failure (not validation): don't write row
                print(f"[FAIL] {ticker}@{as_of} value error: {ve}")
                results.append(f"[FAIL] {ticker}@{as_of}")
                break

            except Exception as e:
                if attempt < MAX_RETRIES:
                    print(f"[RETRY] {ticker}@{as_of} (attempt {attempt}) – {e}")
                    time.sleep(2 * attempt)
                else:
                    print(f"[FAIL] {ticker}@{as_of} after {MAX_RETRIES} attempts: {e}")
                    traceback.print_exc()
                    results.append(f"[FAIL] {ticker}@{as_of}")
                    break

    changed = False
    if collected:
        full_df = pl.concat(collected, how="vertical").unique(subset=["as_of"]).sort("as_of")
        print(f"[SAVE] Appending {full_df.height} new rows for {ticker}")
        # overwrite mode → do not merge with existing
        changed = save_or_append(full_df, ticker, merge_with_existing=(OVERWRITE_MODE != "overwrite"))

    # Build stats for this ticker based on the message lines
    ticker_stats = RunStats()
    ticker_stats.update_from_lines(results)
    if changed:
        ticker_stats.changed_tickers = 1

    return "\n".join(results), changed, ticker_stats


def has_enough_price_data(inputs: dict, as_of: date, required_days: int = 260) -> bool:
    if "prices" not in inputs:
        return False
    df = inputs["prices"]
    return df.filter(pl.col("date") <= pl.lit(as_of)).height >= required_days


def merge_all_feature_vectors(force_merge: bool = False):
    paths = sorted(Path(OUTPUT_DIR).glob("*.parquet"))
    if not paths:
        raise RuntimeError("No feature vector files found.")

    merged_file = os.path.join(OUTPUT_DIR, "features_all_tickers_timeseries.parquet")

    def _parts_newer_than_merged() -> bool:
        if not os.path.exists(merged_file):
            return True
        merged_mtime = os.path.getmtime(merged_file)
        latest_part_mtime = max(os.path.getmtime(p) for p in paths)
        return latest_part_mtime > merged_mtime

    if not (force_merge or _parts_newer_than_merged()):
        print("⏩ Skipped merging – merged file is up to date.")
        return

    # Build superset of columns
    all_columns = set()
    for path in paths:
        df = pl.read_parquet(path)
        all_columns.update(df.columns)
    all_columns = sorted(all_columns)

    dfs = []
    for path in paths:
        df = pl.read_parquet(path)
        df = fill_missing_columns(df, all_columns)
        # Enforce consistent dtypes (floats for numerics)
        schema = {}
        for col in df.columns:
            dtype = df[col].dtype
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                schema[col] = pl.Float64
            elif dtype in [pl.Utf8, pl.Boolean, pl.Float64, pl.Date]:
                schema[col] = dtype
            else:
                schema[col] = pl.Float64
        df = df.cast(schema)
        dfs.append(df)

    merged_df = pl.concat(dfs, how="vertical").sort(["ticker", "as_of"])
    merged_df.write_parquet(merged_file)
    print(f"✅ Merged {len(paths)} files into {merged_file}")


def _write_status_files(stats: RunStats):
    os.makedirs(STATUS_DIR, exist_ok=True)

    # processed.json (ledger)
    from glob import glob
    from pathlib import Path as _Path
    done = { _Path(p).stem: "ok" for p in glob(os.path.join(OUTPUT_DIR, "*.parquet")) }
    _Path(os.path.join(STATUS_DIR, "processed.json")).write_text(json.dumps(done, indent=2))

    # last_run.json (summary)
    _Path(os.path.join(STATUS_DIR, "last_run.json")).write_text(json.dumps(asdict(stats), indent=2))


def main():
    _maybe_preflight_fmp()
    ensure_output_dir()
    tickers = load_tickers(TICKERS_FILE)
    all_dates = get_dates_between(START_DATE, END_DATE)
    print(f"🟢 Generating features for {len(tickers)} tickers × {len(all_dates)} dates...")

    any_changed = False
    agg = RunStats()

    for ticker in tqdm(tickers, desc="Processing tickers"):
        message, changed, tstats = generate_features_for_ticker(ticker, all_dates)
        any_changed = any_changed or changed
        # aggregate
        agg.ok += tstats.ok
        agg.skipped += tstats.skipped
        agg.flagged += tstats.flagged
        agg.failed += tstats.failed
        agg.changed_tickers += tstats.changed_tickers

        if message:
            tqdm.write(message)
        time.sleep(SLEEP_BETWEEN_CALLS)

    merge_all_feature_vectors(force_merge=FORCE_MERGE or any_changed)

    # Persist status files for the workflow
    _write_status_files(agg)

    # Print concise summary
    print(
        f"🏁 Summary: ok={agg.ok}, skipped={agg.skipped}, flagged={agg.flagged}, "
        f"failed={agg.failed}, changed_tickers={agg.changed_tickers}"
    )
    print(f"All dates generated: {[d.isoformat() for d in get_dates_between(START_DATE, END_DATE)]}")

    # Exit policy:
    # - default & STRICT: only fail on hard failures; flagged never fails
    sys.exit(1 if agg.failed > 0 else 0)


if __name__ == "__main__":
    main()
