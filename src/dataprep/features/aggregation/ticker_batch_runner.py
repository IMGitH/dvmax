import os
import time
import traceback
from datetime import date
from typing import List
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.dataprep.fetcher.ticker_data_sources import fetch_all_per_ticker
from src.dataprep.features.aggregation.ticker_row_builder import build_feature_table_from_inputs
from src.dataprep.features.aggregation.validate_dynamic_row import validate_dynamic_row
from src.dataprep.constants import EXPECTED_COLUMNS

# === Configuration ===
# START_DATE = date(2020, 12, 31)  # <-- adjust your backtesting start
START_DATE = date(2023, 1, 1)  # <-- adjust your backtesting start

END_DATE = date.today().replace(month=12, day=31)  # e.g. last full year
FREQ = "1Y"  # or use month intervals via custom date generation

OUTPUT_DIR = "features_data/timeseries"
TICKERS_FILE = "us_tickers_subset_limited.txt"
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3
# OVERWRITE_MODE controls how existing feature data is handled:
# - "none": skip existing rows entirely
# - "append": skip existing rows, append only new ones (default)
# - "partial": overwrite rows only for dates being regenerated
# - "all": ignore existing file completely and overwrite everything
# - "merged": only affects final merged file, not per-ticker files
OVERWRITE_MODE = os.environ.get("OVERWRITE_MODE", "append").lower()
VALID_OVERWRITE_MODES = {"none", "append", "partial", "all", "merged"}
if OVERWRITE_MODE not in VALID_OVERWRITE_MODES:
    raise ValueError(f"Invalid OVERWRITE_MODE: '{OVERWRITE_MODE}'. Must be one of {VALID_OVERWRITE_MODES}")

OUTPUT_DIR = os.path.join("features_data", "tickers_history")

# === Utilities ===
def save_static_row(static_df: pl.DataFrame):
    static_path = os.path.join("features_data", "tickers_static", "static_ticker_info.parquet")
    os.makedirs(os.path.dirname(static_path), exist_ok=True)

    if os.path.exists(static_path):
        existing = pl.read_parquet(static_path)

        # Ensure schema compatibility
        all_columns = sorted(set(existing.columns) | set(static_df.columns))
        existing = fill_missing_columns(existing, all_columns)
        static_df = fill_missing_columns(static_df, all_columns)

        # Match dtypes exactly
        cast_schema = {}
        for col in all_columns:
            dtype_existing = existing[col].dtype
            dtype_new = static_df[col].dtype
            if dtype_existing != dtype_new:
                if is_numeric_dtype(dtype_existing) and is_numeric_dtype(dtype_new):
                    cast_schema[col] = dtype_existing  # Keep existing schema
                else:
                    cast_schema[col] = dtype_existing  # Also fallback for strings/bools

        static_df = static_df.cast(cast_schema)
        existing = existing.cast(cast_schema)

        combined = pl.concat([existing, static_df], how="vertical").unique(subset=["ticker"])

    else:
        combined = cast_and_round_numeric(static_df)

    combined.write_parquet(static_path, compression="zstd")
    print(f"💾 Static info saved (total: {combined.height} rows) → {static_path}")

    
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

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def validate_schema(df: pl.DataFrame, ticker: str):
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"[SCHEMA] {ticker} missing columns: {missing}")


def get_parquet_path(ticker: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{ticker}.parquet")


def fill_missing_columns(df: pl.DataFrame, all_columns: List[str]) -> pl.DataFrame:
    for col in all_columns:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    return df.select(all_columns)

def is_numeric_dtype(dtype: pl.DataType) -> bool:
    return isinstance(dtype, (
        pl.Float32, pl.Float64,
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
    ))

def cast_and_round_numeric(df: pl.DataFrame) -> pl.DataFrame:
    cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if is_numeric_dtype(dtype):
            # Round first in float64, then cast to float32
            cols.append(pl.col(col).round(2).cast(pl.Float32).alias(col))
        else:
            cols.append(pl.col(col))
    return df.select(cols)

def save_or_append(df: pl.DataFrame, ticker: str, merge_with_existing: bool = True):
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"[BUG] df is not a Polars DataFrame for {ticker}. Got: {type(df)}")

    if "as_of" not in df.columns:
        raise ValueError(f"[BUG] 'as_of' missing in df for {ticker}. Got columns: {df.columns}")

    path = get_parquet_path(ticker)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if merge_with_existing and os.path.exists(path):
        df_existing = pl.read_parquet(path)

        # Ensure both have the same set of columns
        all_columns = sorted(set(df.columns) | set(df_existing.columns))
        df = fill_missing_columns(df, all_columns)
        df_existing = fill_missing_columns(df_existing, all_columns)

        # Round and cast to Float32 first
        df = cast_and_round_numeric(df)
        df_existing = cast_and_round_numeric(df_existing)

        # Match dtypes exactly
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

        df = pl.concat([df_existing, df], how="vertical").unique(subset=["as_of"])

    else:
        df = cast_and_round_numeric(df)

    df = df.sort("as_of")
    df.write_parquet(path, compression="zstd")
    print(f"💾 Saved: {ticker} ({df.height} rows, compressed) → {path}")


def fetch_and_build_features(ticker: str, as_of: date) -> tuple[pl.DataFrame, pl.DataFrame]:
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

    dynamic_df = dynamic_df.with_columns([
        pl.lit(ticker).alias("ticker"),
        pl.lit(as_of).cast(pl.Date).alias("as_of")
    ])
    dynamic_df = dynamic_df.select(["ticker", "as_of"] + [col for col in dynamic_df.columns if col not in ("ticker", "as_of")])

    return dynamic_df, static_df

def generate_features_for_ticker(ticker: str, all_dates: List[date]):
    collected = []
    results = []
    static_written = False

    # Load existing data to skip already-processed dates
    existing_path = get_parquet_path(ticker)
    existing_df = None  # <-- ensure it's always defined
    existing_dates = set()
    if os.path.exists(existing_path):
        try:
            existing_df = pl.read_parquet(existing_path)
            existing_dates = set(existing_df["as_of"].cast(pl.Date).to_list())
        except Exception as e:
            print(f"[WARN] Failed to read existing data for {ticker} – treating as new: {e}")

    for as_of in all_dates:
        if as_of in existing_dates:
            if OVERWRITE_MODE == "none":
                results.append(f"[SKIP] {ticker}@{as_of} already exists")
                continue
            elif OVERWRITE_MODE == "partial" and existing_df is not None:
                # Remove the old row for this date so it can be replaced
                existing_df = existing_df.filter(pl.col("as_of") != as_of)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                dynamic_df, static_df = fetch_and_build_features(ticker, as_of)

                # Run validation here
                validate_dynamic_row(dynamic_df, ticker, prev_df=existing_df)

                # Add to collection for saving later
                collected.append(dynamic_df)

                # Save static row only once
                if not static_written:
                    save_static_row(static_df)
                    static_written = True

                results.append(f"[OK] {ticker}@{as_of}")
                break  # Done for this date

            except ValueError as ve:
                short_reason = str(ve).split(":")[0].replace(" ", "_").replace("[", "").replace("]", "")
                quarantine_path = f"features_data/_invalid/{ticker}_{as_of}_{short_reason}.parquet"
                os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)

                if 'dynamic_df' in locals() and isinstance(dynamic_df, pl.DataFrame):
                    if "as_of" in dynamic_df.columns:
                        dynamic_df = dynamic_df.sort("as_of")
                    dynamic_df.write_parquet(quarantine_path)

                    error_txt = quarantine_path.replace(".parquet", ".txt")
                    with open(error_txt, "w") as f:
                        f.write(f"{ticker}@{as_of}\nReason: {ve}")

                print(f"[INVALID] {ticker}@{as_of} quarantined → {quarantine_path} due to: {ve}")
                results.append(f"[INVALID] {ticker}@{as_of}: {ve}")
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

    if collected:
        full_df = pl.concat(collected, how="vertical").unique(subset=["as_of"])
        print(f"[SAVE] Appending {full_df.height} new rows for {ticker}")
        save_or_append(full_df, ticker, merge_with_existing=True)

    return "\n".join(results)


def has_enough_price_data(inputs: dict, as_of: date, required_days: int = 260) -> bool:
    if "prices" not in inputs:
        return False
    df = inputs["prices"]
    return df.filter(pl.col("date") <= pl.lit(as_of)).height >= required_days


def merge_all_feature_vectors():
    paths = sorted(Path(OUTPUT_DIR).glob("*.parquet"))
    if not paths:
        raise RuntimeError("No feature vector files found.")

    # Determine superset of all columns
    all_columns = set()
    for path in paths:
        df = pl.read_parquet(path)
        all_columns.update(df.columns)
    all_columns = sorted(all_columns)

    dfs = []
    for path in paths:
        df = pl.read_parquet(path)
        df = fill_missing_columns(df, all_columns)

        # Enforce consistent dtypes (float for all numerics)
        schema = {}
        for col in df.columns:
            dtype = df[col].dtype
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                schema[col] = pl.Float64
            elif dtype in [pl.Utf8, pl.Boolean, pl.Float64, pl.Date]:
                schema[col] = dtype
            else:
                schema[col] = pl.Float64  # fallback to float

        df = df.cast(schema)
        dfs.append(df)

    merged_df = pl.concat(dfs, how="vertical").sort(["ticker", "as_of"])

    merged_file = os.path.join(OUTPUT_DIR, "features_all_tickers_timeseries.parquet")
    if OVERWRITE_MODE in ("all", "merged") or not os.path.exists(merged_file):
        merged_df.write_parquet(merged_file)
        print(f"✅ Merged {len(paths)} files into {merged_file}")
    else:
        print(f"⏩ Skipped merging – file already exists.")


    merged_file = os.path.join(OUTPUT_DIR, "features_all_tickers_timeseries.parquet")
    if OVERWRITE_MODE in ("all", "merged") or not os.path.exists(merged_file):
        merged_df.write_parquet(merged_file)
        print(f"✅ Merged {len(paths)} files into {merged_file}")
    else:
        print(f"⏩ Skipped merging – file already exists.")


def main():
    ensure_output_dir()
    tickers = load_tickers(TICKERS_FILE)
    all_dates = get_dates_between(START_DATE, END_DATE)
    print(f"🟢 Generating features for {len(tickers)} tickers × {len(all_dates)} dates...")

    for ticker in tqdm(tickers, desc="Processing tickers"):
        message = generate_features_for_ticker(ticker, all_dates)
        if message:
            tqdm.write(message)
        time.sleep(SLEEP_BETWEEN_CALLS)

    merge_all_feature_vectors()
    print(f"All dates generated: {[d.isoformat() for d in get_dates_between(START_DATE, END_DATE)]}")


if __name__ == "__main__":
    main()
