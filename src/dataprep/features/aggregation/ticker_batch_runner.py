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
from src.dataprep.constants import EXPECTED_COLUMNS

# === Configuration ===
# START_DATE = date(2020, 12, 31)  # <-- adjust your backtesting start
START_DATE = date(2024, 1, 1)  # <-- adjust your backtesting start

END_DATE = date.today().replace(month=12, day=31)  # e.g. last full year
FREQ = "1Y"  # or use month intervals via custom date generation

OUTPUT_DIR = "features_parquet/timeseries"
TICKERS_FILE = "us_tickers_subset.txt"
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3
OVERWRITE_MODE = os.environ.get("OVERWRITE_MODE", "none").lower()

OUTPUT_DIR = os.path.join("features_parquet", "tickers_data")

# === Utilities ===

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



def save_or_append(df: pl.DataFrame, ticker: str, merge_with_existing: bool = True):
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"[BUG] df is not a Polars DataFrame for {ticker}. Got: {type(df)}")

    if "as_of" not in df.columns:
        raise ValueError(f"[BUG] 'as_of' missing in df for {ticker}. Got columns: {df.columns}")

    path = get_parquet_path(ticker)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if merge_with_existing and os.path.exists(path):
        df_existing = pl.read_parquet(path)

        # Ensure both have same schema
        all_columns = sorted(set(df.columns) | set(df_existing.columns))
        df_existing = fill_missing_columns(df_existing, all_columns)
        df = fill_missing_columns(df, all_columns)

        df = pl.concat([df_existing, df], how="vertical").unique(subset=["as_of"])

    df.write_parquet(path)
    print(f"💾 Saved: {ticker} ({df.height} rows) → {path}")


def fetch_and_build_features(ticker: str, as_of: date) -> pl.DataFrame:
    inputs = fetch_all_per_ticker(ticker, div_lookback_years=5, other_lookback_years=5)
    if not has_enough_price_data(inputs, as_of):
        raise ValueError("Not enough price data available for required historical features")
    df = build_feature_table_from_inputs(ticker, inputs, as_of)
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"[BUG] build_feature_table_from_inputs did not return a Polars DataFrame for {ticker}@{as_of} — got {type(df)}")
    if df.is_empty():
        raise ValueError(f"[WARN] build_feature_table_from_inputs returned empty df for {ticker}@{as_of}")
    df = df.with_columns([
        pl.lit(ticker).alias("ticker"),
        pl.lit(as_of).cast(pl.Date).alias("as_of")
    ])
    df = df.select(["ticker", "as_of"] + [col for col in df.columns if col not in ("ticker", "as_of")])
    return df


def generate_features_for_ticker(ticker: str, all_dates: List[date]):
    collected = []
    results = []

    for as_of in all_dates:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = fetch_and_build_features(ticker, as_of)
                if df.is_empty():
                    results.append(f"[WARN] No data for {ticker} on {as_of}")
                    break
                collected.append(df)
                results.append(f"[OK] {ticker} on {as_of}")
                break

            except ValueError as ve:
                if "Not enough" in str(ve):
                    results.append(f"[SKIP] {ticker}@{as_of}: {ve}")
                    break
                if attempt < MAX_RETRIES:
                    print(f"[RETRY] {ticker}@{as_of} (attempt {attempt}) – {ve}")
                    time.sleep(2 * attempt)
                else:
                    results.append(f"[FAIL] {ticker}@{as_of} after {MAX_RETRIES} attempts: {ve}")
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

    merged_df = pl.concat(dfs, how="vertical")

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
