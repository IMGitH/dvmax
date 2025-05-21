import os
import time
import traceback
from datetime import date
from typing import List
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.dataprep.fetcher.fetch_all import fetch_all
from src.dataprep.report.feature_table import build_feature_table_from_inputs

# === Configuration ===
AS_OF_DATE = date.today()
DATE_STR = AS_OF_DATE.strftime("%d-%m-%Y")
OUTPUT_DIR = os.path.join("features_parquet", DATE_STR)
MERGED_FILE = os.path.join(OUTPUT_DIR, "features_all.parquet")
TICKERS_FILE = "us_tickers_subset.txt"  # or "us_tickers.txt"
SLEEP_BETWEEN_CALLS = 1.0  # seconds
MAX_RETRIES = 3

# Options: "none", "all", "merged"
OVERWRITE_MODE = os.environ.get("OVERWRITE_MODE", "none").lower()

EXPECTED_COLUMNS = [
    "ticker", "6m_return", "12m_return", "volatility", "max_drawdown_1y",
    "sector_relative_6m", "sma_50_200_delta", "net_debt_to_ebitda",
    "ebit_interest_cover", "ebit_interest_cover_capped", "eps_cagr_3y",
    "fcf_cagr_3y", "dividend_yield", "dividend_cagr_3y", "dividend_cagr_5y",
    "yield_vs_5y_median", "pe_ratio", "pfcf_ratio", "payout_ratio", "country",

    # Binary presence indicators for nullable metrics
    "has_eps_cagr_3y", "has_fcf_cagr_3y",
    "has_dividend_yield", "has_dividend_cagr_3y", "has_dividend_cagr_5y",
    "has_ebit_interest_cover"

    # Note: sector one-hot columns are still dynamic and handled separately
]

def load_tickers(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path) as f:
        return [line.strip().upper() for line in f if line.strip()]

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_parquet_path(ticker: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{ticker}.parquet")

def should_skip(ticker: str) -> bool:
    if OVERWRITE_MODE == "all":
        return False
    return os.path.exists(get_parquet_path(ticker))

def validate_schema(df: pl.DataFrame, ticker: str):
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"[SCHEMA] {ticker} missing columns: {missing}")

def save_features(df: pl.DataFrame, ticker: str):
    validate_schema(df, ticker)
    df.write_parquet(get_parquet_path(ticker))

def fetch_and_build_features(ticker: str) -> pl.DataFrame:
    inputs = fetch_all(ticker, div_lookback_years=5, other_lookback_years=5)
    return build_feature_table_from_inputs(ticker, inputs, AS_OF_DATE)

def generate_features_for_ticker(ticker: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if should_skip(ticker):
                return f"[SKIP] {ticker} already exists."

            df = fetch_and_build_features(ticker)
            if df.is_empty():
                return f"[WARN] No data for {ticker}"

            save_features(df, ticker)
            return f"[OK] {ticker}"

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"[RETRY] {ticker} (attempt {attempt}) – {e}")
                time.sleep(2 * attempt)
            else:
                print(f"[FAIL] {ticker} after {MAX_RETRIES} attempts: {e}")
                traceback.print_exc()
                return f"[FAIL] {ticker}"

def merge_all_feature_vectors():
    paths = sorted(Path(OUTPUT_DIR).glob("*.parquet"))
    if not paths:
        raise RuntimeError("No feature vector files found.")

    ref_schema = pl.read_parquet(paths[0]).schema
    dfs = [pl.read_parquet(p).cast(ref_schema) for p in paths]
    merged_df = pl.concat(dfs, how="vertical")

    if OVERWRITE_MODE in ("all", "merged") or not os.path.exists(MERGED_FILE):
        merged_df.write_parquet(MERGED_FILE)
        print(f"✅ Merged {len(paths)} files into {MERGED_FILE}")
    else:
        print(f"⏩ Skipped merging – {MERGED_FILE} already exists.")

def main():
    ensure_output_dir()
    tickers = load_tickers(TICKERS_FILE)
    print(f"🟢 Generating features for {len(tickers)} tickers (as of {AS_OF_DATE})...")

    for ticker in tqdm(tickers, desc="Processing tickers"):
        message = generate_features_for_ticker(ticker)
        if message:
            tqdm.write(message)
        time.sleep(SLEEP_BETWEEN_CALLS)

    merge_all_feature_vectors()

if __name__ == "__main__":
    main()
