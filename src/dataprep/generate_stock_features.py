import os
import time
import traceback
from datetime import date
from typing import List

import polars as pl
from tqdm import tqdm

from src.dataprep.fetcher.fetch_all import fetch_all
from src.dataprep.report.feature_table import build_feature_table_from_inputs

# === Configuration ===
AS_OF_DATE = date.today()
OUTPUT_DIR = "features_parquet"
TICKERS_FILE = "us_tickers.txt"
SLEEP_BETWEEN_CALLS = 1.0  # seconds
MAX_RETRIES = 3


def load_tickers(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path) as f:
        return [line.strip().upper() for line in f if line.strip()]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def should_skip(ticker: str) -> bool:
    return os.path.exists(os.path.join(OUTPUT_DIR, f"{ticker}.parquet"))


def save_features(df: pl.DataFrame, ticker: str):
    df.write_parquet(os.path.join(OUTPUT_DIR, f"{ticker}.parquet"))


def fetch_and_build_features(ticker: str) -> pl.DataFrame:
    inputs = fetch_all(ticker, div_lookback_years=5, other_lookback_years=4)
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


def main():
    ensure_output_dir()
    tickers = load_tickers(TICKERS_FILE)

    print(f"🟢 Generating features for {len(tickers)} tickers (as of {AS_OF_DATE})...")

    for ticker in tqdm(tickers, desc="Processing tickers"):
        message = generate_features_for_ticker(ticker)
        if message:
            print(message)
        time.sleep(SLEEP_BETWEEN_CALLS)


if __name__ == "__main__":
    main()
