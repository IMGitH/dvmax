import os
import shutil
from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.dataprep.report import generate_stock_features as gsf


@pytest.fixture
def temp_output_dir(tmp_path):
    # Override global variables for test isolation
    gsf.OUTPUT_DIR = str(tmp_path / "features_parquet")
    gsf.MERGED_FILE = str(tmp_path / "features_parquet" / "features_all.parquet")
    gsf.AS_OF_DATE = date(2024, 5, 1)
    yield tmp_path
    shutil.rmtree(gsf.OUTPUT_DIR, ignore_errors=True)


@patch("src.dataprep.report.generate_stock_features.fetch_all")
@patch("src.dataprep.report.generate_stock_features.build_feature_table_from_inputs")
def test_generate_feature_parquets(mock_build, mock_fetch, tmp_path):
    tickers = ["AAPL", "MSFT"]
    output_dir = tmp_path / "features_parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override config
    gsf.OUTPUT_DIR = str(output_dir)
    gsf.MERGED_FILE = str(output_dir / "features_all.parquet")
    gsf.OVERWRITE_MODE = "all"

    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("\n".join(tickers))
    gsf.TICKERS_FILE = str(ticker_file)
    gsf.AS_OF_DATE = date(2024, 5, 1)

    # Mock fetch_all to accept any kwargs
    def mock_fetch_side_effect(ticker, **kwargs):
        return {"ticker": ticker, "dummy": "data"}
    mock_fetch.side_effect = mock_fetch_side_effect

    # Mock build_feature_table_from_inputs based on ticker
    def mock_build_side_effect(ticker, inputs, as_of):
        is_msft = ticker == "MSFT"
        return pl.DataFrame({
            "ticker": [ticker],
            "6m_return": [0.1 if not is_msft else 0.15],
            "12m_return": [0.2 if not is_msft else 0.25],
            "volatility": [0.3 if not is_msft else 0.35],
            "max_drawdown_1y": [0.4],
            "sector_relative_6m": [0.5],
            "sma_50_200_delta": [0.6],
            "net_debt_to_ebitda": [1.0],
            "ebit_interest_cover": [5.0],
            "ebit_interest_cover_capped": [5.0],
            "eps_cagr_3y": [0.1],
            "fcf_cagr_3y": [0.2],
            "dividend_yield": [0.03],
            "dividend_cagr_3y": [0.05],
            "dividend_cagr_5y": [0.07],
            "yield_vs_5y_median": [0.01],
            "pe_ratio": [15.0],
            "pfcf_ratio": [12.0],
            "payout_ratio": [0.4],
            "country": ["US"],
            "has_eps_cagr_3y": [1],
            "has_fcf_cagr_3y": [1],
            "has_dividend_yield": [1],
            "has_dividend_cagr_3y": [1],
            "has_dividend_cagr_5y": [1],
            "has_ebit_interest_cover": [1],
        })
    mock_build.side_effect = mock_build_side_effect

    # Run the actual pipeline
    gsf.main()

    # Validate individual files
    for ticker in tickers:
        path = Path(gsf.OUTPUT_DIR) / f"{ticker}.parquet"
        assert path.exists(), f"{path} was not created"

    # Validate merged file
    merged_path = Path(gsf.MERGED_FILE)
    assert merged_path.exists(), "Merged parquet file was not created"
    merged_df = pl.read_parquet(merged_path)

    # Validate both tickers are in the merged file
    found_tickers = merged_df["ticker"].unique().to_list()
    assert sorted(found_tickers) == sorted(tickers), f"Unexpected tickers: {found_tickers}"

    # Confirm data is not duplicated and is ticker-specific
    msft_row = merged_df.filter(pl.col("ticker") == "MSFT")
    aapl_row = merged_df.filter(pl.col("ticker") == "AAPL")
    assert msft_row["6m_return"][0] == 0.15
    assert aapl_row["6m_return"][0] == 0.1
