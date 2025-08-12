
import os
import shutil
import tempfile
from datetime import date
import polars as pl

from src.dataprep.features.aggregation.ticker_batch_runner import (
    generate_features_for_ticker,
    save_or_append,
    get_parquet_path,
)

def mock_feature_df(ticker, as_of):
    return pl.DataFrame([
        {
            "ticker": ticker,
            "as_of": as_of,
            "6m_return": 0.1,
            "12m_return": 0.2,
            "volatility": 0.3,
            "max_drawdown_1y": 0.1,
            "sector_relative_6m": 0.01,
            "sma_50_200_delta": 0.05,
            "net_debt_to_ebitda": 1.5,
            "ebit_interest_cover": 10.0,
            "ebit_interest_cover_capped": 10.0,
            "eps_cagr_3y": 0.1,
            "fcf_cagr_3y": 0.1,
            "dividend_yield": 0.01,
            "dividend_cagr_3y": 0.01,
            "dividend_cagr_5y": 0.01,
            "yield_vs_5y_median": 0.01,
            "pe_ratio": 15.0,
            "pfcf_ratio": 18.0,
            "payout_ratio": 0.3,
            "has_eps_cagr_3y": 1,
            "has_fcf_cagr_3y": 1,
            "has_dividend_yield": 1,
            "has_dividend_cagr_3y": 1,
            "has_dividend_cagr_5y": 1,
            "has_ebit_interest_cover": 1,
        }
    ])

def test_append_skips_existing(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    try:
        ticker = "TEST"
        os.makedirs(os.path.join(tmpdir, "tickers_history"), exist_ok=True)
        monkeypatch.setattr("src.dataprep.features.aggregation.ticker_batch_runner.OUTPUT_DIR", os.path.join(tmpdir, "tickers_history"))

        path = get_parquet_path(ticker)
        initial_df = mock_feature_df(ticker, date(2024, 1, 1))
        save_or_append(initial_df, ticker, merge_with_existing=False)

        assert os.path.exists(path)
        original_rows = pl.read_parquet(path)
        assert original_rows.height == 1

        # Try saving again with same date (should skip if implemented properly)
        second_df = mock_feature_df(ticker, date(2024, 1, 1))
        save_or_append(second_df, ticker, merge_with_existing=True)
        updated_rows = pl.read_parquet(path)
        assert updated_rows.height == 1  # no duplicate

        # Add a new date
        third_df = mock_feature_df(ticker, date(2025, 1, 1))
        save_or_append(third_df, ticker, merge_with_existing=True)
        final_rows = pl.read_parquet(path)
        assert final_rows.height == 2  # one new row added

    finally:
        shutil.rmtree(tmpdir)
