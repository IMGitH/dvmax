import pytest
from src.dataprep.fetcher import StockFetcher

def test_fetch_dividends_returns_valid_dataframe():
    fetcher = StockFetcher()
    df = fetcher.fetch_dividends("AAPL", start_date="2020-01-01", end_date="2024-01-01")
    assert df.height > 0
    assert "date" in df.columns
    assert "dividend" in df.columns

def test_fetch_prices_returns_valid_dataframe():
    fetcher = StockFetcher()
    df = fetcher.fetch_prices("AAPL", start_date="2020-01-01", end_date="2024-01-01")
    assert df.height > 0
    assert "date" in df.columns
    assert "close" in df.columns

def test_fetch_ratios_returns_valid_dataframe():
    fetcher = StockFetcher()
    df = fetcher.fetch_ratios("AAPL")
    assert df.height > 0
    assert "date" in df.columns
    assert "peRatio" in df.columns
    assert "payoutRatio" in df.columns
    assert "dividendYield" in df.columns