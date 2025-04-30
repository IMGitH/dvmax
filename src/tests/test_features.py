import pytest
import polars as pl
import datetime
from src.dataprep.features import FeatureEngineer, ensure_date_column, find_nearest_price

def test_compute_6m_return():
    engineer = FeatureEngineer()
    price_df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01", "2024-07-01"],
        "close": [100, 120]
    }))
    result = engineer.compute_6m_return(price_df, as_of_date=datetime.date(2024, 7, 1))
    assert isinstance(result, float)
    assert pytest.approx(result, rel=1e-2) == 0.2

def test_compute_volatility():
    engineer = FeatureEngineer()
    price_df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "close": [100, 101, 102]
    }))
    result = engineer.compute_volatility(price_df)
    assert isinstance(result, float)
    assert result >= 0

def test_compute_dividend_yield():
    engineer = FeatureEngineer()
    dividend_df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01"],
        "dividend": [4]
    }))
    price_df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01"],
        "close": [100]
    }))
    result = engineer.compute_dividend_yield(dividend_df, price_df)
    assert isinstance(result, float)
    assert pytest.approx(result, rel=1e-3) == 0.04