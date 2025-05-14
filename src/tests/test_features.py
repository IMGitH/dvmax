import pytest
import polars as pl
import datetime
from src.dataprep.features import compute_6m_return, compute_volatility, ensure_date_column

def test_compute_6m_return():
    df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01", "2024-07-01"],
        "close": [100, 120]
    }))
    result = compute_6m_return(df, as_of_date=datetime.date(2024, 7, 1))
    assert isinstance(result, float)
    assert pytest.approx(result, rel=1e-2) == 0.2

def test_compute_volatility():
    df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "close": [100, 101, 102]
    }))
    result = compute_volatility(df)
    assert isinstance(result, float)
    assert result >= 0
