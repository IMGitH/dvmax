from src.dataprep.features import find_nearest_price, ensure_date_column
import polars as pl
import datetime

def test_find_nearest_price_returns_closest_value():
    df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01", "2024-01-10", "2024-01-20"],
        "close": [100, 110, 120]
    }))
    result = find_nearest_price(df, datetime.date(2024, 1, 12))  # closest to Jan 10
    assert result == 110

def test_find_nearest_price_on_exact_match():
    df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01", "2024-01-10"],
        "close": [100, 110]
    }))
    result = find_nearest_price(df, datetime.date(2024, 1, 10))  # exact date
    assert result == 110

def test_find_nearest_price_with_unsorted_data():
    df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-10", "2024-01-01", "2024-01-20"],
        "close": [110, 100, 120]
    }))
    result = find_nearest_price(df, datetime.date(2024, 1, 5))  # closest to Jan 1
    assert result == 100
