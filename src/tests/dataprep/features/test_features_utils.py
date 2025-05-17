from src.dataprep.features import find_nearest_price, ensure_date_column
import polars as pl
import datetime

def test_find_nearest_price_returns_closest_value():
    df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01", "2024-01-10", "2024-01-20"],
        "close": [100, 110, 120]
    }))
    target_date = datetime.date(2024, 1, 12)
    print("\n=== test_find_nearest_price_returns_closest_value ===")
    print(f"Target date: {target_date}")
    print(df)
    result = find_nearest_price(df, target_date)
    print(f"Nearest price: {result}")

    assert result == 110


def test_find_nearest_price_on_exact_match():
    df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-01", "2024-01-10"],
        "close": [100, 110]
    }))
    target_date = datetime.date(2024, 1, 10)
    print("\n=== test_find_nearest_price_on_exact_match ===")
    print(f"Target date: {target_date}")
    print(df)
    result = find_nearest_price(df, target_date)
    print(f"Nearest price: {result}")
    assert result == 110


def test_find_nearest_price_with_unsorted_data():
    df = ensure_date_column(pl.DataFrame({
        "date": ["2024-01-10", "2024-01-01", "2024-01-20"],
        "close": [110, 100, 120]
    }))
    target_date = datetime.date(2024, 1, 5)
    print("\n=== test_find_nearest_price_with_unsorted_data ===")
    print(f"Target date: {target_date}")
    print(df)
    result = find_nearest_price(df, target_date)
    print(f"Nearest price: {result}")
    assert result == 100
