from src.dataprep.features import (
    find_nearest_price,
    ensure_date_column,
    adjust_series_for_splits
)
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


def test_adjust_series_for_splits_on_dividends():
    div_df = pl.DataFrame({
        "date": ["2023-01-01", "2023-06-01", "2023-12-31"],
        "dividend": [1.0, 1.2, 1.5]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    split_df = pl.DataFrame({
        "date": [datetime.date(2023, 7, 1)],
        "split_ratio": [2]
    })

    adjusted = adjust_series_for_splits(div_df, split_df, "dividend")
    assert adjusted[0, "dividend"] == 1.0
    assert adjusted[1, "dividend"] == 1.2
    assert adjusted[2, "dividend"] == 0.75
