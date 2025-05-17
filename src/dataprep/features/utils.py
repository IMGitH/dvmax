import polars as pl
import datetime

def ensure_date_column(df: pl.DataFrame, column_name: str = "date") -> pl.DataFrame:
    if df[column_name].dtype != pl.Date:
        return df.with_columns(
            pl.col(column_name).str.strptime(pl.Date, "%Y-%m-%d")
        )
    return df


def find_nearest_price(df: pl.DataFrame, target_date: datetime.date) -> float:
    filtered = df.filter(pl.col("date") <= target_date)
    if filtered.is_empty():
        raise ValueError(f"No price data available on or before {target_date}")
    return filtered[-1, "close"]