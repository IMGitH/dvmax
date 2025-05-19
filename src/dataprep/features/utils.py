import polars as pl
import datetime
import logging


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


def adjust_series_for_splits(
    df: pl.DataFrame, 
    split_df: pl.DataFrame, 
    column: str,
    skip_warning: bool = False
) -> pl.DataFrame:
    """
    Generic utility to adjust any time series column for stock splits.
    E.g., dividends, EPS, FCF/share.
    """
    if "date" not in df.columns or column not in df.columns:
        raise ValueError(f"Both 'date' and '{column}' columns must be present in the input dataframe.")

    if split_df.is_empty():
        if not skip_warning:
            logging.warning("[Splits] No split history available — skipping adjustment.")
        return df

    for row in split_df.iter_rows(named=True):
        date, ratio = row["date"], row["split_ratio"]
        df = df.with_columns(
            pl.when(pl.col("date") < date)
            .then(pl.col(column) / ratio)
            .otherwise(pl.col(column))
            .alias(column)
        )

    return df
