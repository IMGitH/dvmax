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
    Adjusts a time series column (e.g., dividends, EPS) for stock splits using cumulative backward adjustment.
    Assumes both `df` and `split_df` contain a 'date' column.
    
    Parameters:
        df: Polars DataFrame with 'date' and the column to adjust.
        split_df: Polars DataFrame with 'date' and 'split_ratio'.
        column: The name of the column in `df` to adjust.
        skip_warning: If True, suppresses warning if split_df is empty.

    Returns:
        A new DataFrame with the adjusted `column`.
    """
    if "date" not in df.columns or column not in df.columns:
        raise ValueError(f"Both 'date' and '{column}' columns must be present in the input dataframe.")

    if split_df.is_empty():
        if not skip_warning:
            logging.warning("[Splits] No split history available — skipping adjustment.")
        return df

    # Step 1: Sort and compute cumulative ratio
    split_df = (
        split_df
        .sort("date")
        .with_columns([
            pl.col("split_ratio").cum_prod().alias("cumulative_ratio"),
            pl.col("date").alias("split_date")
        ])
    )

    # Step 2: Join split info to main df using backward join
    df = df.sort("date")
    df = df.join_asof(split_df, left_on="date", right_on="split_date", strategy="backward") # "backward" since we dont have enough data

    # Step 3: Fill nulls and compute adjusted values safely
    df = df.with_columns([
        pl.col("cumulative_ratio").fill_null(1.0).alias("adj_factor")
    ])
    
    df = df.with_columns([
        (pl.col(column) / pl.col("adj_factor")).alias(column)
    ])

    # Step 4: Drop temporary columns
    return df.drop("split_date", "split_ratio", "cumulative_ratio", "adj_factor")
