import polars as pl
import datetime

def ensure_date_column(df: pl.DataFrame, column_name: str = "date") -> pl.DataFrame:
    if df[column_name].dtype != pl.Date:
        return df.with_columns(
            pl.col(column_name).str.strptime(pl.Date, "%Y-%m-%d")
        )
    return df

def find_nearest_price(df: pl.DataFrame, target_date: datetime.date) -> float:
    return (
        df.with_columns((pl.col("date") - pl.lit(target_date)).abs().alias("date_diff"))
        .sort("date_diff")
        .get_column("close")[0]
    )
