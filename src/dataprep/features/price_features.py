import polars as pl
import datetime
from src.dataprep.features.utils import ensure_date_column, find_nearest_price

def compute_6m_return(df: pl.DataFrame, as_of_date: datetime.date | None = None) -> float:
    df = ensure_date_column(df, "date").sort("date")
    if "close" not in df.columns:
        raise ValueError("Expected a 'close' column in the DataFrame")

    as_of_date = as_of_date or datetime.date.today()
    six_months_ago = as_of_date - datetime.timedelta(days=6 * 30)

    price_now = find_nearest_price(df, as_of_date)
    price_past = find_nearest_price(df, six_months_ago)
    return (price_now - price_past) / price_past

def compute_volatility(df: pl.DataFrame) -> float:
    df = ensure_date_column(df, "date").sort("date")
    if "close" not in df.columns:
        raise ValueError("Expected a 'close' column in the DataFrame")

    returns = df.select((pl.col("close") / pl.col("close").shift(1) - 1).alias("daily_return")).drop_nulls()
    std_dev = returns["daily_return"].std()
    return 0.0 if std_dev is None else std_dev * (252 ** 0.5)
