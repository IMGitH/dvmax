from src.dataprep.features.utils import ensure_date_column, find_nearest_price
import polars as pl
import datetime
from dateutil.relativedelta import relativedelta
import logging
from datetime import date, timedelta

def compute_return_over_period(
    df: pl.DataFrame,
    period: relativedelta,
    as_of_date: datetime.date | None = None
) -> float:
    df = ensure_date_column(df, "date").sort("date")
    if "close" not in df.columns:
        raise ValueError("Expected a 'close' column in the DataFrame")

    as_of_date = as_of_date or datetime.date.today()
    past_date = as_of_date - period

    try:
        price_now = find_nearest_price(df, as_of_date)
        price_past = find_nearest_price(df, past_date)
    except ValueError as e:
        logging.warning(e)
        return 0.0
    return (price_now - price_past) / price_past


def compute_6m_return(df: pl.DataFrame, as_of_date: datetime.date | None = None) -> float:
    return compute_return_over_period(df, relativedelta(months=6), as_of_date)


def compute_12m_return(df: pl.DataFrame, as_of_date: datetime.date | None = None) -> float:
    return compute_return_over_period(df, relativedelta(years=1), as_of_date)


def compute_volatility(df: pl.DataFrame) -> float:
    df = ensure_date_column(df, "date").sort("date")
    if "close" not in df.columns:
        raise ValueError("Expected a 'close' column in the DataFrame")

    returns = df.select((pl.col("close") / pl.col("close").shift(1) - 1).alias("daily_return")).drop_nulls()
    std_dev = returns["daily_return"].std()
    return 0.0 if std_dev is None else std_dev * (252 ** 0.5)


def compute_max_drawdown(df: pl.DataFrame) -> float:
    df = ensure_date_column(df, "date").sort("date")

    if "close" not in df.columns or df.height < 2:
        return 0.0

    prices = df["close"].to_list()

    peak = prices[0]
    max_drawdown = 0.0

    for price in prices[1:]:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown  # already positive


def compute_sector_relative_return(
    stock_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    lookback_days: int,
    as_of: date
) -> float:
    """
    Computes the return of the stock relative to its sector over a given lookback period.
    """
    cutoff = as_of - timedelta(days=lookback_days)

    stock_df = stock_df.filter((pl.col("date") >= cutoff) & (pl.col("date") <= as_of)).sort("date")
    sector_df = sector_df.filter((pl.col("date") >= cutoff) & (pl.col("date") <= as_of)).sort("date")

    if stock_df.height < 2 or sector_df.height < 2:
        return 0.0

    stock_start = stock_df[0, "close"]
    stock_end = stock_df[-1, "close"]
    sector_start = sector_df[0, "close"]
    sector_end = sector_df[-1, "close"]

    if stock_start <= 0 or stock_end <= 0 or sector_start <= 0 or sector_end <= 0:
        return 0.0

    stock_return = stock_end / stock_start - 1
    sector_return = sector_end / sector_start - 1

    return stock_return - sector_return



def compute_payout_ratio(df: pl.DataFrame) -> float:
    if "payoutRatio" in df.columns:
        valid = df.drop_nulls("payoutRatio").filter(pl.col("payoutRatio") > 0)
        if not valid.is_empty():
            return valid[-1, "payoutRatio"]
    return 0.0
