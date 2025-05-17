import polars as pl
import datetime
from src.dataprep.features.utils import ensure_date_column, find_nearest_price


def compute_return_over_period(df: pl.DataFrame, period_days: int, as_of_date: datetime.date | None = None) -> float:
    df = ensure_date_column(df, "date").sort("date")
    if "close" not in df.columns:
        raise ValueError("Expected a 'close' column in the DataFrame")

    as_of_date = as_of_date or datetime.date.today()
    past_date = as_of_date - datetime.timedelta(days=period_days)

    price_now = find_nearest_price(df, as_of_date)
    price_past = find_nearest_price(df, past_date)
    return (price_now - price_past) / price_past


def compute_6m_return(df: pl.DataFrame, as_of_date: datetime.date | None = None) -> float:
    return compute_return_over_period(df, 6 * 30, as_of_date)


def compute_12m_return(df: pl.DataFrame, as_of_date: datetime.date | None = None) -> float:
    return compute_return_over_period(df, 365, as_of_date)


def compute_volatility(df: pl.DataFrame) -> float:
    df = ensure_date_column(df, "date").sort("date")
    if "close" not in df.columns:
        raise ValueError("Expected a 'close' column in the DataFrame")

    returns = df.select((pl.col("close") / pl.col("close").shift(1) - 1).alias("daily_return")).drop_nulls()
    std_dev = returns["daily_return"].std()
    return 0.0 if std_dev is None else std_dev * (252 ** 0.5)


def compute_max_drawdown(df: pl.DataFrame) -> float:
    df = df.sort("date")
    if "close" not in df.columns or df.height < 2:
        return 0.0

    if df.schema["date"] == pl.Utf8:
        df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    prices = df["close"].to_list()

    max_drawdown = 0.0
    peak = prices[0]
    for price in prices[1:]:
        if price > peak:
            peak = price
        drawdown = (price - peak) / peak
        max_drawdown = min(max_drawdown, drawdown)

    return abs(max_drawdown)


def compute_sma_delta(df: pl.DataFrame, short: int = 50, long: int = 200) -> float:
    df = ensure_date_column(df).sort("date")
    if "close" not in df.columns or df.height < long:
        return 0.0

    short_sma = df[-short:, "close"].mean()
    long_sma = df[-long:, "close"].mean()

    if long_sma == 0:
        return 0.0

    return (short_sma - long_sma) / long_sma