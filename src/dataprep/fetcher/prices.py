import polars as pl
from datetime import datetime
from src.dataprep.fetcher.base import FMPClient
from src.dataprep.fetcher.utils import default_date_range
from datetime import timedelta


def fetch_prices(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int | None = None,
    grace_days: int = 7
) -> pl.DataFrame:
    """
    Fetches historical daily closing prices for a given stock from FMP.

    Parameters:
        ticker (str): Stock ticker symbol.
        start_date (str, optional): Start date in 'YYYY-MM-DD'. If None, defaults to 5 years ago.
        end_date (str, optional): End date in 'YYYY-MM-DD'. If None, defaults to today.
        lookback_years (int): Used if start_date and end_date are not provided.

    Returns:
        pl.DataFrame: DataFrame with 'date' and 'close' columns.

    Raises:
        RuntimeError: If the returned data does not cover the requested start date.
    """
    start_date, end_date = default_date_range(
        lookback_years=lookback_years,
        start_date=start_date,
        end_date=end_date
    )

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    client = FMPClient()
    data = client.fetch(
        f"historical-price-full/{ticker}",
        {"from": start_date, "to": end_date}
    ).get("historical", [])

    if not data:
        raise RuntimeError(f"No price data returned for {ticker} between {start_date} and {end_date}.")

    df = pl.DataFrame(data).select(["date", "close"]).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )

    # Validation
    actual_start = df.select(pl.col("date").min()).item()
    actual_end = df.select(pl.col("date").max()).item()

    if actual_start > start_dt:
        raise RuntimeError(f"Data for {ticker} starts at {actual_start}, after requested {start_date}.")
    if actual_end < end_dt - timedelta(days=grace_days):
        raise RuntimeError(f"Data for {ticker} ends at {actual_end}, too far before requested {end_date}.")
    if actual_start > start_dt + timedelta(days=grace_days):
        raise RuntimeError(
            f"Data for {ticker} starts at {actual_start}, which is more than {grace_days} days after requested start {start_date}."
    )

    return df
