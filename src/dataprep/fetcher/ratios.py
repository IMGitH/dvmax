import datetime
import polars as pl
from src.dataprep.fetcher.base import FMPClient
from src.dataprep.fetcher.utils import default_date_range

def fetch_ratios(ticker: str, period: str = "annual", start_date: str | None = None, end_date: str | None = None) -> pl.DataFrame:
    if period not in {"annual", "quarter"}:
        raise ValueError("Period must be 'annual' or 'quarter'")

    start_date, end_date = start_date or default_date_range()[0], end_date or default_date_range()[1]
    client = FMPClient()
    params = {"period": period} if period == "quarter" else {}
    try:
        data = client.fetch(f"ratios/{ticker}", params)
    except PermissionError as e:
        print(f"[WARN] {e}")
        return pl.DataFrame()

    if not data:
        return pl.DataFrame()

    df = pl.DataFrame(data).with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
    )

    range_start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    range_end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    df = df.filter((pl.col("date") >= range_start) & (pl.col("date") <= range_end))

    return df.select([
        "date", "priceEarningsRatio", "payoutRatio", "priceToSalesRatio",
        "enterpriseValueMultiple", "priceFairValue", "returnOnEquity",
        "debtEquityRatio", "netProfitMargin", "dividendYield"
    ])
