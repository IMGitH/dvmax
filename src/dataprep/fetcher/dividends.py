import polars as pl
from src.dataprep.fetcher.base import FMPClient
from src.dataprep.fetcher.utils import default_date_range

def fetch_dividends(ticker: str, start_date: str | None = None, end_date: str | None = None) -> pl.DataFrame:
    start_date, end_date = start_date or default_date_range()[0], end_date or default_date_range()[1]
    client = FMPClient()
    data = client.fetch(f"historical-price-full/stock_dividend/{ticker}", {"from": start_date, "to": end_date}).get("historical", [])
    if not data:
        return pl.DataFrame()
    return pl.DataFrame(data).select(["date", "dividend"]).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )
