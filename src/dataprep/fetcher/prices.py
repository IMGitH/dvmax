import polars as pl
import datetime
from src.dataprep.fetcher.base import FMPClient

def fetch_prices(ticker: str, start_date: str | None = None, end_date: str | None = None) -> pl.DataFrame:
    today = datetime.date.today().isoformat()
    start_date = start_date or (datetime.date.today() - datetime.timedelta(days=5 * 365)).isoformat()
    end_date = end_date or today

    client = FMPClient()
    data = client.fetch(
        f"historical-price-full/{ticker}",
        {"from": start_date, "to": end_date}
    ).get("historical", [])

    if not data:
        return pl.DataFrame()

    return pl.DataFrame(data).select(["date", "close"]).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )
