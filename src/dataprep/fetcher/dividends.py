import polars as pl
from datetime import datetime, timedelta
import yfinance as yf
from src.dataprep.fetcher.utils import default_date_range
from src.dataprep.fetcher.base import FMPClient
from dateutil.relativedelta import relativedelta

def fetch_dividends(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int | None = None,
    grace_quarters: int = 1
) -> pl.DataFrame:
    """
    Fetches dividend data for a given stock using FMP API as primary source,
    falling back to Yahoo Finance if FMP lacks full historical range.

    Parameters:
        ticker (str): Ticker symbol (e.g., "AAPL").
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        lookback_years (int): Number of years to look back if no dates are provided.

    Returns:
        pl.DataFrame: A DataFrame with 'date' and 'dividend' columns.

    Raises:
        RuntimeError: If neither FMP nor Yahoo provides data covering the start_date.
    """
    # Unified and safe date resolution
    start_date, end_date = default_date_range(
        lookback_years=lookback_years,
        start_date=start_date,
        end_date=end_date,
        quarter_mode=True
    )
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()

    # === FMP Primary Fetch ===
    client = FMPClient()
    try:
        response = client.fetch(
            f"historical-price-full/stock_dividend/{ticker}",
            {"from": start_date, "to": end_date}
        )
        data = response.get("historical", [])

        if data:
            df_fmp = pl.DataFrame(data).select(["date", "dividend"]).with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
            )
            earliest_fmp = df_fmp.select(pl.col("date").min()).item() if df_fmp.height > 0 else None
            grace_period = relativedelta(months=3 * grace_quarters)
            max_acceptable_start = start_dt + grace_period
            if earliest_fmp and earliest_fmp <= max_acceptable_start:
                return df_fmp
            print("⚠️ FMP data does not go back to requested start date. Trying Yahoo fallback...")
    except Exception as e:
        print(f"⚠️ FMP fetch failed: {e}")

    # === Yahoo Finance Fallback ===
    try:
        yf_ticker = yf.Ticker(ticker)
        dividends = yf_ticker.dividends
        if dividends.empty:
            raise ValueError("No dividend data from Yahoo Finance.")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        df_yf = (
            pl.DataFrame({"date": dividends.index, "dividend": dividends.values})
            .with_columns(pl.col("date").cast(pl.Date))
            .filter((pl.col("date") >= pl.lit(start_dt)) & (pl.col("date") <= pl.lit(end_dt)))
        )

        earliest_yf = df_yf.select(pl.col("date").min()).item() if not df_yf.is_empty() else None
        if earliest_yf and earliest_yf <= start_dt:
            return df_yf

        raise ValueError("Yahoo Finance data too recent.")

    except Exception as e:
        raise RuntimeError(f"Dividend fetch failed for {ticker}: {str(e)}")
