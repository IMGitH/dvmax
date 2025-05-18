import polars as pl
from src.dataprep.fetcher.base import FMPClient

def fetch_ratios(ticker: str, limit:int, period: str = "annual") -> pl.DataFrame:
    """
    Fetches valuation and profitability ratios for a given ticker using FMP free-tier API.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        period (str): Either "annual" or "quarter". Defaults to "annual".
        limit (int): Number of most recent records to return (max 4 for annual data on free tier).

    Returns:
        pl.DataFrame: DataFrame with selected ratios and date column.

    Notes:
        - Only the most recent X annual records are available from FMP for free users.
        - This function slices locally to return up to `limit` entries, sorted by date descending.
    """
    if period not in {"annual", "quarter"}:
        raise ValueError("Period must be 'annual' or 'quarter'")
    # if not (1 <= limit <= 4):
    #     raise ValueError("limit must be between 1 and 4 (FMP free-tier constraint)")

    client = FMPClient()
    params = {"period": period} if period == "quarter" else {}

    try:
        data = client.fetch(f"ratios/{ticker}", params)
    except PermissionError as e:
        print(f"[WARN] {e}")
        return pl.DataFrame()

    if not data:
        return pl.DataFrame()

    df = pl.DataFrame(data)

    if df.schema["date"] == pl.Utf8:
        df = df.with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"))

    df = df.sort("date", descending=True).head(limit).sort("date")

    return df.select([
        "date", "priceEarningsRatio", "priceToFreeCashFlowsRatio", 
        "payoutRatio", "priceToSalesRatio", "enterpriseValueMultiple", 
        "priceFairValue", "returnOnEquity", "debtEquityRatio", 
        "netProfitMargin", "dividendYield"
    ])
