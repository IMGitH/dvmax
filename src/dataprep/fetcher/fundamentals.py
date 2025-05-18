import polars as pl
from datetime import datetime
from src.dataprep.fetcher.base import FMPClient


def _fetch_fundamental(endpoint: str, ticker: str, limit: int, period: str = "annual") -> pl.DataFrame:
    """
    Internal utility to fetch financial data (e.g. income statement, balance sheet) from FMP.

    Parameters:
        endpoint (str): API endpoint path (e.g., "income-statement").
        ticker (str): Stock ticker symbol.
        limit (int): Number of most recent records to return (max 4 for free-tier annual data).
        period (str): Either "annual" or "quarter".

    Returns:
        pl.DataFrame: The requested subset of financial data, parsed and sorted.
    """
    if period not in {"annual", "quarter"}:
        raise ValueError("Period must be 'annual' or 'quarter'")
    if not (1 <= limit <= 4):
        raise ValueError("limit must be between 1 and 4 (FMP free-tier constraint)")

    client = FMPClient()
    params = {"period": period} if period == "quarter" else {}
    data = client.fetch(f"{endpoint}/{ticker}", params)
    if not data:
        return pl.DataFrame()

    df = pl.DataFrame(data)
    if df.schema.get("date") == pl.Utf8:
        df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    return df.sort("date", descending=True).head(limit).sort("date")


def fetch_income_statement_fund(ticker: str, limit: int, period: str = "annual") -> pl.DataFrame:
    return _fetch_fundamental("income-statement", ticker, limit, period).select([
        "date", "incomeBeforeTax", "interestExpense"
    ])


def fetch_balance_sheet_fund(ticker: str, limit: int, period: str = "annual") -> pl.DataFrame:
    return _fetch_fundamental("balance-sheet-statement", ticker, limit, period).select([
        "date", "cashAndShortTermInvestments", "totalDebt"
    ])


def fetch_cashflow_statement_fund(ticker: str, limit: int, period: str = "annual") -> pl.DataFrame:
    return _fetch_fundamental("cash-flow-statement", ticker, limit, period).select([
        "date", "depreciationAndAmortization", "capitalExpenditure"
    ])
