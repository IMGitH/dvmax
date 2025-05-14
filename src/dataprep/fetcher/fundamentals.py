import polars as pl
from datetime import datetime
from src.dataprep.fetcher.base import FMPClient
from src.dataprep.fetcher.utils import default_date_range

def _fetch_fundamental(endpoint: str, ticker: str, period: str, start_date: str | None, end_date: str | None) -> pl.DataFrame:
    if period not in {"annual", "quarter"}:
        raise ValueError("Period must be 'annual' or 'quarter'")

    start_date, end_date = start_date or default_date_range()[0], end_date or default_date_range()[1]

    client = FMPClient()
    data = client.fetch(f"{endpoint}/{ticker}", {"period": period} if period == "quarter" else {})
    if not data:
        return pl.DataFrame()

    df = pl.DataFrame(data).with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
    )

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))

def fetch_income_statement(ticker: str, period="annual", start_date=None, end_date=None) -> pl.DataFrame:
    return _fetch_fundamental("income-statement", ticker, period, start_date, end_date).select([
        "date", "incomeBeforeTax", "interestExpense"
    ])

def fetch_balance_sheet(ticker: str, period="annual", start_date=None, end_date=None) -> pl.DataFrame:
    return _fetch_fundamental("balance-sheet-statement", ticker, period, start_date, end_date).select([
        "date", "cashAndShortTermInvestments", "totalDebt"
    ])

def fetch_cashflow_statement(ticker: str, period="annual", start_date=None, end_date=None) -> pl.DataFrame:
    return _fetch_fundamental("cash-flow-statement", ticker, period, start_date, end_date).select([
        "date", "depreciationAndAmortization", "capitalExpenditure"
    ])


if __name__ == "__main__":
    df = fetch_income_statement("AAPL", period="annual")
