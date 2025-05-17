from src.dataprep.fetcher import (
    fetch_dividends, 
    fetch_prices, 
    fetch_ratios,
    fetch_income_statement_fund,
    fetch_balance_sheet_fund,
    fetch_cashflow_statement_fund
)
import polars as pl
import datetime


def test_fetch_dividends_returns_valid_dataframe():
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    df = fetch_dividends(ticker, start_date=start_date, end_date=end_date)

    print("\n=== test_fetch_dividends_returns_valid_dataframe ===")
    print(f"Input: ticker={ticker}, start_date={start_date}, end_date={end_date}")
    print(df.head())

    assert df.height > 0
    assert "date" in df.columns
    assert "dividend" in df.columns


def _check_fetch_prices(ticker: str, start_date: str, end_date: str, label: str):
    df = fetch_prices(ticker, start_date=start_date, end_date=end_date)

    print(f"\n=== {label} ===")
    print(f"Input: ticker={ticker}, start_date={start_date}, end_date={end_date}")
    print(f"Returned rows: {df.height}")
    print("First few rows:")
    print(df.head())

    if df.schema["date"] == pl.Utf8:
        df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    df = df.sort("date")

    first_row = df[0]
    last_row = df[-1]
    first_date = first_row["date"].item()
    last_date = last_row["date"].item()

    print(f"\nFirst row:\n{first_row}")
    print(f"Last row:\n{last_row}")
    print(f"Returned date range: {first_date} to {last_date}")

    assert df.height > 0
    assert "date" in df.columns
    assert "close" in df.columns
    assert first_date >= datetime.date.fromisoformat(start_date)
    assert last_date <= datetime.date.fromisoformat(end_date)
    assert last_date > first_date


def test_fetch_prices_returns_valid_dataframe():
    _check_fetch_prices(
        ticker="AAPL",
        start_date="2020-01-01",
        end_date="2024-01-01",
        label="test_fetch_prices_returns_valid_dataframe"
    )


def test_fetch_prices_returns_valid_dataframe_large_range():
    _check_fetch_prices(
        ticker="AAPL",
        start_date="2005-01-01",
        end_date="2024-01-01",
        label="test_fetch_prices_returns_valid_dataframe_large_range"
    )


def _check_ratios_dataframe(df: pl.DataFrame, ticker: str, period: str):
    print("\n=== test_fetch_ratios_returns_valid_dataframe ===")
    print(f"Input: ticker={ticker}, period={period}")
    print(df.head())

    assert df.height > 0
    assert "date" in df.columns

    required_columns = [
        "priceEarningsRatio", "payoutRatio", "dividendYield",
        "priceToSalesRatio", "enterpriseValueMultiple", "priceFairValue",
        "returnOnEquity", "debtEquityRatio", "netProfitMargin"
    ]
    for col in required_columns:
        assert col in df.columns


def test_fetch_ratios_returns_valid_dataframe():
    print("\n=== test_fetch_ratios_returns_valid_dataframe ===")
    df = fetch_ratios("AAPL", "annual")
    print(f"Returned {df.height} rows")
    _check_ratios_dataframe(df, ticker="AAPL", period="annual")


def test_fetch_ratios_limited_years():
    print("\n=== test_fetch_ratios_limited_years ===")
    df = fetch_ratios("AAPL", limit=2)
    print(f"Returned {df.height} rows")
    print(df.head())

    assert df.height == 2
    assert "date" in df.columns
    assert "priceEarningsRatio" in df.columns


def test_fetch_income_statement_fund_returns_valid_dataframe():
    ticker = "AAPL"
    df = fetch_income_statement_fund(ticker)

    print("\n=== test_fetch_income_statement_fund_returns_valid_dataframe ===")
    print(f"Input: ticker={ticker}")
    print(f"Returned rows: {df.height}")
    print(df.head())

    if df.schema["date"] == pl.Utf8:
        df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    assert df.height > 0
    assert "date" in df.columns
    assert "incomeBeforeTax" in df.columns
    assert "interestExpense" in df.columns


def test_fetch_balance_sheet_fund_returns_valid_dataframe():
    ticker = "AAPL"
    df = fetch_balance_sheet_fund(ticker)

    print("\n=== test_fetch_balance_sheet_fund_returns_valid_dataframe ===")
    print(f"Input: ticker={ticker}")
    print(df.head())

    assert df.height > 0
    assert "date" in df.columns
    assert "cashAndShortTermInvestments" in df.columns
    assert "totalDebt" in df.columns


def test_fetch_cashflow_statement_fund_returns_valid_dataframe():
    ticker = "AAPL"
    df = fetch_cashflow_statement_fund(ticker)

    print("\n=== test_fetch_cashflow_statement_fund_returns_valid_dataframe ===")
    print(f"Input: ticker={ticker}")
    print(df.head())

    assert df.height > 0
    assert "date" in df.columns
    assert "depreciationAndAmortization" in df.columns
    assert "capitalExpenditure" in df.columns
