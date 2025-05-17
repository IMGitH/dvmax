from src.dataprep.fetcher import (
    fetch_dividends, 
    fetch_prices, 
    fetch_ratios,
    fetch_income_statement,
    fetch_balance_sheet,
    fetch_cashflow_statement
)

def test_fetch_dividends_returns_valid_dataframe():
    df = fetch_dividends("AAPL", start_date="2020-01-01", end_date="2024-01-01")
    assert df.height > 0
    assert "date" in df.columns
    assert "dividend" in df.columns

def test_fetch_prices_returns_valid_dataframe():
    df = fetch_prices("AAPL", start_date="2020-01-01", end_date="2024-01-01")
    assert df.height > 0
    assert "date" in df.columns
    assert "close" in df.columns

def test_fetch_ratios_returns_valid_dataframe():
    df = fetch_ratios("AAPL", "annual")
    # df = fetcher.fetch_ratios("AAPL", "quarter")
    # 📌 Reminder for ML pipeline:
    assert df.height > 0
    assert "date" in df.columns
    assert "priceEarningsRatio" in df.columns
    assert "payoutRatio" in df.columns
    assert "dividendYield" in df.columns
    assert "priceToSalesRatio" in df.columns
    assert "enterpriseValueMultiple" in df.columns
    assert "priceFairValue" in df.columns
    assert "returnOnEquity" in df.columns
    assert "debtEquityRatio" in df.columns
    assert "netProfitMargin" in df.columns

def test_fetch_income_statement_returns_valid_dataframe():
    df = fetch_income_statement("AAPL", period="annual")
    assert df.height > 0
    assert "date" in df.columns
    assert "incomeBeforeTax" in df.columns
    assert "interestExpense" in df.columns

def test_fetch_balance_sheet_returns_valid_dataframe():
    df = fetch_balance_sheet("AAPL", period="annual")
    assert df.height > 0
    assert "date" in df.columns
    assert "cashAndShortTermInvestments" in df.columns
    assert "totalDebt" in df.columns

def test_fetch_cashflow_statement_returns_valid_dataframe():
    df = fetch_cashflow_statement("AAPL", period="annual")
    assert df.height > 0
    assert "date" in df.columns
    assert "depreciationAndAmortization" in df.columns
    assert "capitalExpenditure" in df.columns
