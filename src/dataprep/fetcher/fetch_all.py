from src.dataprep.fetcher.prices import fetch_prices
from src.dataprep.fetcher.dividends import fetch_dividends
from src.dataprep.fetcher.ratios import fetch_ratios
from src.dataprep.fetcher.fundamentals import fetch_balance_sheet_fund, fetch_income_statement_fund
from src.dataprep.fetcher.company import fetch_company_profile
from src.dataprep.fetcher.splits import fetch_splits

def fetch_all(ticker: str, div_lookback_years: int, other_lookback_years: int) -> dict:
    return {
        "prices": fetch_prices(ticker, lookback_years=div_lookback_years),
        "dividends": fetch_dividends(ticker, lookback_years=div_lookback_years),
        "ratios": fetch_ratios(ticker, limit=other_lookback_years),
        "balance": fetch_balance_sheet_fund(ticker, limit=other_lookback_years),
        "income": fetch_income_statement_fund(ticker, limit=other_lookback_years),
        "profile": fetch_company_profile(ticker),
        "splits": fetch_splits(ticker)
    }
