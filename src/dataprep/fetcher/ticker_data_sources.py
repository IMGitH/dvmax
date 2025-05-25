from src.dataprep.fetcher.ticker_params.prices import fetch_prices
from src.dataprep.fetcher.ticker_params.dividends import fetch_dividends
from src.dataprep.fetcher.ticker_params.ratios import fetch_ratios
from src.dataprep.fetcher.ticker_params.fundamentals import fetch_balance_sheet_fund, fetch_income_statement_fund
from src.dataprep.fetcher.ticker_params.company import fetch_company_profile
from src.dataprep.fetcher.ticker_params.splits import fetch_splits
from src.dataprep.fetcher.ticker_params.sector import fetch_sector_index
from src.dataprep.fetcher.client import fmp_client
import logging


def fetch_all_per_ticker(ticker: str, div_lookback_years: int, other_lookback_years: int) -> dict:
    fmp_client.request_count = 0

    profile = fetch_company_profile(ticker)
    result = {
        "prices": fetch_prices(ticker, lookback_years=div_lookback_years),
        "dividends": fetch_dividends(ticker, lookback_years=div_lookback_years),
        "ratios": fetch_ratios(ticker, limit=other_lookback_years),
        "balance": fetch_balance_sheet_fund(ticker, limit=other_lookback_years),
        "income": fetch_income_statement_fund(ticker, limit=other_lookback_years),
        "profile": profile,
        "splits": fetch_splits(ticker),
        "sector_index": fetch_sector_index(ticker, limit=other_lookback_years, profile=profile)
    }

    logging.info(f"🔍 Total FMP API requests for ticker {ticker}: {fmp_client.request_count}")
    return result
