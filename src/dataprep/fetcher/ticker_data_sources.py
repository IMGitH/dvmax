from src.dataprep.fetcher.prices import fetch_prices
from src.dataprep.fetcher.dividends import fetch_dividends
from src.dataprep.fetcher.ratios import fetch_ratios
from src.dataprep.fetcher.fundamentals import fetch_balance_sheet_fund, fetch_income_statement_fund
from src.dataprep.fetcher.company import fetch_company_profile
from src.dataprep.fetcher.splits import fetch_splits
from src.dataprep.fetcher.sector import fetch_sector_index
from src.dataprep.fetcher.macro import WorldBankAPI
from src.dataprep.fetcher.client import fmp_client
from src.dataprep.constants import MACRO_INDICATORS
import logging


def fetch_macro_by_country(country: str, start: int, end: int) -> dict:
    try:
        macro_api = WorldBankAPI()
        return macro_api.fetch_macro_indicators(
            indicator_map=MACRO_INDICATORS,
            country_name=country,
            start=start,
            end=end
        )
    except Exception as e:
        logging.warning(f"[Macro] Failed to fetch macro data for {country}: {e}")
        return None


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
