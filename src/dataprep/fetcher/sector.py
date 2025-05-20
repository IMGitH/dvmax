import polars as pl
from src.dataprep.fetcher.company import fetch_company_profile
from src.dataprep.fetcher.prices import fetch_prices
from src.dataprep.fetcher.sector_constants import SECTOR_TO_ETF, SECTOR_NORMALIZATION


def extract_sector_name(profile) -> str:
    sector = ""
    if isinstance(profile, pl.DataFrame):
        if "sector" in profile.columns and profile.height > 0:
            sector = profile[0, "sector"]
    elif isinstance(profile, dict):
        sector = profile.get("sector", "")
    
    return SECTOR_NORMALIZATION.get(sector, sector) if sector else ""

def extract_sector_name(profile) -> str:
    if isinstance(profile, pl.DataFrame):
        if "sector" in profile.columns and profile.height > 0:
            return profile[0, "sector"]
    elif isinstance(profile, dict):
        return profile.get("sector", "")
    return ""

def fetch_sector_index(ticker: str, limit: int = 3, profile:str = None) -> pl.DataFrame:
    """
    Determines the appropriate sector ETF for a stock and fetches its historical price data.

    If the sector is not found or unmapped, falls back to SPY.
    """
    if profile is None or \
        (isinstance(profile, pl.DataFrame) and profile.is_empty()) or \
            (isinstance(profile, dict) and not profile):
        profile = fetch_company_profile(ticker)
    sector_name = extract_sector_name(profile)
    sector_etf = SECTOR_TO_ETF.get(sector_name, "SPY")  # fallback

    return fetch_prices(sector_etf, lookback_years=limit)
