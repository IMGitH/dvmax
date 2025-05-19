import polars as pl
from src.dataprep.fetcher.company import fetch_company_profile
from src.dataprep.fetcher.prices import fetch_prices

SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Basic Materials": "XLB",  # sometimes shown as "Materials"
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

def fetch_sector_index(ticker: str, limit: int = 3, profile:str = None) -> pl.DataFrame:
    """
    Determines the appropriate sector ETF for a stock and fetches its historical price data.

    If the sector is not found or unmapped, falls back to SPY.
    """
    profile = profile or fetch_company_profile(ticker)
    sector_name = profile.get("sector", "")
    sector_etf = SECTOR_TO_ETF.get(sector_name, "SPY")  # fallback

    return fetch_prices(sector_etf, lookback_years=limit)
