from typing import Literal
import yfinance as yf
import polars as pl
from src.dataprep.fetcher.base import FMPClient
from src.dataprep.fetcher.sector_constants import SECTOR_NORMALIZATION, SECTOR_TO_ETF

def fetch_company_profile(ticker: str, mode: Literal["auto", "fmp", "yfinance"] = "auto") -> dict:
    if mode == "yfinance":
        info = yf.Ticker(ticker).info
        return info if info else {}
    
    if mode == "fmp":
        client = FMPClient()
        data = client.fetch(f"profile/{ticker}")
        return data[0] if data else {}

    # auto fallback mode
    info = fetch_company_profile(ticker, "yfinance")
    if not info or "sector" not in info:
        info = fetch_company_profile(ticker, "fmp")
    return info


def extract_sector_name(profile: dict, ticker: str) -> str:
    sector = ""

    if isinstance(profile, pl.DataFrame):
        if "sector" in profile.columns and profile.height > 0:
            sector = profile[0, "sector"]
    elif isinstance(profile, dict):
        sector = profile.get("sector", "")

    if not sector:
        raise ValueError(f"Missing sector for ticker: {ticker}. Please investigate profile: {profile}")

    normalized = SECTOR_NORMALIZATION.get(sector, sector)

    # Optional: validate it's supported
    if normalized not in SECTOR_TO_ETF:
        raise ValueError(f"Unknown or unsupported sector '{normalized}' for {ticker}. Add to SECTOR_TO_ETF or normalize it.")

    return normalized
