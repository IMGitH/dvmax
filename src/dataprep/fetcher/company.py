from typing import Literal
import yfinance as yf
from src.dataprep.fetcher.base import FMPClient

def fetch_company_profile(ticker: str, mode: Literal["fmp", "yfinance"] = "yfinance") -> dict:
    if mode == "yfinance":
        info = yf.Ticker(ticker).info
        if not info:
            raise RuntimeError(f"No company info from yfinance for {ticker}")
        return info

    client = FMPClient()
    data = client.fetch(f"profile/{ticker}")
    return data[0] if data else {}
