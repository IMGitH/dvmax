from src.dataprep.fetcher.base import FMPClient

def fetch_company_profile(ticker: str) -> dict:
    client = FMPClient()
    data = client.fetch(f"profile/{ticker}")
    return data[0] if data else {}
