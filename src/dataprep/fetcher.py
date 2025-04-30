import os
import requests
import polars as pl
from dotenv import load_dotenv

class StockFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def _fetch_endpoint(self, endpoint: str, params: dict) -> dict:
        """Generic internal fetcher for any FMP endpoint."""
        url = f"{self.base_url}/{endpoint}"
        headers = {"User-Agent": "Mozilla/5.0"}
        params["apikey"] = self.api_key

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

    def fetch_metric(self, ticker: str, start_date: str, end_date: str, endpoint_template: str, fields: list) -> pl.DataFrame:
        """Generic fetcher to retrieve specific fields for a given metric."""
        params = {"from": start_date, "to": end_date}
        data = self._fetch_endpoint(endpoint_template.format(ticker=ticker), params).get("historical", [])

        if not data:
            return pl.DataFrame()

        df = pl.DataFrame(data)
        return df.select(fields).with_columns(
            pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
        )

    def fetch_dividends(self, ticker: str, start_date: str, end_date: str) -> pl.DataFrame:
        """Fetch dividends history for a stock from FMP."""
        return self.fetch_metric(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            endpoint_template="historical-price-full/stock_dividend/{ticker}",
            fields=["date", "dividend"]
        )

    def fetch_prices(self, ticker: str, start_date: str, end_date: str) -> pl.DataFrame:
        """Fetch historical closing prices for a stock from FMP."""
        return self.fetch_metric(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            endpoint_template="historical-price-full/{ticker}",
            fields=["date", "close"]
        )

    def fetch_ratios(self, ticker: str) -> pl.DataFrame:
        """Fetch financial ratios history for a stock from FMP."""
        data = self._fetch_endpoint(f"ratios/{ticker}", params={})

        if not data:
            return pl.DataFrame()

        df = pl.DataFrame(data)
        return df.select([
            "date",
            "priceEarningsRatio",
            "payoutRatio",
            "dividendYield",
            "priceToSalesRatio",
            "enterpriseValueMultiple",
            "priceFairValue"
        ]).with_columns(
            pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
        )