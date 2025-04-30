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

        if response.status_code == 403:
            try:
                json = response.json()
                if "Exclusive Endpoint" in json.get("Error Message", ""):
                    raise PermissionError(
                        f"Access to '{endpoint}' is restricted under your current FMP plan.\n"
                        "Consider upgrading at https://site.financialmodelingprep.com/developer/docs/pricing"
                    )
            except Exception:
                pass  # fallback to generic error

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

    def fetch_ratios(self, ticker: str, period: str = "annual") -> pl.DataFrame:
        # TO-DO: START + END

        """Fetch financial ratios history for a stock from FMP.
        period: 'annual' (default) or 'quarter' for quarterly data.
        """
        if period not in {"annual", "quarter"}:
            raise ValueError("Period must be either 'annual' or 'quarter'")

        params = {"period": period if period == "quarter" else None}
        try:
            data = self._fetch_endpoint(f"ratios/{ticker}", params={k: v for k, v in params.items() if v})
        except PermissionError as e:
            print(f"[WARN] {e}")
            return pl.DataFrame()

        if not data:
            return pl.DataFrame()

        df = pl.DataFrame(data)
        return df.select([
            "date",
            "priceEarningsRatio",
            "payoutRatio",
            "priceToSalesRatio",
            "enterpriseValueMultiple",
            "priceFairValue",
            "returnOnEquity",
            "debtEquityRatio",
            "netProfitMargin"
        ]).with_columns(
            pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
        )
    