import os
import requests
import polars as pl
from dotenv import load_dotenv
import datetime

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

    def _default_date_range(self) -> tuple[str, str]:
        """Return (start_date, end_date) with start = 4 years ago, end = last full quarter."""
        today = datetime.date.today()
        # Find last full quarter end
        quarter_month = (today.month - 1) // 3 * 3
        last_quarter_end = datetime.date(today.year, quarter_month, 1) - datetime.timedelta(days=1)
        start = datetime.date(last_quarter_end.year - 4, last_quarter_end.month, last_quarter_end.day)
        return start.isoformat(), last_quarter_end.isoformat()

    def fetch_metric(self, ticker: str, start_date: str | None, end_date: str | None, endpoint_template: str, fields: list) -> pl.DataFrame:
        """Generic fetcher to retrieve specific fields for a given metric."""
        if not start_date or not end_date:
            start_date, end_date = self._default_date_range()

        params = {"from": start_date, "to": end_date}
        data = self._fetch_endpoint(endpoint_template.format(ticker=ticker), params).get("historical", [])

        if not data:
            return pl.DataFrame()

        df = pl.DataFrame(data)
        return df.select(fields).with_columns(
            pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
        )

    def fetch_dividends(self, ticker: str, start_date: str | None = None, end_date: str | None = None) -> pl.DataFrame:
        """Fetch dividends history for a stock from FMP."""
        return self.fetch_metric(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            endpoint_template="historical-price-full/stock_dividend/{ticker}",
            fields=["date", "dividend"]
        )

    def fetch_prices(self, ticker: str, start_date: str | None = None, end_date: str | None = None) -> pl.DataFrame:
        """Fetch historical closing prices for a stock from FMP."""
        return self.fetch_metric(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            endpoint_template="historical-price-full/{ticker}",
            fields=["date", "close"]
        )

    def fetch_ratios(self, ticker: str, period: str = "annual", start_date: str | None = None, end_date: str | None = None) -> pl.DataFrame:
        """Fetch financial ratios history for a stock from FMP.
        period: 'annual' (default) or 'quarter' for quarterly data.
        Optional: filter using start_date and end_date.
        """
        if period not in {"annual", "quarter"}:
            raise ValueError("Period must be either 'annual' or 'quarter'")

        if not start_date or not end_date:
            start_date, end_date = self._default_date_range()

        params = {"period": period if period == "quarter" else None}
        try:
            data = self._fetch_endpoint(f"ratios/{ticker}", params={k: v for k, v in params.items() if v})
        except PermissionError as e:
            print(f"[WARN] {e}")
            return pl.DataFrame()

        if not data:
            return pl.DataFrame()

        df = pl.DataFrame(data).with_columns(
            pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
        )

        range_start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        range_end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

        df = df.filter((pl.col("date") >= range_start) & (pl.col("date") <= range_end))

        self._validate_ratio_row_count(df, ticker, period, range_start, range_end)

        return df.select([
            "date",
            "priceEarningsRatio",
            "payoutRatio",
            "priceToSalesRatio",
            "enterpriseValueMultiple",
            "priceFairValue",
            "returnOnEquity",
            "debtEquityRatio",
            "netProfitMargin",
            "dividendYield"
        ])

    def _validate_ratio_row_count(self, df: pl.DataFrame, ticker: str, period: str, start: datetime.date, end: datetime.date):
        """Raise if the expected number of rows is too low for the provided date range."""
        if df.height == 0:
            raise ValueError(f"No ratio data returned for {ticker} between {start} and {end}.")

        expected_rows = ((end.year - start.year) + 1) * (4 if period == "quarter" else 1)
        min_expected = max(1, expected_rows // 3)  # allow some missing data

        if df.height < min_expected:
            raise ValueError(
                f"Too few rows of {period} data for {ticker} between {start} and {end}: "
                f"expected at least {min_expected}, got {df.height}."
            )
