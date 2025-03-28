import requests
import pandas as pd # consider using polars in the future
from dotenv import load_dotenv
import os


class StockAnalyzer:
    def __init__(self):
        self.base_url = "https://financialmodelingprep.com/api/v3"
        load_dotenv()
        self.api_key = os.getenv("FMP_API_KEY")

    def fetch_fmp_fundamentals(self, ticker, start_date, end_date, time_increment='quarter'):
        headers = {"User-Agent": "Mozilla/5.0"}

        # Fetch dividend data (usually available in free mode)
        dividend_df = self.get_dividend_df(ticker, start_date, end_date, headers)

        # Try fetching real data, else load mock
        ratios_df = self._get_or_mock_ratios(ticker, time_increment)
        cf_df = self._get_or_mock_cashflow(ticker, time_increment)

        # Merge and process
        merged_df = pd.merge(ratios_df, cf_df, on='date', how='outer')
        if not dividend_df.empty:
            merged_df = pd.merge(merged_df, dividend_df[['date', 'dividend']], on='date', how='outer')

        mask = (merged_df['date'] >= pd.to_datetime(start_date)) & (merged_df['date'] <= pd.to_datetime(end_date))
        result_df = merged_df.loc[mask].sort_values(by='date')

        return result_df.reset_index(drop=True)

    def get_dividend_df(self, ticker, start_date, end_date, headers):
        dividend_url = f"{self.base_url}/historical-price-full/stock_dividend/{ticker}?from={start_date}&to={end_date}&apikey={self.api_key}"
        dividend_resp = requests.get(dividend_url, headers=headers)
        dividend_data = dividend_resp.json().get('historical', [])
        dividend_df = pd.DataFrame(dividend_data)
        if not dividend_df.empty:
            dividend_df['date'] = pd.to_datetime(dividend_df['date'])
        return dividend_df

    def _get_or_mock_ratios(self, ticker, time_increment):
        url = f"{self.base_url}/ratios/{ticker}?period={time_increment}&limit=100&apikey={self.api_key}"
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            return df[['date', 'peRatio', 'dividendYield', 'payoutRatio']]
        except Exception:
            print("[INFO] Using mock ratios data.")
            mock_data = [
                {'date': '2023-12-31', 'peRatio': 28.5, 'dividendYield': 0.006, 'payoutRatio': 0.18},
                {'date': '2023-09-30', 'peRatio': 27.2, 'dividendYield': 0.006, 'payoutRatio': 0.17},
                {'date': '2023-06-30', 'peRatio': 29.1, 'dividendYield': 0.006, 'payoutRatio': 0.19}
            ]
            df = pd.DataFrame(mock_data)
            df['date'] = pd.to_datetime(df['date'])
            return df

    def _get_or_mock_cashflow(self, ticker, time_increment):
        url = f"{self.base_url}/cash-flow-statement/{ticker}?period={time_increment}&limit=100&apikey={self.api_key}"
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            return df[['date', 'freeCashFlow']]
        except Exception:
            print("[INFO] Using mock cash flow data.")
            mock_data = [
                {'date': '2023-12-31', 'freeCashFlow': 21000000000},
                {'date': '2023-09-30', 'freeCashFlow': 19000000000},
                {'date': '2023-06-30', 'freeCashFlow': 22000000000}
            ]
            df = pd.DataFrame(mock_data)
            df['date'] = pd.to_datetime(df['date'])
            return df


if __name__ == "__main__":
    analyzer = StockAnalyzer()
    df = analyzer.fetch_fmp_fundamentals('AAPL', '2020-01-01', '2023-12-31', time_increment='quarter')
    print(df)
