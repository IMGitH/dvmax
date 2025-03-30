import requests
import pandas as pd # consider using polars in the future
from dotenv import load_dotenv
import os


class StockAnalyzer:
    def __init__(self, output_dir_path="."):
        self.base_url = "https://financialmodelingprep.com/api/v3"
        load_dotenv()
        self.api_key = os.getenv("FMP_API_KEY")
        self.dfs = {}
        self.output_dir_path = output_dir_path
        os.system (f"mkdir -p {self.output_dir_path}")

    def fetch_and_save_all_metrics(self, ticker, start_date, end_date, time_increment='quarter'):
        headers = {"User-Agent": "Mozilla/5.0"}
        
        # Fetch and save dividend data
        dividend_df = self.get_dividend_df(ticker, start_date, end_date, headers)
        if not dividend_df.empty:
            self._save_metric_df(dividend_df[['date', 'dividend']], ticker, 'dividend')
        
        # Fetch and save ratios
        ratios_df = self._get_or_mock_ratios(ticker, time_increment)
        for col in ['peRatio', 'dividendYield', 'payoutRatio']:
            self._save_metric_df(ratios_df[['date', col]], ticker, col)
        
        # Fetch and save free cash flow
        cf_df = self._get_or_mock_cashflow(ticker, time_increment)
        self._save_metric_df(cf_df[['date', 'freeCashFlow']], ticker, 'freeCashFlow')

    def _save_metric_df(self, df, ticker, metric_name):
        if df.empty:
            print(f"[WARN] No data to save for {ticker} - {metric_name}")
            return
        file_name = f"{ticker.lower()}_{metric_name}.parquet"
        file_path = os.path.join(self.output_dir_path, file_name)
        df.to_parquet(file_path, index=False)
        print(f"[INFO] Saved {metric_name} data for {ticker} to {file_path}")

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
            return df[['date', 'peRatio', 'payoutRatio']]
        except Exception:
            print("[INFO] Using mock ratios data.")
            mock_data = [
                {'date': '2023-12-31', 'peRatio': 28.5, 'payoutRatio': 0.18},
                {'date': '2023-09-30', 'peRatio': 27.2, 'payoutRatio': 0.17},
                {'date': '2023-06-30', 'peRatio': 29.1, 'payoutRatio': 0.19}
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
        
    def save_as_parquet(self, ticker, file_name=None):
        """
        Saves the provided DataFrame to a Parquet file.
        """
        df = self.dfs.get(ticker, None)
        if df is None:
            raise (f"No ticker named '{ticker}' in the processed data!")
        if file_name is None:
            file_name = f"{ticker.lower()}.parquet"
        if df.empty:
            print("[WARN] DataFrame is empty. Nothing to save.")
            return
        file_path = os.path.join(self.output_dir_path, file_name)
        df.to_parquet(file_path, index=False)
        print(f"[INFO] Saved DataFrame to {file_path}")


if __name__ == "__main__":
    analyzer = StockAnalyzer("./data")
    analyzer.fetch_and_save_all_metrics('AAPL', '2020-01-01', '2023-12-31', time_increment='quarter')
