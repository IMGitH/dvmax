import requests
import polars as pl
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import datetime


class StockAnalyzer:
    def __init__(self, output_dir_path="."):
        self.base_url = "https://financialmodelingprep.com/api/v3"
        load_dotenv()
        self.api_key = os.getenv("FMP_API_KEY")
        self.dfs = {}
        self.output_dir_path = output_dir_path
        os.system(f"mkdir -p {self.output_dir_path}")

    def fetch_and_save_all_metrics(self, ticker, start_date, end_date,
                                   time_increment='quarter',
                                   pre_clean=True):
        def clean(ticker):
            ticker_path = self.get_ticker_out_path(ticker)
            os.system(f"rm -rf {ticker_path}")
        if pre_clean:
            clean(ticker)
        headers = {"User-Agent": "Mozilla/5.0"}
        # Fetch and save dividend data
        dividend_df = self.get_dividend_df(ticker, start_date, end_date, headers)
        if dividend_df.height > 0:
            self._save_metric_df(dividend_df.select(['date', 'dividend']), ticker, 'dividend')
        # Fetch and save ratios
        ratios_df = self._get_or_mock_ratios(ticker, time_increment)
        for col in ['peRatio', 'dividendYield', 'payoutRatio']:
            if col in ratios_df.columns:
                self._save_metric_df(ratios_df.select(['date', col]), ticker, col)
        # Fetch and save free cash flow
        cf_df = self._get_or_mock_cashflow(ticker, time_increment)
        self._save_metric_df(cf_df.select(['date', 'freeCashFlow']), ticker, 'freeCashFlow')
        # Calculate and save 6M and 12M price returns
        returns_df = self.calculate_price_returns(ticker, start_date, end_date)
        if returns_df.height > 0:
            for col in ['return_6m', 'return_12m']:
                self._save_metric_df(returns_df.select(['date', col]), ticker, col)

    def _save_metric_df(self, df: pl.DataFrame, ticker, metric_name):
        if df.height == 0:
            print(f"[WARN] No data to save for {ticker} - {metric_name}")
            return
        ticker_path = self.get_ticker_out_path(ticker)
        os.system(f"mkdir -p {ticker_path}")
        file_name = f"{metric_name}.parquet"
        file_path = os.path.join(ticker_path, file_name)
        df.write_parquet(file_path)
        print(f"[INFO] Saved {metric_name} data for {ticker} to {file_path}")

    def get_ticker_out_path(self, ticker):
        ticker_path = os.path.join(self.output_dir_path, ticker.lower())
        return ticker_path

    def get_dividend_df(self, ticker, start_date, end_date, headers):
        dividend_url = f"{self.base_url}/historical-price-full/stock_dividend/{ticker}?from={start_date}&to={end_date}&apikey={self.api_key}"
        dividend_resp = requests.get(dividend_url, headers=headers)
        dividend_data = dividend_resp.json().get('historical', [])
        if not dividend_data:
            return pl.DataFrame()
        df = pl.DataFrame(dividend_data)
        return df.with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"))

    def _get_or_mock_ratios(self, ticker, time_increment) -> pl.DataFrame:
        url = f"{self.base_url}/ratios/{ticker}?period={time_increment}&limit=100&apikey={self.api_key}"
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
            df = pl.DataFrame(data)
            return df.with_columns(pl.col("date").str.strptime(pl.Date, fmt="%Y-%m-%d")).select(['date', 'peRatio', 'payoutRatio', 'dividendYield'])
        except Exception:
            print("[WARN] Using mock ratios data.")
            mock_data = [
                {'date': '2024-12-31', 'peRatio': 28.5, 'payoutRatio': 0.18, 'dividendYield': 0.005},
                {'date': '2024-09-30', 'peRatio': 27.2, 'payoutRatio': 0.17, 'dividendYield': 0.0045},
                {'date': '2024-06-30', 'peRatio': 29.1, 'payoutRatio': 0.19, 'dividendYield': 0.0048}
            ]
            df = pl.DataFrame(mock_data)
            return df.with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"))

    def _get_or_mock_cashflow(self, ticker, time_increment) -> pl.DataFrame:
        url = f"{self.base_url}/cash-flow-statement/{ticker}?period={time_increment}&limit=100&apikey={self.api_key}"
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
            df = pl.DataFrame(data)
            return df.with_columns(pl.col("date").str.strptime(pl.Date, fmt="%Y-%m-%d")).select(['date', 'freeCashFlow'])
        except Exception:
            print("[WARN] Using mock cash flow data.")
            mock_data = [
                {'date': '2024-12-31', 'freeCashFlow': 21000000000},
                {'date': '2024-09-30', 'freeCashFlow': 19000000000},
                {'date': '2024-06-30', 'freeCashFlow': 22000000000}
            ]
            df = pl.DataFrame(mock_data)
            return df.with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"))

    def _get_or_mock_price_history(self, ticker, start_date, end_date) -> pl.DataFrame:
        url = f"{self.base_url}/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={self.api_key}"
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json().get("historical", [])
            if not data:
                raise ValueError("Empty price history")
            df = pl.DataFrame(data)
            return df.with_columns(pl.col("date").str.strptime(pl.Date, fmt="%Y-%m-%d")).select(["date", "close"])
        except Exception:
            print("[WARN] Using mock price data.")
            mock_data = [
                {'date': '2024-12-31', 'close': 190.0},
                {'date': '2024-06-30', 'close': 165.0},
                {'date': '2023-12-31', 'close': 145.0}
            ]
            df = pl.DataFrame(mock_data)
            return df.with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"))
   
    def calculate_price_returns(self, ticker, start_date, end_date) -> pl.DataFrame:
        today = datetime.date.today()
        price_df = self._get_or_mock_price_history(ticker, start_date, end_date).sort("date")

        latest_price_df = price_df.filter(pl.col("date") <= today).sort("date", descending=True)
        if latest_price_df.height == 0:
            print(f"[WARN] No recent price data to calculate returns for {ticker}.")
            return pl.DataFrame()

        latest_price = latest_price_df[0, "close"]

        # Use helper to get historical prices
        return_6m = self._calc_return_from_price_df(price_df, latest_price, today, months_back=6, buffer_days=30)
        return_12m = self._calc_return_from_price_df(price_df, latest_price, today, months_back=12, buffer_days=45)

        if return_6m is not None:
            print(f"[INFO] {ticker} - 6M Return: {return_6m:.2%}")
        else:
            print(f"[WARN] {ticker} - 6M return unavailable")

        if return_12m is not None:
            print(f"[INFO] {ticker} - 12M Return: {return_12m:.2%}")
        else:
            print(f"[WARN] {ticker} - 12M return unavailable")

        return pl.DataFrame({
            "date": [today],
            "return_6m": [return_6m],
            "return_12m": [return_12m]
        })
    
    def _calc_return_from_price_df(self, df: pl.DataFrame, latest_price: float, today: datetime.date, months_back: int, buffer_days: int) -> float | None:
        from datetime import timedelta

        target_days = months_back * 30  # approximate months
        cutoff = today - timedelta(days=target_days)
        min_date = cutoff - timedelta(days=buffer_days)

        hist_df = df.filter(
            (pl.col("date") <= cutoff) & (pl.col("date") >= min_date)
        ).sort("date", descending=True)

        if hist_df.height == 0:
            return None

        historical_price = hist_df[0, "close"]
        return (latest_price - historical_price) / historical_price

    def plot_metrics(self, ticker, metrics=None):
        ticker_path = self.get_ticker_out_path(ticker)
        if not os.path.exists(ticker_path):
            raise FileNotFoundError(f"Data path for {ticker} does not exist: {ticker_path}")

        if metrics is None:
            metrics = ['dividend', 'peRatio', 'payoutRatio', 'dividendYield', 'freeCashFlow', 'return_6m', 'return_12m']

        for metric in metrics:
            file_path = os.path.join(ticker_path, f"{metric}.parquet")
            if not os.path.exists(file_path):
                print(f"[WARN] Missing file for metric: {metric}")
                continue
            df = pl.read_parquet(file_path).sort("date")
            plt.figure(figsize=(10, 5))
            plt.plot(df['date'].to_numpy(), df[metric].to_numpy(), marker='o', label=metric)
            plt.xlabel('Date')
            plt.ylabel(metric)
            plt.title(f"{metric} over time for {ticker.upper()}")
            plt.grid(True)
            plt.tight_layout()
            plt.xticks(rotation=45)
            plt.legend()
            output_img = os.path.join(ticker_path, f"{metric}.png")
            plt.savefig(output_img)
            print(f"[INFO] Saved plot for {metric} to {output_img}")
            plt.close()

if __name__ == "__main__":
    analyzer = StockAnalyzer("./data")
    analyzer.fetch_and_save_all_metrics('AAPL', '2020-01-01', '2023-12-31', time_increment='quarter')
    analyzer.plot_metrics('AAPL')
