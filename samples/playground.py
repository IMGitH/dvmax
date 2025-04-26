import requests
import polars as pl
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import datetime
import math
import numpy as np

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
                {"date": "2025-03-31", "peRatio": 26.7, "payoutRatio": 0.18, "dividendYield": 0.0053},
                {'date': '2024-12-31', 'peRatio': 28.5, 'payoutRatio': 0.18, 'dividendYield': 0.005},
                {'date': '2024-09-30', 'peRatio': 27.2, 'payoutRatio': 0.17, 'dividendYield': 0.0045},
                {'date': '2024-06-30', 'peRatio': 29.1, 'payoutRatio': 0.19, 'dividendYield': 0.0048},
                {'date': '2024-03-31', 'peRatio': 25.1, 'payoutRatio': 0.17, 'dividendYield': 0.0049}
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
                {"date": "2025-03-31", "freeCashFlow": 20000000000},
                {'date': '2024-12-31', 'freeCashFlow': 21000000000},
                {'date': '2024-09-30', 'freeCashFlow': 19000000000},
                {'date': '2024-06-30', 'freeCashFlow': 22000000000},
                {'date': '2024-03-31', 'freeCashFlow': 18000000000},
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
                {"date": "2025-03-31", "close": 150.0},
                {'date': '2024-12-31', 'close': 120.0},
                {'date': '2024-09-30', 'close': 110.0},
                {'date': '2024-06-30', 'close': 120.0},
                {'date': '2024-03-31', 'close': 100.0}
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
    
    def _calc_return_from_price_df(
        self,
        df: pl.DataFrame,
        latest_price: float,
        today: datetime.date,
        months_back: int,
        buffer_days: int
    ) -> float | None:
        from datetime import timedelta

        target_days = months_back * 30  # approximate months
        cutoff = today - timedelta(days=target_days)
        min_date = cutoff - timedelta(days=buffer_days)
        max_date = cutoff + timedelta(days=buffer_days)

        # Narrow to relevant date range
        hist_df = df.filter(
            (pl.col("date") >= min_date) & (pl.col("date") <= max_date)
        ).with_columns(
            (pl.col("date").cast(pl.Date) - pl.lit(cutoff)).abs().alias("date_diff")
        ).sort("date_diff")

        if hist_df.height == 0:
            return None

        historical_price = hist_df[0, "close"]
        return (latest_price - historical_price) / historical_price


    def plot_metrics(self, ticker, metrics=None):
        ticker_path = self.get_ticker_out_path(ticker)
        if not os.path.exists(ticker_path):
            raise FileNotFoundError(f"Data path for {ticker} does not exist: {ticker_path}")

        if metrics is None:
            metrics = [
    'dividend', 'peRatio', 'payoutRatio', 'dividendYield', 'freeCashFlow',
    'return_6m', 'return_12m', 'volatility', 'max_drawdown',
    'div_cagr_3y', 'div_cagr_5y', 'consec_increase_years',
    'eps_growth_3y', 'net_debt_to_ebitda', 'roe', 'roa', 'net_margin',
    'relative_return_6m', 'relative_return_12m',
    'interest_rate', 'gdp_growth', 'inflation', 'fx_volatility'
]

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


# -- EXTENSION FOR MISSING METRICS --
# This adds volatility, max drawdown, dividend CAGR, consecutive increase years,
# EPS Growth, Net Debt/EBITDA, ROE/ROA/Margins, macro indicators, and sector relative return.


class StockAnalyzerExtended(StockAnalyzer):
    def get_sector_from_fmp(self, ticker: str) -> str | None:
        url = f"{self.base_url}/profile/{ticker}?apikey={self.api_key}"
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
            if data and 'sector' in data[0]:
                sector = data[0]['sector']
                print(f"[INFO] Sector for {ticker.upper()}: {sector}")
                return sector
            else:
                print(f"[WARN] Sector not found in profile for {ticker}")
                return None
        except Exception as e:
            print(f"[WARN] Failed to fetch sector for {ticker}: {e}")
            return None
    ETF_MAPPING = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLE": "Energy",
        "XLV": "Health Care",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLU": "Utilities",
        "XLC": "Communication Services"
        # Add more if needed
    }

    def analyze_additional_metrics(self, ticker, start_date, end_date):
        combined_metrics = []
        price_df = self._get_or_mock_price_history(ticker, start_date, end_date).sort("date")
        if price_df.height == 0:
            print(f"[WARN] No price data for extended metrics on {ticker}")
            return

        volatility = self.calculate_volatility(price_df)
        drawdown = self.calculate_max_drawdown(price_df)

        today = datetime.date.today()
        df = pl.DataFrame({
            "date": [today],
            "volatility": [volatility],
            "max_drawdown": [drawdown]
        })
        self._save_metric_df(df.select(['date', 'volatility']), ticker, "volatility")
        combined_metrics.append(df.select(['date', 'volatility']))
        self._save_metric_df(df.select(['date', 'max_drawdown']), ticker, "max_drawdown")
        combined_metrics.append(df.select(['date', 'max_drawdown']))

        dividend_df = self.get_dividend_df(ticker, start_date, end_date, headers={"User-Agent": "Mozilla/5.0"})
        if dividend_df.height > 0:
            cagr3, cagr5 = self.calculate_dividend_cagr(dividend_df)
            consec_years = self.calculate_consecutive_increase_years(dividend_df)
            df = pl.DataFrame({
                "date": [today] * 3,
                "metric": ["div_cagr_3y", "div_cagr_5y", "consec_increase_years"],
                "value": [cagr3, cagr5, consec_years]
            }).pivot(index="date", values="value", on="metric")
            self._save_metric_df(df.select(['date', 'div_cagr_3y']), ticker, "div_cagr_3y")
            combined_metrics.append(df.select(['date', 'div_cagr_3y']))
        self._save_metric_df(df.select(['date', 'div_cagr_5y']), ticker, "div_cagr_5y")
        self._save_metric_df(df.select(['date', 'consec_increase_years']), ticker, "consec_increase_years")

        ratios_df = self._get_or_mock_ratios(ticker, "quarter")
        if ratios_df.height > 0:
            ratios_df = ratios_df.sort("date")
            eps_growth = self.calculate_eps_growth(ratios_df)
            roe = ratios_df["returnOnEquity"][-1] if "returnOnEquity" in ratios_df.columns else None
            roa = ratios_df["returnOnAssets"][-1] if "returnOnAssets" in ratios_df.columns else None
            margin = ratios_df["netProfitMargin"][-1] if "netProfitMargin" in ratios_df.columns else None
            debt_ebitda = ratios_df["netDebtToEBITDA"][-1] if "netDebtToEBITDA" in ratios_df.columns else None

            df = pl.DataFrame({
                "date": [today],
                "eps_growth_3y": [eps_growth],
                "net_debt_to_ebitda": [debt_ebitda],
                "roe": [roe],
                "roa": [roa],
                "net_margin": [margin]
            })
            self._save_metric_df(df.select(['date', 'eps_growth_3y']), ticker, "eps_growth_3y")
            self._save_metric_df(df.select(['date', 'net_debt_to_ebitda']), ticker, "net_debt_to_ebitda")
            self._save_metric_df(df.select(['date', 'roe']), ticker, "roe")
            self._save_metric_df(df.select(['date', 'roa']), ticker, "roa")
            self._save_metric_df(df.select(['date', 'net_margin']), ticker, "net_margin")
            combined_metrics.append(df.select(['date', 'eps_growth_3y']))
            combined_metrics.append(df.select(['date', 'net_debt_to_ebitda']))
            combined_metrics.append(df.select(['date', 'roe']))
            combined_metrics.append(df.select(['date', 'roa']))
            combined_metrics.append(df.select(['date', 'net_margin']))
            self._save_metric_df(df.select(['date', 'eps_growth_3y']), ticker, "eps_growth_3y")
        self._save_metric_df(df.select(['date', 'net_debt_to_ebitda']), ticker, "net_debt_to_ebitda")
        self._save_metric_df(df.select(['date', 'roe']), ticker, "roe")
        self._save_metric_df(df.select(['date', 'roa']), ticker, "roa")
        self._save_metric_df(df.select(['date', 'net_margin']), ticker, "net_margin")

        self._save_metric_df(df.select(['date', 'eps_growth_3y']), ticker, "eps_growth_3y")
        self._save_metric_df(df.select(['date', 'net_debt_to_ebitda']), ticker, "net_debt_to_ebitda")
        self._save_metric_df(df.select(['date', 'roe']), ticker, "roe")
        self._save_metric_df(df.select(['date', 'roa']), ticker, "roa")
        self._save_metric_df(df.select(['date', 'net_margin']), ticker, "net_margin")
        combined_metrics.append(df.select(['date', 'eps_growth_3y']))
        combined_metrics.append(df.select(['date', 'net_debt_to_ebitda']))
        combined_metrics.append(df.select(['date', 'roe']))
        combined_metrics.append(df.select(['date', 'roa']))
        combined_metrics.append(df.select(['date', 'net_margin']))

        # Sector-relative return
        sector = self.get_sector_from_fmp(ticker)
        etf = None
        if sector:
            for k, v in self.ETF_MAPPING.items():
                if v.lower() == sector.lower():
                    etf = k
                    break

        if not etf:
            print(f"[WARN] No matching ETF found for sector: {sector}")
        else:
            rel_df = self._calculate_relative_returns(ticker, etf, start_date, end_date)
            if rel_df.height:
                self._save_metric_df(rel_df.select(['date', 'relative_return_6m']), ticker, "relative_return_6m")
                self._save_metric_df(rel_df.select(['date', 'relative_return_12m']), ticker, "relative_return_12m")
                combined_metrics.append(rel_df.select(['date', 'relative_return_12m']))

        # Mock macroeconomic data
        macro_df = pl.DataFrame({
            "date": [today],
            "interest_rate": [0.045],
            "gdp_growth": [0.02],
            "inflation": [0.035]
        })
        self._save_metric_df(macro_df.select(['date', 'interest_rate']), ticker, "interest_rate")
        self._save_metric_df(macro_df.select(['date', 'gdp_growth']), ticker, "gdp_growth")
        self._save_metric_df(macro_df.select(['date', 'inflation']), ticker, "inflation")
        combined_metrics.append(macro_df.select(['date', 'interest_rate', 'gdp_growth', 'inflation']))

        # Mock currency volatility
        fx_df = pl.DataFrame({
            "date": [today],
            "fx_volatility": [0.07]  # e.g., std dev of USD/EUR over last 6 months
        })
        self._save_metric_df(fx_df.select(['date', 'fx_volatility']), ticker, "fx_volatility")
        combined_metrics.append(fx_df.select(['date', 'fx_volatility']))

            # Save combined metrics if any
        if combined_metrics:
            from functools import reduce
            try:
                merged = reduce(lambda left, right: left.join(right, on="date", how="full", suffix=None), combined_metrics)
                merged = merged.sort("date")
                self._save_metric_df(merged, ticker, "metrics_combined")
            except Exception as e:
                print(f"[WARN] Failed to create combined metrics: {e}")

    def _calculate_relative_returns(self, ticker, etf, start_date, end_date) -> pl.DataFrame:
        today = datetime.date.today()
        stock_returns = self.calculate_price_returns(ticker, start_date, end_date)
        etf_returns = self.calculate_price_returns(etf, start_date, end_date)

        if stock_returns.height == 0 or etf_returns.height == 0:
            return pl.DataFrame()

        return_6m = stock_returns[0, "return_6m"] - etf_returns[0, "return_6m"]
        return_12m = stock_returns[0, "return_12m"] - etf_returns[0, "return_12m"]

        return pl.DataFrame({
            "date": [today],
            "relative_return_6m": [return_6m],
            "relative_return_12m": [return_12m]
        })

    def calculate_volatility(self, df: pl.DataFrame) -> float:
        returns = df.select([
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("daily_return")
        ]).drop_nulls()
        std = returns["daily_return"].std()
        return std * math.sqrt(252) if std is not None else None

    def calculate_max_drawdown(self, df: pl.DataFrame) -> float:
        prices = df["close"].to_numpy()
        max_drawdown = 0
        peak = prices[0]
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def calculate_dividend_cagr(self, df: pl.DataFrame) -> tuple[float, float]:
        df = df.sort("date")
        df = df.with_columns([pl.col("date").dt.year().alias("year")])
        annual_divs = df.group_by("year").agg(pl.col("dividend").sum())
        annual_divs = annual_divs.sort("year")

        def cagr(start, end, years):
            if start == 0 or years == 0:
                return None
            return (end / start) ** (1 / years) - 1

        years = annual_divs["year"].to_list()
        values = annual_divs["dividend"].to_list()

        cagr3 = cagr(values[-4], values[-1], 3) if len(values) >= 4 else None
        cagr5 = cagr(values[-6], values[-1], 5) if len(values) >= 6 else None
        return cagr3, cagr5

    def calculate_consecutive_increase_years(self, df: pl.DataFrame) -> int:
        df = df.sort("date")
        df = df.with_columns([pl.col("date").dt.year().alias("year")])
        annual_divs = df.group_by("year").agg(pl.col("dividend").sum()).sort("year")
        values = annual_divs["dividend"].to_list()
        count = 0
        for i in range(len(values) - 1, 0, -1):
            if values[i] > values[i - 1]:
                count += 1
            else:
                break
        return count

    def calculate_eps_growth(self, df: pl.DataFrame) -> float:
        eps = df["eps"] if "eps" in df.columns else None
        if eps is None or len(eps) < 13:
            return None
        start, end = eps[-13], eps[-1]  # approx. 3Y back
        if start == 0:
            return None
        return (end / start) ** (1 / 3) - 1

if __name__ == "__main__":
    analyzer = StockAnalyzerExtended("./data")
    analyzer.fetch_and_save_all_metrics("AAPL", "2020-01-01", "2023-12-31", time_increment="quarter")
    analyzer.analyze_additional_metrics("AAPL", "2020-01-01", "2023-12-31")
    analyzer.plot_metrics("AAPL")
