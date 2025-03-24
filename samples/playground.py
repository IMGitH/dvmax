import requests
import pandas as pd
from datetime import datetime
import os


class StockAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("FMP_KEY")

    def fetch_fmp_fundamentals(self, ticker, start_date, end_date, time_increment='quarter'):
        """
        Fetches dividend-related and fundamental intrinsic data from FMP API for a given ticker.
        
        Args:
            ticker (str): The stock ticker (e.g., 'AAPL').
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            time_increment (str): 'quarter' or 'annual'.
            
        Returns:
            pd.DataFrame: DataFrame containing dividend and fundamental metrics over time.
        """
        
        base_url = "https://financialmodelingprep.com/api/v3"
        headers = {"User-Agent": "Mozilla/5.0"}

        # Historical dividend data
        dividend_url = f"{base_url}/historical-price-full/stock_dividend/{ticker}?from={start_date}&to={end_date}&apikey={self.api_key}"
        dividend_resp = requests.get(dividend_url, headers=headers)
        dividend_data = dividend_resp.json().get('historical', [])

        dividend_df = pd.DataFrame(dividend_data)
        dividend_df['date'] = pd.to_datetime(dividend_df['date'])
        
        # Fundamental metrics (Income statement, Balance sheet, Cash flow)
        fmp_modules = {
            'ratios': f"{base_url}/ratios/{ticker}?period={time_increment}&limit=100&apikey={api_key}",
            'cash_flow': f"{base_url}/cash-flow-statement/{ticker}?period={time_increment}&limit=100&apikey={api_key}"
        }

        ratios_resp = requests.get(fmp_modules['ratios'], headers=headers)
        ratios_data = ratios_resp.json()
        ratios_df = pd.DataFrame(ratios_data)
        ratios_df['date'] = pd.to_datetime(ratios_df['date'])

        # Extract only useful columns
        ratios_df = ratios_df[['date', 'peRatio', 'dividendYield', 'payoutRatio']]

        cf_resp = requests.get(fmp_modules['cash_flow'], headers=headers)
        cf_data = cf_resp.json()
        cf_df = pd.DataFrame(cf_data)
        cf_df['date'] = pd.to_datetime(cf_df['date'])
        cf_df = cf_df[['date', 'freeCashFlow']]

        # Merge all data
        merged_df = pd.merge(ratios_df, cf_df, on='date', how='outer')
        if not dividend_df.empty:
            merged_df = pd.merge(merged_df, dividend_df[['date', 'dividend']], on='date', how='outer')

        # Filter by date range
        mask = (merged_df['date'] >= pd.to_datetime(start_date)) & (merged_df['date'] <= pd.to_datetime(end_date))
        result_df = merged_df.loc[mask].sort_values(by='date')

        return result_df.reset_index(drop=True)

if __name__ == "__main__":
    a = StockAnalyzer ()
    df = a.fetch_fmp_fundamentals('AAPL', '2020-01-01', '2023-12-31', time_increment='quarter')
    print(df)
