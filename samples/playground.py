import yfinance as yf
import pandas as pd
import datetime

# --- CONFIG ---
STOCK_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Start small, scale later
START_DATE = '2010-01-01'
END_DATE = '2024-01-01'
SP500_TICKER = '^GSPC'
FED_RATE_TICKER = '^IRX'  # 13-week Treasury Bill rate (proxy for Fed Rate)

# --- Fetch Historical Prices ---
def get_stock_data(ticker):
    data = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1mo')
    data = data[['Adj Close']]
    data.rename(columns={'Adj Close': ticker}, inplace=True)
    return data

# --- Fetch All Data ---
def fetch_all_data():
    price_data = []
    for ticker in STOCK_TICKERS + [SP500_TICKER]:
        df = get_stock_data(ticker)
        price_data.append(df)

    merged_prices = pd.concat(price_data, axis=1)
    merged_prices = merged_prices.dropna()

    # Fetch Fed Rate (monthly)
    fed_rate = yf.download(FED_RATE_TICKER, start=START_DATE, end=END_DATE, interval='1mo')
    fed_rate = fed_rate[['Adj Close']].rename(columns={'Adj Close': 'FedRate'})
    fed_rate = fed_rate / 100  # Convert to decimal rate

    merged = merged_prices.merge(fed_rate, left_index=True, right_index=True)
    return merged

# --- Calculate 12M Returns ---
def calculate_returns(df):
    returns = df.pct_change(periods=12)
    returns = returns.shift(-12)  # Align future return with current date
    return returns

# --- Create Label ---
def create_labels(stock_returns, sp500_returns, fed_rate):
    labels = {}
    for ticker in STOCK_TICKERS:
        beat_sp500 = stock_returns[ticker] >= sp500_returns
        beat_fed = stock_returns[ticker] >= fed_rate['FedRate']
        labels[ticker] = (beat_sp500 & beat_fed).astype(int)
    return pd.DataFrame(labels, index=stock_returns.index)

# --- Main ---
if __name__ == "__main__":
    merged_data = fetch_all_data()
    
    # Split prices
    price_df = merged_data[STOCK_TICKERS]
    sp500_df = merged_data[[SP500_TICKER]]
    fed_df = merged_data[['FedRate']]

    stock_returns = calculate_returns(price_df)
    sp500_returns = calculate_returns(sp500_df).rename(columns={SP500_TICKER: 'SP500'})

    labels = create_labels(stock_returns, sp500_returns['SP500'], fed_df)

    # Preview
    print("Sample Data:")
    print(labels.head())

    # Save for model training
    stock_returns.to_csv("stock_returns.csv")
    labels.to_csv("stock_labels.csv")
