import pandas as pd
import yfinance as yf
import re
from tqdm import tqdm

TICKERS_FILE = "config/us_tickers.txt"


def is_valid_ticker(ticker: str) -> bool:
    return bool(re.fullmatch(r"[A-Z.]{1,6}", ticker)) and not ticker.startswith("^")


def fetch_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, header=0)
    for table in tables:
        if "Symbol" in table.columns:
            return table["Symbol"].astype(str).tolist()
    raise ValueError("❌ Could not find S&P 500 ticker table")


def fetch_nasdaq100_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url, header=0)
    for table in tables:
        if "Ticker" in table.columns:
            return table["Ticker"].astype(str).tolist()
    raise ValueError("❌ Could not find Nasdaq-100 ticker table")


def fetch_dow30_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    tables = pd.read_html(url, header=0)
    for table in tables:
        if "Symbol" in table.columns:
            return table["Symbol"].astype(str).tolist()
    raise ValueError("❌ Could not find Dow 30 ticker table")


def fetch_russell1000_from_ishares() -> list[str]:
    url = "https://www.ishares.com/us/products/239726/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv"

    try:
        # Load with no headers, we'll detect the correct ones manually
        df = pd.read_csv(url, skiprows=10)

        # Auto-detect: first column is ticker, fourth is asset class (usually)
        df.columns = [f"col{i}" for i in range(len(df.columns))]

        # Only keep rows where the 'Asset Class' (col3 or col4) is 'Equity'
        equity_df = df[df["col3"].astype(str).str.strip() == "Equity"]

        # Filter out empty tickers and known non-tickers
        tickers = (
            equity_df["col0"]
            .astype(str)
            .str.strip()
            .loc[lambda s: s.str.fullmatch(r"[A-Z.]{1,6}")]
            .tolist()
        )

        return tickers

    except Exception as e:
        print(f"⚠️ Failed to fetch Russell 1000 from iShares: {e}")
        return []

def validate_ticker_with_yfinance(ticker: str) -> bool:
    try:
        df = yf.download(ticker, period="30d", auto_adjust=False, progress=False)

        # Defensive programming: make sure we're dealing with expected types
        if df is None:
            return False

        if isinstance(df, pd.Series):
            return False  # Sometimes yfinance misbehaves and returns a Series

        if df.empty:
            return False

        if "Close" not in df.columns:
            return False

        # This is a scalar int, so it's safe
        valid_points_series = df["Close"].notna().sum()
        valid_points = valid_points_series.iloc[0] if hasattr(valid_points_series, "iloc") else valid_points_series
        return bool(valid_points >= 5)

    except Exception as e:
        tqdm.write(f"⚠️ Failed to validate {ticker} with yfinance: {e}")
        return False


def save_tickers_to_file(tickers: list[str]):
    with open(TICKERS_FILE, "w") as f:
        for t in sorted(set(tickers)):
            f.write(f"{t}\n")
    print(f"✅ Saved {len(set(tickers))} validated tickers to {TICKERS_FILE}")


def main():
    print("📈 Fetching S&P 500...")
    sp500 = fetch_sp500_tickers()

    print("📈 Fetching Nasdaq-100...")
    nasdaq100 = fetch_nasdaq100_tickers()

    print("📈 Fetching Dow 30...")
    dow30 = fetch_dow30_tickers()

    print("📈 Fetching Russell 1000 (via iShares)...")
    russell1000 = fetch_russell1000_from_ishares()

    print("🔍 Deduplicating and filtering tickers...")
    combined = sp500 + nasdaq100 + dow30 + russell1000
    combined_filtered = [t.upper() for t in combined if is_valid_ticker(t)]

    print(f"🧪 Validating {len(combined_filtered)} tickers with yfinance...")
    valid_tickers = [
        t for t in tqdm(combined_filtered, desc="Validating tickers")
        if validate_ticker_with_yfinance(t)
    ]

    save_tickers_to_file(valid_tickers)


if __name__ == "__main__":
    main()
