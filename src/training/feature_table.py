import polars as pl
import os
from src.dataprep.fetcher import StockFetcher
from src.dataprep.features import FeatureEngineer

def build_feature_table(ticker: str, output_dir: str = "./output") -> pl.DataFrame:
    fetcher = StockFetcher()
    engineer = FeatureEngineer()

    # Fetch base data
    price_df = fetcher.fetch_prices(ticker)
    ratio_df = fetcher.fetch_ratios(ticker)

    if price_df.height == 0:
        raise ValueError("Price data is empty")

    # Compute engineered features
    ret_6m = engineer.compute_6m_return(price_df)
    vol = engineer.compute_volatility(price_df)

    latest_date = price_df.select("date").max()[0, 0]
    feature_row = pl.DataFrame({
        "date": [latest_date],
        "return_6m": [ret_6m],
        "volatility": [vol]
    })

    # Join on latest date
    full_df = ratio_df.join(feature_row, on="date", how="left")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{ticker.lower()}_features.parquet")
    full_df.write_parquet(path)
    print(f"[SUCCESS] Feature table saved to: {path}")

    return full_df


if __name__ == "__main__":
    build_feature_table("AAPL")
