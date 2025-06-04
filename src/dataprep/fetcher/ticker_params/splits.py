from typing import Literal
import polars as pl
import yfinance as yf

def fetch_splits(ticker: str, mode: Literal["yfinance", "fmp"] = "yfinance") -> pl.DataFrame:
    if mode == "yfinance":
        splits = yf.Ticker(ticker).splits
        if splits.empty:
            return pl.DataFrame()
        return pl.DataFrame({
            "date": splits.index.to_list(),
            "split_ratio": splits.values.tolist()
        }).with_columns(pl.col("date").cast(pl.Date))

    raise NotImplementedError("FMP does not provide split data on free tier. Use yfinance instead.")
