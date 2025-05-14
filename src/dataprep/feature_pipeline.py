import polars as pl
from src.dataprep.features.fundamental_features import (
    compute_net_debt_to_ebitda,
    compute_ebit_interest_cover,
)

def build_fundamental_features(
    balance_df: pl.DataFrame, income_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Build derived features from balance sheet and income statement data.
    
    Includes:
    - Net Debt / EBITDA proxy
    - EBIT Interest Coverage proxy

    Assumes both DataFrames contain a 'date' column in aligned date format.
    """
    df = income_df.join(balance_df, on="date", how="inner")
    df = compute_net_debt_to_ebitda(df)
    df = compute_ebit_interest_cover(df)
    return df.select(["date", "net_debt_to_ebitda", "ebit_interest_cover"])
