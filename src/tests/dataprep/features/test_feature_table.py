import pytest
import polars as pl
from datetime import date
from unittest.mock import patch
from src.dataprep.features import build_feature_table

@patch("src.dataprep.fetcher.fetch_prices")
@patch("src.dataprep.fetcher.fetch_dividends")
@patch("src.dataprep.fetcher.fetch_ratios")
@patch("src.dataprep.fetcher.fetch_balance_sheet_fund")
@patch("src.dataprep.fetcher.fetch_income_statement_fund")
@patch("src.dataprep.fetcher.fetch_company_profile")
def test_build_feature_table_basic(
    mock_profile, mock_income, mock_balance,
    mock_ratios, mock_dividends, mock_prices
):
    # --- Mock prices ---
    prices_df = pl.DataFrame({
        "date": ["2023-01-01", "2023-06-01", "2023-12-31"],
        "close": [100, 150, 120]
    })
    mock_prices.return_value = prices_df

    # --- Mock dividends ---
    dividends_df = pl.DataFrame({
        "date": [f"{2018+i}-06-01" for i in range(6)],
        "dividend": [1.0, 1.1, 1.21, 1.33, 1.46, 1.6]
    })
    mock_dividends.return_value = dividends_df

    # --- Mock ratios ---
    ratios_df = pl.DataFrame({
        "date": ["2023-01-01", "2023-12-31"],
        "priceEarningsRatio": [22.0, 24.0],
        "priceToFreeCashFlowsRatio": [18.0, 19.5],
        "dividendYield": [0.015, 0.018]
    })
    mock_ratios.return_value = ratios_df

    # --- Mock income statement ---
    income_df = pl.DataFrame({
        "date": ["2023-12-31"],
        "incomeBeforeTax": [100],
        "interestExpense": [10]
    })
    mock_income.return_value = income_df

    # --- Mock balance sheet ---
    balance_df = pl.DataFrame({
        "date": ["2023-12-31"],
        "totalDebt": [300],
        "cashAndShortTermInvestments": [0]
    })
    mock_balance.return_value = balance_df

    # --- Mock company profile ---
    mock_profile.return_value = {"sector": "Technology"}

    print("\n=== Running build_feature_table ===")
    df = build_feature_table("AAPL", as_of=date(2023, 12, 31),
                             div_lookback_years=5,
                             other_lookback_years=3)
    print_features(df)

    # --- Assertions ---
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1
    assert df[0, "ticker"] == "AAPL"
    assert "net_debt_to_ebitda" in df.columns
    assert "6m_return" in df.columns
    assert "sector_technology" in df.columns


def print_features(df: pl.DataFrame):
    print("\n=== Feature Table Summary ===")
    print(f"- Shape: {df.shape}")
    
    print("\n=== Columns ===")
    for col in df.columns:
        print(f"• {col}")
    
    print("\n=== Full Row Data ===")
    row = df[0].to_dict(as_series=False)
    for key, value in row.items():
        print(f"{key:>20}: {value[0]}")

    print("\n=== Sector One-Hot Encoding ===")
    sector_cols = [col for col in df.columns if col.startswith("sector_")]
    sector_row = df.select(sector_cols).to_dicts()[0]
    for col, val in sector_row.items():
        print(f"{col:>20}: {val}")