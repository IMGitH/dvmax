import polars as pl
from unittest.mock import patch
from datetime import date
from tests.dataprep.features.aggregation.test_row_builder import get_random_prices
from src.dataprep.features.aggregation.row_builder import build_feature_table_from_inputs
from src.dataprep.visualization.report import print_feature_report_from_df
from src.dataprep.features.aggregation.row_builder import add_has_flags
import numpy as np


@patch("src.dataprep.fetcher.fetch_all.fetch_prices")
@patch("src.dataprep.fetcher.fetch_all.fetch_dividends")
@patch("src.dataprep.fetcher.fetch_all.fetch_ratios")
@patch("src.dataprep.fetcher.fetch_all.fetch_balance_sheet_fund")
@patch("src.dataprep.fetcher.fetch_all.fetch_income_statement_fund")
@patch("src.dataprep.fetcher.fetch_all.fetch_company_profile")
@patch("src.dataprep.fetcher.fetch_all.fetch_splits")
def test_print_report_with_mocked_data(
    mock_splits, mock_profile, mock_income, mock_balance,
    mock_ratios, mock_dividends, mock_prices
):
    mock_prices.return_value = get_random_prices()

    mock_dividends.return_value = pl.DataFrame({
        "date": ["2019-01-01", "2020-01-01", "2024-01-01"],
        "dividend": [0.6, 1, 2]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    mock_ratios.return_value = pl.DataFrame({
        "date": ["2023-12-31"],
        "priceEarningsRatio": [15.0],
        "priceToFreeCashFlowsRatio": [20.0]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    mock_balance.return_value = pl.DataFrame({
        "date": ["2023-12-31"],
        "fcf": [150],
        "totalDebt": [100],
        "cashAndShortTermInvestments": [50]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    mock_income.return_value = pl.DataFrame({
        "date": ["2023-12-31"],
        "operatingIncome": [210],
        "incomeBeforeTax": [200],
        "interestExpense": [10],
        "depreciationAndAmortization": [20],
        "eps": [5]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    mock_splits.return_value = pl.DataFrame()

    mock_profile.return_value = {
        "date": ["2023-12-31"],
        "sector": "Technology",
        "country": "United States",
    }

    from src.dataprep.fetcher.fetch_all import fetch_all
    inputs = fetch_all("MOCK", div_lookback_years=5, other_lookback_years=5)
    df = build_feature_table_from_inputs("MOCK", inputs, as_of=date.today())

    # Validate presence and correctness of binary flags
    binary_flags = [
        "has_eps_cagr_3y", "has_fcf_cagr_3y",
        "has_dividend_yield", "has_dividend_cagr_3y", "has_dividend_cagr_5y",
        "has_ebit_interest_cover"
    ]

    for col in binary_flags:
        assert col in df.columns, f"Missing binary flag column: {col}"
        val = df[0, col]
        assert val in (0, 1), f"{col} should be 0 or 1, got {val}"

    print_feature_report_from_df(df, inputs, date.today())  # optional visual

def test_add_has_flags():
    feature_row = {
        "eps_cagr_3y": 0.12,
        "fcf_cagr_3y": np.nan,
        "dividend_yield": 0.03,
        "dividend_cagr_3y": 0.10,
        "dividend_cagr_5y": 0.16,
        "ebit_interest_cover": np.nan
    }

    nullable_keys = [
        "eps_cagr_3y", "fcf_cagr_3y",
        "dividend_yield", "dividend_cagr_3y", "dividend_cagr_5y",
        "ebit_interest_cover"
    ]

    updated = add_has_flags(feature_row.copy(), nullable_keys)

    assert updated["has_eps_cagr_3y"] == 1
    assert updated["has_fcf_cagr_3y"] == 0
    assert updated["has_dividend_yield"] == 1
    assert updated["has_dividend_cagr_3y"] == 1
    assert updated["has_dividend_cagr_5y"] == 1
    assert updated["has_ebit_interest_cover"] == 0
