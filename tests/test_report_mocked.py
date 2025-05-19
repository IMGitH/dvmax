import polars as pl
from unittest.mock import patch
from datetime import date

from src.dataprep.report.feature_table import build_feature_table_from_inputs
from src.dataprep.report.report import print_feature_report_from_df
from src.dataprep.features import (
    compute_6m_return,
    compute_12m_return,
    compute_volatility,
    compute_max_drawdown,
    compute_net_debt_to_ebitda,
    compute_ebit_interest_cover,
    compute_dividend_cagr,
    compute_yield_vs_median,
    compute_eps_cagr,
    compute_fcf_cagr,
    encode_sector,
    compute_sector_relative_return, 
    valuation_features
)


@patch("src.dataprep.fetcher.fetch_all.fetch_prices")
@patch("src.dataprep.fetcher.fetch_all.fetch_dividends")
@patch("src.dataprep.fetcher.fetch_all.fetch_ratios")
@patch("src.dataprep.fetcher.fetch_all.fetch_balance_sheet_fund")
@patch("src.dataprep.fetcher.fetch_all.fetch_income_statement_fund")
@patch("src.dataprep.fetcher.fetch_all.fetch_company_profile")
@patch("src.dataprep.fetcher.fetch_all.fetch_splits")
def test_print_report_with_mocked_data(mock_splits, mock_profile, mock_income, mock_balance, mock_ratios, mock_dividends, mock_prices):
    mock_prices.return_value = pl.DataFrame({
    "date": ["2024-01-01", "2024-06-01"],
    "close": [100, 120]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

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

    from src.dataprep.fetcher.fetch_all import fetch_all
    inputs = fetch_all("MOCK", div_lookback_years=5, other_lookback_years=4)
    df = build_feature_table_from_inputs("MOCK", inputs, as_of=date.today())

    print_feature_report_from_df(df, inputs, date.today())  # visually validate for now
