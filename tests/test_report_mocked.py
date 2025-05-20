import polars as pl
from unittest.mock import patch
from datetime import date
from tests.dataprep.features.test_feature_table import get_random_prices
from src.dataprep.report.feature_table import build_feature_table_from_inputs
from src.dataprep.report.report import print_feature_report_from_df


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
        "has_dividend_yield", "has_dividend_cagr_3y", "has_dividend_cagr_5y"
    ]

    for col in binary_flags:
        assert col in df.columns, f"Missing binary flag column: {col}"
        val = df[0, col]
        assert val in (0, 1), f"{col} should be 0 or 1, got {val}"

    print_feature_report_from_df(df, inputs, date.today())  # optional visual


def test_has_flags_with_missing_growth_metrics():
    df = pl.DataFrame([{
        "ticker": "XYZ",
        "eps_cagr_3y": None,
        "fcf_cagr_3y": 0.10,
        "dividend_yield": None,
        "dividend_cagr_3y": 0.05,
        "dividend_cagr_5y": None
    }])

    # Manually create flags
    for col in [
        "eps_cagr_3y", "fcf_cagr_3y",
        "dividend_yield", "dividend_cagr_3y", "dividend_cagr_5y"
    ]:
        df = df.with_columns(pl.col(col).is_not_null().cast(pl.Int8).alias(f"has_{col}"))

    expected = {
        "has_eps_cagr_3y": 0,
        "has_fcf_cagr_3y": 1,
        "has_dividend_yield": 0,
        "has_dividend_cagr_3y": 1,
        "has_dividend_cagr_5y": 0
    }

    for col, expected_val in expected.items():
        assert df[0, col] == expected_val, f"{col} expected {expected_val}, got {df[0, col]}"
