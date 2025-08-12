import polars as pl
from datetime import date
from src.dataprep.features.aggregation.ticker_row_builder import build_feature_table_from_inputs


def mock_inputs():
    dates = pl.date_range(date(2023, 1, 1), date(2024, 12, 31), interval="1d", eager=True)

    return {
        "prices": pl.DataFrame({"date": dates, "close": pl.Series([100 + i for i in range(len(dates))])}),
        "dividends": pl.DataFrame({"date": dates[::12], "dividend": pl.Series([0.5] * len(dates[::12]))}),
        "ratios": pl.DataFrame({
            "date": dates,
            "dividendYield": [0.015] * len(dates),
            "peRatio": [15.0] * len(dates),
            "pfcfRatio": [18.0] * len(dates),
            "fcfCagr3y": [0.1] * len(dates),
        }),
        "income": pl.DataFrame({
            "date": dates,
            "operatingIncome": [200.0] * len(dates),  # required for ebit_interest_cover
            "interestExpense": [20.0] * len(dates),
        }),
        "balance": pl.DataFrame({
            "date": dates,
            "totalDebt": [500.0] * len(dates),
            "cash": [100.0] * len(dates),
            "ebitda": [250.0] * len(dates),
        }),
        "splits": pl.DataFrame({"date": [], "ratio": []}),
        "profile": {"sector": "Technology", "country": "United States"},
        "sector_index": pl.DataFrame({
            "date": dates,
            "close": pl.Series([3000 + i for i in range(len(dates))])
        }),
    }


def test_build_feature_table_from_inputs():
    ticker = "MOCK"
    as_of = date(2024, 12, 31)
    inputs = mock_inputs()

    dynamic_df, static_df = build_feature_table_from_inputs(ticker, inputs, as_of)

    # Basic structure check
    assert isinstance(dynamic_df, pl.DataFrame)
    assert isinstance(static_df, pl.DataFrame)
    assert dynamic_df.height == 1
    assert static_df.height == 1

    # Required dynamic columns
    expected_dynamic_keys = {
        "ticker", "as_of", "6m_return", "12m_return", "volatility", "max_drawdown_1y",
        "sector_relative_6m", "sma_50_200_delta", "net_debt_to_ebitda", "ebit_interest_cover",
        "ebit_interest_cover_capped", "eps_cagr_3y", "fcf_cagr_3y", "dividend_yield",
        "dividend_cagr_3y", "dividend_cagr_5y", "yield_vs_5y_median", "pe_ratio", "pfcf_ratio",
        "payout_ratio", "has_eps_cagr_3y", "has_fcf_cagr_3y", "has_dividend_yield",
        "has_dividend_cagr_3y", "has_dividend_cagr_5y", "has_ebit_interest_cover"
    }
    assert set(dynamic_df.columns) >= expected_dynamic_keys

    # Required static columns
    assert static_df["ticker"][0] == ticker
    assert static_df["country"][0] == "United States"
    assert any(col.startswith("sector_") for col in static_df.columns)
