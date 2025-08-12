import pytest
import polars as pl
from datetime import date

from src.dataprep.features.aggregation.validate_dynamic_row import validate_dynamic_row

# === Sample valid and invalid data for testing ===

def make_valid_df():
    return pl.DataFrame({
        "as_of": [date(2025, 1, 1)],
        "6m_return": [0.1],
        "12m_return": [0.2],
        "volatility": [0.25],
        "max_drawdown_1y": [0.3],
        "sector_relative_6m": [0.1],
        "sma_50_200_delta": [0.05],
        "net_debt_to_ebitda": [1.0],
        "ebit_interest_cover": [20.0],
        "ebit_interest_cover_capped": [20.0],
        "eps_cagr_3y": [0.1],
        "fcf_cagr_3y": [0.1],
        "dividend_yield": [0.03],
        "dividend_cagr_3y": [0.05],
        "dividend_cagr_5y": [0.07],
        "yield_vs_5y_median": [0.01],
        "pe_ratio": [20.0],
        "pfcf_ratio": [25.0],
        "payout_ratio": [0.4],
    })

def make_invalid_range_df():
    df = make_valid_df()
    return df.with_columns([
        pl.lit(999.0).alias("dividend_yield")  # Out of expected range
    ])

def make_invalid_trend_df():
    current = make_valid_df()
    previous = current.with_columns([
        pl.lit(date(2024, 1, 1)).alias("as_of"),
        pl.lit(0.01).alias("dividend_yield")  # Large jump from 0.01 → 0.03 is OK
    ])
    current = current.with_columns([
        pl.lit(0.3).alias("dividend_yield")  # 0.01 → 0.3 triggers >×25 ratio
    ])
    return current, previous

# === Tests ===

def test_valid_passes():
    df = make_valid_df()
    validate_dynamic_row(df, ticker="MOCK")  # should not raise

def test_range_violation_flags_instead_of_raises():
    df = make_invalid_range_df()
    status, violations, out = validate_dynamic_row(df, "XYZ", prev_df=None, sector="Technology")
    assert status == "flagged"
    assert any("dividend_yield out-of-bounds" in v for v in violations)

def test_trend_violation_flags_instead_of_raises():
    current = pl.DataFrame({
        "as_of": [date(2024, 12, 31)],
        "dividend_yield": [0.2],  # within absolute range but big jump
    })
    previous = pl.DataFrame({
        "as_of": [date(2023, 12, 31)],
        "dividend_yield": [0.01],  # drastic increase
    })
    status, violations, _ = validate_dynamic_row(current, "XYZ", prev_df=previous, sector="Technology")
    assert status == "flagged"
    assert any("dividend_yield abnormal change" in v for v in violations)
