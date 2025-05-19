import polars as pl
import pytest
from src.dataprep.features.growth_features import (
    compute_dividend_cagr, 
    compute_eps_cagr, 
    compute_fcf_cagr,
    compute_cagr_generic
)


def test_compute_dividend_cagr_basic():
    df = pl.DataFrame({
        "date": ["2019-01-01", "2020-01-01", "2024-01-01"],
        "dividend": [1.0, 1.1, 2.0]
    })
    splits_df = pl.DataFrame()
    result = compute_dividend_cagr(df, splits_df, years=5)
    expected = (2.0 / 1.0) ** (1 / 5) - 1
    print("\n=== test_compute_dividend_cagr_basic ===")
    print(f"Start dividend: 1.0")
    print(f"End dividend: 2.0")
    print(f"Years: 5")
    print(f"Expected CAGR: {expected:.6f}")
    print(f"Computed CAGR: {result:.6f}")
    assert result == pytest.approx(expected, rel=1e-4)


def test_compute_cagr_basic():
    df = pl.DataFrame({
        "date": ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
        "eps": [2.0, 2.5, 3.0, 4.0]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    cagr = compute_cagr_generic(df, "eps", 3)
    expected = (4.0 / 2.0) ** (1 / 3) - 1
    assert round(cagr, 4) == round(expected, 4)


def test_compute_eps_cagr_no_split():
    df = pl.DataFrame({
        "date": ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
        "eps": [2.0, 2.5, 3.0, 4.0]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    # No adjustment → EPS CAGR = (4.0 / 2.0) ** (1/3) - 1
    expected = (4.0 / 2.0) ** (1 / 3) - 1
    cagr = compute_eps_cagr(df, 3)
    assert round(cagr, 4) == round(expected, 4)


def test_compute_fcf_cagr_no_split():
    df = pl.DataFrame({
        "date": ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
        "freeCashFlowPerShare": [1.0, 1.2, 1.5, 2.0]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    # No adjustment → FCF CAGR = (2.0 / 1.0) ** (1/3) - 1
    expected = (2.0 / 1.0) ** (1 / 3) - 1
    cagr = compute_fcf_cagr(df, 3)
    assert round(cagr, 4) == round(expected, 4)
