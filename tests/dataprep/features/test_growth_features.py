import polars as pl
import pytest
import numpy as np
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
    assert result is not None
    assert result == pytest.approx(expected, rel=1e-4)


def test_compute_cagr_basic():
    df = pl.DataFrame({
        "date": ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
        "eps": [2.0, 2.5, 3.0, 4.0]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    cagr = compute_cagr_generic(df, "eps", 3)
    expected = (4.0 / 2.0) ** (1 / 3) - 1
    assert cagr is not None
    assert round(cagr, 4) == round(expected, 4)


def test_compute_eps_cagr_no_split():
    df = pl.DataFrame({
        "date": ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
        "eps": [2.0, 2.5, 3.0, 4.0]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    expected = (4.0 / 2.0) ** (1 / 3) - 1
    cagr = compute_eps_cagr(df, 3)
    assert cagr is not None
    assert round(cagr, 4) == round(expected, 4)


def test_compute_fcf_cagr_no_split():
    df = pl.DataFrame({
        "date": ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
        "freeCashFlowPerShare": [1.0, 1.2, 1.5, 2.0]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    expected = (2.0 / 1.0) ** (1 / 3) - 1
    cagr = compute_fcf_cagr(df, 3)
    assert not np.isnan(cagr)
    assert round(cagr, 4) == round(expected, 4)

def test_compute_cagr_returns_none_when_insufficient_data():
    # Only one data point — cannot compute CAGR
    df = pl.DataFrame({
        "date": ["2024-01-01"],
        "eps": [2.0]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    result = compute_eps_cagr(df, 3)
    assert np.isnan(result)


def test_compute_cagr_returns_none_when_grace_window_misses():
    # The earliest point is too far from the target start date
    df = pl.DataFrame({
        "date": ["2022-01-01", "2023-01-01", "2024-01-01"],
        "eps": [2.0, 2.5, 3.0]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    # Looking back 5 years = 2019, but earliest point is 2022
    result = compute_eps_cagr(df, 5)
    assert np.isnan(result)
