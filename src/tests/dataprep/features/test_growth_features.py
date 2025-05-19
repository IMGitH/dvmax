import polars as pl
from src.dataprep.features.growth_features import (
    compute_cagr, 
    compute_eps_cagr, 
    compute_fcf_cagr
)


def test_compute_cagr_basic():
    df = pl.DataFrame({
        "date": ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
        "eps": [2.0, 2.5, 3.0, 4.0]
    }).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    cagr = compute_cagr(df, "eps", 3)
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
