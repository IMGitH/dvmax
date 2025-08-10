import os
from datetime import date
import pandas as pd
import polars as pl
import numpy as np
import pytest

from src.dataprep.features.aggregation import macro_batch_runner as mbr


def _mk_macro_df(years):
    # Build a tidy “WorldBank-like” DataFrame compatible with engineer_macro_features
    df = pd.DataFrame({
        "date": pd.to_datetime([f"{y}-12-31" for y in years]),
        "GDP (USD)": [1_000, 1_200, 1_350, 1_500][:len(years)],
        "GDP per Capita (const USD)": [50_000, 51_000, 52_500, 53_000][:len(years)],
        "Inflation (%)": [2.0, 2.5, 3.0, 3.5][:len(years)],
        "Unemployment (%)": [4.0, 3.9, 4.1, 4.2][:len(years)],
        "Private Consumption (% GDP)": [65.0, 66.0, 66.5, 67.0][:len(years)],
        "Exports (% GDP)": [12.0, 12.5, 12.0, 11.8][:len(years)],
    })
    pl_df = pl.from_pandas(df)
    return pl_df.with_columns(pl.col("date").cast(pl.Date))


def test_engineer_macro_features_happy_path(tmp_path, monkeypatch):
    outroot = tmp_path.as_posix()
    # years include ref-2, ref-1, selected (for inflation/unemp)
    years = [2021, 2022, 2023, 2024]
    df = _mk_macro_df(years)

    path = mbr.engineer_macro_features(
        df=df,
        as_of=date(2024, 12, 31),
        country="United States",
        output_root=outroot
    )
    assert os.path.exists(path)

    out = pl.read_parquet(path)
    # as_of_year == 2024, backfilled_year == 2023
    last = out.sort("as_of_year").tail(1)
    assert last["as_of_year"].item() == 2024
    assert last["backfilled_year"].item() == 2023

    # No NaNs permitted by function contract
    for c, dt in last.schema.items():
        if dt in (pl.Float32, pl.Float64):
            assert not np.isnan(last[c].item())


def test_engineer_macro_features_uses_prev_year_when_current_year(tmp_path):
    outroot = tmp_path.as_posix()
    years = [2022, 2023, 2024]  # current-year request (e.g., 2025) should downshift to 2024
    df = _mk_macro_df(years)

    # as_of uses "today()".year; simulate by passing current year explicitly
    as_of = date(date.today().year, 12, 31)
    path = mbr.engineer_macro_features(
        df=df,
        as_of=as_of,
        country="United States",
        output_root=outroot
    )
    out = pl.read_parquet(path).sort("as_of_year").tail(1)
    # Ensures the function wrote the *previous* year
    assert out["as_of_year"].item() == as_of.year - 1


def test_fetch_and_save_macro_with_mocked_worldbank(tmp_path, monkeypatch):
    # Mock WorldBankAPI.fetch_macro_indicators to avoid network
    class DummyWB:
        def fetch_macro_indicators(self, indicator_map, country_name, start, end):
            years = list(range(start, end + 1))
            # minimal but sufficient columns for engineer_macro_features
            df = pd.DataFrame({
                "date": pd.to_datetime([f"{y}-12-31" for y in years]),
                "GDP (USD)": [1000 + 10*y for y in years],
                "GDP per Capita (const USD)": [50000 + y for y in years],
                "Inflation (%)": [2.0 for _ in years],
                "Unemployment (%)": [4.0 for _ in years],
                "Private Consumption (% GDP)": [65.0 for _ in years],
                "Exports (% GDP)": [12.0 for _ in years],
            }).set_index("date")
            return df

    monkeypatch.setattr(mbr, "WorldBankAPI", lambda: DummyWB())

    out = mbr.fetch_and_save_macro(
        country="United States",
        start_year=2020,
        end_year=2023,
        output_root=tmp_path.as_posix()
    )
    assert out is not None
    assert os.path.exists(out)
    df = pl.read_parquet(out)
    # Started loop from start_year+2 => expect rows for 2022 and 2023
    assert set(df["as_of_year"].to_list()) >= {2022, 2023}
