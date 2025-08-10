import os
from datetime import date
import polars as pl
import pytest

from src.dataprep.features.aggregation import ticker_batch_runner as tbr


def _mk_dyn_row(ticker: str, as_of: date, feat=1.0):
    return pl.DataFrame({
        "ticker": [ticker],
        "as_of": [as_of],
        "feature_x": [feat],
        # keep denominators to satisfy validator if you want no nullification
        "free_cash_flow": [10.0],
        "ebitda": [10.0],
        "pfcf_ratio": [5.0],
        "net_debt_to_ebitda": [1.0],
    })


def _mk_static(sector="Technology"):
    return pl.DataFrame({"ticker": ["ZZZ"], "sector": [sector]})


def test_save_or_append_new_and_merge(tmp_path, monkeypatch):
    # Route output to tmp
    tbr.OUTPUT_DIR = tmp_path.as_posix()

    # 1) new file
    df1 = _mk_dyn_row("AAA", date(2024, 12, 31))
    changed1 = tbr.save_or_append(df1, "AAA", merge_with_existing=True)
    assert changed1 is True
    assert (tmp_path / "AAA.parquet").exists()

    # 2) append a new as_of → height should change
    df2 = _mk_dyn_row("AAA", date(2023, 12, 31))
    changed2 = tbr.save_or_append(df2, "AAA", merge_with_existing=True)
    assert changed2 is True

    # sanity
    df_read = pl.read_parquet(tmp_path / "AAA.parquet")
    assert set(df_read["as_of"].to_list()) == {date(2023, 12, 31), date(2024, 12, 31)}


def test_generate_features_append_and_flagged(tmp_path, monkeypatch):
    # Redirect paths
    tbr.OUTPUT_DIR = tmp_path.as_posix()
    tbr.AUDIT_DIR = (tmp_path / "_audit").as_posix()
    os.makedirs(tbr.AUDIT_DIR, exist_ok=True)

    # Pretend an existing file with 2021
    existing = _mk_dyn_row("ZZZ", date(2021, 12, 31))
    existing.write_parquet(tmp_path / "ZZZ.parquet")

    # Monkeypatch fetch/build to return per-date rows + static/sector
    def fake_fetch_and_build(ticker, as_of):
        return _mk_dyn_row(ticker, as_of), _mk_static("Technology"), "Technology"

    # First date is skipped (exists), second is flagged and kept
    def fake_validate(df, ticker, prev_df=None, sector=None):
        if df["as_of"].item() == date(2021, 12, 31):
            return "ok", [], df
        return "flagged", ["pfcf_ratio_oob>800:1000.0"], df

    monkeypatch.setattr(tbr, "fetch_and_build_features", fake_fetch_and_build)
    monkeypatch.setattr(tbr, "validate_dynamic_row", fake_validate)

    # Run
    logs, changed, stats = tbr.generate_features_for_ticker(
        "ZZZ", [date(2021, 12, 31), date(2022, 12, 31)]
    )
    assert changed is True
    assert stats.flagged >= 1

    # Parquet contains both years, with validation columns added for flagged row
    df = pl.read_parquet(tmp_path / "ZZZ.parquet").sort("as_of")
    assert df.height == 2
    # 2022 should be flagged
    row2022 = df.filter(pl.col("as_of") == date(2022, 12, 31))
    assert row2022["validation_status"].item() == "flagged"
    assert "pfcf_ratio_oob" in row2022["violations"].item()

    # Sidecar audit exists
    assert any(p.name.startswith("ZZZ_2022-12-31") for p in (tmp_path / "_audit").glob("*.txt"))


def test_merge_all_feature_vectors_force(tmp_path, monkeypatch):
    tbr.OUTPUT_DIR = tmp_path.as_posix()
    # Two minimal ticker files
    for sym in ("AAA", "BBB"):
        _mk_dyn_row(sym, date(2024, 12, 31)).write_parquet(tmp_path / f"{sym}.parquet")
    tbr.merge_all_feature_vectors(force_merge=True)
    assert (tmp_path / "features_all_tickers_timeseries.parquet").exists()
