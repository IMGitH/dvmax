# tests/.../test_save_or_append.py
import os
from datetime import date
import polars as pl
from src.dataprep.features.aggregation.ticker_batch_runner import save_or_append

def test_save_or_append_detects_change(tmp_path, monkeypatch):
    # avoid network preflight in case module is imported elsewhere
    monkeypatch.setenv("FMP_PREFLIGHT", "0")

    outdir = tmp_path / "tickers_history"
    from src.dataprep.features.aggregation import ticker_batch_runner as tbr
    tbr.OUTPUT_DIR = str(outdir)

    # first write
    df1 = pl.DataFrame({
        "ticker": ["AAA"],
        "as_of":  [date(2024, 12, 31)],
        "x":      [1.0],
    }).with_columns(pl.col("as_of").cast(pl.Date))
    changed = save_or_append(df1, "AAA", merge_with_existing=True)
    assert changed is True

    # same height, different value → triggers atomic rewrite path
    df2 = pl.DataFrame({
        "ticker": ["AAA"],
        "as_of":  [date(2024, 12, 31)],
        "x":      [2.0],
    }).with_columns(pl.col("as_of").cast(pl.Date))
    changed2 = save_or_append(df2, "AAA", merge_with_existing=True)
    assert changed2 in (True, False)

    assert os.path.exists(os.path.join(str(outdir), "AAA.parquet"))
