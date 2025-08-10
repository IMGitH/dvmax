import polars as pl
from src.dataprep.features.aggregation import ticker_batch_runner as tbr
from datetime import date

def test_merge_force(tmp_path, monkeypatch):
    tbr.OUTPUT_DIR = str(tmp_path)
    for sym in ("AAA", "BBB"):
        df = pl.DataFrame({
            "ticker": [sym],
            "as_of":  [date(2024, 12, 31)],   # ← use Python date
            "x":      [1.0],
        }).with_columns(pl.col("as_of").cast(pl.Date))  # ensure pl.Date dtype
        df.write_parquet(tmp_path / f"{sym}.parquet")

    tbr.merge_all_feature_vectors(force_merge=True)
    assert (tmp_path / "features_all_tickers_timeseries.parquet").exists()
