import polars as pl
import json
from datetime import date
from src.dataprep.features.aggregation import ticker_batch_runner as tbr

def _row(t, d):
    return (
        pl.DataFrame(
            {
                "ticker": [t],
                "as_of":  [d],  # <-- python datetime.date
                "x":      [1.0],
            }
        )
        .with_columns(pl.col("as_of").cast(pl.Date))  # ensure pl.Date dtype
    )

def test_preflight_skip(monkeypatch, capsys):
    monkeypatch.setenv("FMP_PREFLIGHT", "0")
    tbr._maybe_preflight_fmp()  # should not raise
    assert "Skipping FMP preflight" in capsys.readouterr().out

def test_overwrite_modes(tmp_path, monkeypatch):
    tbr.OUTPUT_DIR = tmp_path.as_posix()
    # seed an existing file
    _row("AAA", date(2022,12,31)).write_parquet(tmp_path/"AAA.parquet")

    # skip mode
    monkeypatch.setenv("OVERWRITE_MODE", "skip")
    tbr.OVERWRITE_MODE = "skip"
    msg, changed, stats = tbr.generate_features_for_ticker("AAA", [date(2022,12,31)])
    if isinstance(msg, list):
        msg = " ".join(msg)
    assert msg.startswith("[SKIP") and "AAA" in msg

def test_force_merge_and_status(tmp_path, monkeypatch):
    tbr.OUTPUT_DIR = tmp_path.as_posix()
    tbr.STATUS_DIR = (tmp_path/"status").as_posix()
    # two tickers
    _row("AAA", date(2024,12,31)).write_parquet(tmp_path/"AAA.parquet")
    _row("BBB", date(2024,12,31)).write_parquet(tmp_path/"BBB.parquet")

    # force merge
    tbr.merge_all_feature_vectors(force_merge=True)
    assert (tmp_path/"features_all_tickers_timeseries.parquet").exists()

    # status
    tbr._write_status_files(tbr.RunStats(ok=2))
    proc = json.loads((tmp_path/"status"/"processed.json").read_text())
    last = json.loads((tmp_path/"status"/"last_run.json").read_text())
    assert "AAA" in proc and "BBB" in proc
    assert last["ok"] == 2
