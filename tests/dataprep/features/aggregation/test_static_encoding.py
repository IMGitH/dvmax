import polars as pl
import os
from pathlib import Path
import importlib

# 1) encoder stability
def test_encode_sector_country_stability():
    from src.dataprep.features.engineering import encode_sector, encode_country
    from src.dataprep.constants import ALL_COUNTRIES, ALL_SECTORS 
    d1 = encode_sector("Information Technology")
    d2 = encode_sector("Energy")
    d3 = encode_sector("UNKNOWN")
    # all vocab-driven columns exist exactly once
    for v in ALL_SECTORS:
        k = "sector_" + v.lower().replace(" ", "_")
        assert k in d1 and k in d2 and k in d3
    assert "sector_other" in d1 and "sector_other" in d2 and "sector_other" in d3
    # country
    c1 = encode_country("USA")
    c2 = encode_country("Israel")
    c3 = encode_country("Nowhere")
    for v in ALL_COUNTRIES:
        k = "country_" + v.lower()
        assert k in c1 and k in c2 and k in c3
    assert c3["country_other"] == 1  # unknown bucket

# 2) save_static_row merges columns & keeps unique tickers
def test_save_static_row_merges(tmp_path, monkeypatch):
    # import module under test
    mod = importlib.import_module("src.dataprep.features.aggregation.ticker_batch_runner")
    # redirect STATIC_DIR
    monkeypatch.setattr(mod, "STATIC_DIR", str(tmp_path))
    Path(mod.STATIC_DIR).mkdir(parents=True, exist_ok=True)

    # two rows with different one-hot columns lit
    df1 = pl.DataFrame([{"ticker": "AAA", "country":"USA", "sector":"Energy",
                         "sector_energy":1, "sector_other":0, "country_usa":1, "country_other":0}])
    df2 = pl.DataFrame([{"ticker": "BBB", "country":"Israel", "sector":"Information Technology",
                         "sector_information_technology":1, "sector_other":0, "country_israel":1, "country_other":0}])

    mod.save_static_row(df1)
    mod.save_static_row(df2)

    out = pl.read_parquet(os.path.join(mod.STATIC_DIR, "static_ticker_info.parquet"))
    assert set(out["ticker"].to_list()) == {"AAA", "BBB"}
    # both OHE cols present after union
    assert "sector_energy" in out.columns and "sector_information_technology" in out.columns

# 3) OHE projection is lean & float32
def test_static_ohe_projection(tmp_path, monkeypatch):
    mod = importlib.import_module("src.dataprep.features.aggregation.ticker_batch_runner")
    monkeypatch.setattr(mod, "STATIC_DIR", str(tmp_path))
    Path(mod.STATIC_DIR).mkdir(parents=True, exist_ok=True)

    # write a minimal static file
    src = os.path.join(mod.STATIC_DIR, "static_ticker_info.parquet")
    df = pl.DataFrame([{"ticker":"AAA","sector_energy":1,"sector_other":0,"country_usa":1,"country_other":0}])
    df.write_parquet(src)

    mod.write_static_ohe_projection()
    dst = os.path.join(mod.STATIC_DIR, "static_ohe.parquet")
    out = pl.read_parquet(dst)
    assert out.columns == ["ticker","sector_energy","sector_other","country_usa","country_other"]
    assert str(out["sector_energy"].dtype).startswith("Float32")
