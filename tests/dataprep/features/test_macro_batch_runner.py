import os
import polars as pl
from datetime import date
from src.dataprep.constants import MACRO_INDICATORS
from src.dataprep.fetcher.macro import WorldBankAPI
from src.dataprep.features.aggregation.macro_batch_runner import engineer_macro_features

def test_engineer_macro_features_saves_parquet(tmp_path):
    country = "United States"
    start = 2015
    end = 2023
    output_dir = tmp_path / "macro"

    api = WorldBankAPI()
    df_raw = api.fetch_macro_indicators(MACRO_INDICATORS, country, start, end)
    df = pl.from_pandas(df_raw.reset_index()).with_columns(pl.col("date").cast(pl.Date))

    path = engineer_macro_features(df, as_of=date(end, 12, 31), country=country, output_dir=str(output_dir))
    assert os.path.exists(path)

    df_out = pl.read_parquet(path)
    expected_cols = {
        "as_of", "gdp_yoy", "gdp_pc_yoy", "inflation_latest", "inflation_yoy",
        "inflation_vol_3y", "unemployment_latest", "consumption_latest", "exports_latest"
    }
    assert set(df_out.columns) == expected_cols
    assert df_out.shape[0] == 1
