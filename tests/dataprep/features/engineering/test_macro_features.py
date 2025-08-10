# tests/macro/test_engineer_features.py
from datetime import date
import polars as pl
from src.dataprep.features.aggregation.macro_batch_runner import engineer_macro_features

def test_engineer_macro_success(tmp_path, caplog):
    df = pl.DataFrame({
        "date": [pl.date(2023,12,31), pl.date(2024,12,31)],
        "GDP (USD)": [10.0, 11.0],
        "GDP per Capita (const USD)": [5.0, 5.5],
        "Inflation (%)": [2.0, 3.0],
        "Unemployment (%)": [4.0, 4.5],
        "Private Consumption (% GDP)": [60.0, 61.0],
        "Exports (% GDP)": [12.0, 13.0],
    })
    out = engineer_macro_features(df, date(date.today().year,12,31), "United States", tmp_path.as_posix())
    assert out.endswith("united_states.parquet")
