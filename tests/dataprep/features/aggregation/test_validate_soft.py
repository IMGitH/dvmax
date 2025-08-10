# tests/.../test_validate_soft.py
from datetime import date
import polars as pl
from src.dataprep.features.aggregation.validate_dynamic_row import validate_dynamic_row

def test_soft_flags_and_rel_jump():
    df = pl.DataFrame({
        "ticker": ["XYZ"],
        "as_of":  [date(2024, 12, 31)],
        "free_cash_flow": [0.2],    # tiny → nullify pfcf
        "pfcf_ratio": [1000.0],
        "ebitda": [0.4],            # tiny → nullify nde
        "net_debt_to_ebitda": [50.0],
    }).with_columns(pl.col("as_of").cast(pl.Date))

    prev = pl.DataFrame({
        "as_of": [date(2023, 12, 31)],
        "pfcf_ratio": [5.0],
        "net_debt_to_ebitda": [1.0],
    }).with_columns(pl.col("as_of").cast(pl.Date))

    status, violations, out = validate_dynamic_row(df, "XYZ", prev_df=prev, sector="Technology")
    assert status == "flagged"
    assert any("pfcf_ratio_nullified_tiny_fcf" in v for v in violations)
    assert any("nde_nullified_tiny_ebitda" in v for v in violations)
    assert out["pfcf_ratio"].to_list()[0] is None
    assert out["net_debt_to_ebitda"].to_list()[0] is None
