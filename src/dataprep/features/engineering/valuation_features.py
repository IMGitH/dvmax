import polars as pl
import logging
import numpy as np
from typing import Tuple

def extract_latest_pe_pfcf(df: pl.DataFrame) -> Tuple[float, float]:
    if df.is_empty():
        logging.warning("[P/E] No ratio data available at all.")
        return np.nan, np.nan

    required = {"priceEarningsRatio", "priceToFreeCashFlowsRatio", "date"}
    missing = required - set(df.columns)
    if missing:
        logging.warning(f"[P/E] Missing columns in ratio data: {missing}")
        return np.nan, np.nan

    # Latest first
    df = df.sort("date", descending=True)

    def latest_positive(col: str) -> float:
        s = df.filter(pl.col(col) > 0).select(pl.col(col)).to_series()
        return float(s[0]) if len(s) else np.nan

    pe   = latest_positive("priceEarningsRatio")
    pfcf = latest_positive("priceToFreeCashFlowsRatio")

    if np.isnan(pe) or np.isnan(pfcf):
        latest_pe   = df[0, "priceEarningsRatio"]
        latest_pfcf = df[0, "priceToFreeCashFlowsRatio"]
        logging.warning(
            f"[P/E] Missing positive values. Latest observed were: P/E={latest_pe}, P/FCF={latest_pfcf}."
        )

    return pe, pfcf
