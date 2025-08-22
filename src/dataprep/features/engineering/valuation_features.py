import polars as pl
import logging
from typing import Optional, Tuple

def extract_latest_pe_pfcf(df: pl.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if df.is_empty():
        logging.warning("[P/E] No ratio data available at all.")
        return None, None

    required = {"priceEarningsRatio", "priceToFreeCashFlowsRatio", "date"}
    missing = required - set(df.columns)
    if missing:
        logging.warning(f"[P/E] Missing columns in ratio data: {missing}")
        return None, None

    # Latest first
    df = df.sort("date", descending=True)

    def latest_positive(col: str) -> Optional[float]:
        s = df.filter(pl.col(col) > 0).select(pl.col(col)).to_series()
        return float(s[0]) if len(s) else None

    pe   = latest_positive("priceEarningsRatio")
    pfcf = latest_positive("priceToFreeCashFlowsRatio")

    if pe is None or pfcf is None:
        # From the actual latest row (after descending sort)
        latest_pe   = df[0, "priceEarningsRatio"]
        latest_pfcf = df[0, "priceToFreeCashFlowsRatio"]
        logging.warning(
            f"[P/E] Missing positive values. Latest observed were: P/E={latest_pe}, P/FCF={latest_pfcf}."
        )

    return pe, pfcf
