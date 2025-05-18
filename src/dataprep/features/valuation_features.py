import polars as pl
import logging

def extract_latest_pe_pfcf(df: pl.DataFrame) -> tuple[float, float]:
    df = df.sort("date", descending=True)

    if df.is_empty():
        logging.warning("[P/E] No ratio data available at all.")
        return 0.0, 0.0

    required_cols = {"priceEarningsRatio", "priceToFreeCashFlowsRatio"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logging.warning(f"[P/E] Missing columns in ratio data: {missing}")
        return 0.0, 0.0

    valid = df.filter(
        (pl.col("priceEarningsRatio") > 0) &
        (pl.col("priceToFreeCashFlowsRatio") > 0)
    )

    if valid.is_empty():
        pe = df[-1, "priceEarningsRatio"] if "priceEarningsRatio" in df.columns else "N/A"
        pfcf = df[-1, "priceToFreeCashFlowsRatio"] if "priceToFreeCashFlowsRatio" in df.columns else "N/A"
        logging.warning(
            f"[P/E] No valid non-zero ratio found. Latest values were: P/E={pe}, P/FCF={pfcf}."
        )
        return 0.0, 0.0

    return valid[0, "priceEarningsRatio"], valid[0, "priceToFreeCashFlowsRatio"]
