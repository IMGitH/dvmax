import polars as pl
import warnings

def extract_latest_pe_pfcf(df: pl.DataFrame) -> tuple[float, float]:
    # TODO: WHY 0 FOR AAPL?
    df = df.sort("date", descending=True)

    if df.is_empty():
        warnings.warn("[P/E] No ratio data available at all.")
        return 0.0, 0.0

    if "priceEarningsRatio" not in df.columns or "priceToFreeCashFlowsRatio" not in df.columns:
        warnings.warn("[P/E] Required columns are missing from ratio data.")
        return 0.0, 0.0

    valid = df.filter(
        (pl.col("priceEarningsRatio") > 0) &
        (pl.col("priceToFreeCashFlowsRatio") > 0)
    )

    if valid.is_empty():
        latest = df[-1]
        pe = latest.get("priceEarningsRatio", "N/A")
        pfcf = latest.get("priceToFreeCashFlowsRatio", "N/A")
        warnings.warn(
            f"[P/E] No valid non-zero ratio found. Latest values were: P/E={pe}, P/FCF={pfcf}."
        )
        return 0.0, 0.0

    return df.select([
        "date",
        "peRatio",                       # <-- real field
        "priceToFreeCashFlowRatio",     # <-- real field
        "dividendYield",
        ...
    ]).rename({
        "peRatio": "priceEarningsRatio",
        "priceToFreeCashFlowRatio": "priceToFreeCashFlowsRatio"
    })