import polars as pl

def extract_latest_pe_pfcf(df: pl.DataFrame) -> tuple[float, float]:
    df = df.sort("date")
    if df.is_empty() or "priceEarningsRatio" not in df.columns or "priceToFreeCashFlowsRatio" not in df.columns:
        return 0.0, 0.0

    pe = df.select("priceEarningsRatio").tail(1).item()
    pfcf = df.select("priceToFreeCashFlowsRatio").tail(1).item()
    return pe, pfcf
