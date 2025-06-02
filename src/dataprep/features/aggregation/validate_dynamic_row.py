import polars as pl
from typing import Optional
import math

# === Reasonable hard limits for each feature (define only if numeric) ===
FEATURE_RANGES = {
    "6m_return": (-1.0, 2.0),
    "12m_return": (-1.0, 3.0),
    "volatility": (0.0, 1.0),
    "max_drawdown_1y": (0.0, 1.0),
    "sector_relative_6m": (-1.5, 1.5),
    "sma_50_200_delta": (-1.0, 1.0),
    "net_debt_to_ebitda": (-10.0, 30.0),
    "ebit_interest_cover": (0.0, 1000.0),
    "eps_cagr_3y": (-1.0, 3.0),
    "fcf_cagr_3y": (-1.0, 3.0),
    "dividend_yield": (0.0, 0.25),
    "dividend_cagr_3y": (-1.0, 3.0),
    "dividend_cagr_5y": (-1.0, 3.0),
    "yield_vs_5y_median": (-1.5, 1.5),
    "pe_ratio": (0.0, 300.0),
    "pfcf_ratio": (0.0, 300.0),
    "payout_ratio": (0.0, 3.0),
}

# === Trend deviation threshold for numeric features ===
TREND_SENSITIVE = {
    "dividend_yield": 2.5,
    "pe_ratio": 5.0,
    "pfcf_ratio": 5.0,
    "net_debt_to_ebitda": 5.0,
    "eps_cagr_3y": 5.0,
}

def validate_dynamic_row(df: pl.DataFrame, ticker: str, prev_df: Optional[pl.DataFrame] = None) -> None:
    """
    Validates the current dynamic dataframe against hard-coded numerical ranges and historical change thresholds.
    Raises ValueError if violations are found.
    """
    if df.is_empty():
        raise ValueError(f"[{ticker}] Empty dynamic_df provided.")

    latest_as_of = df["as_of"].max()
    current = df.filter(pl.col("as_of") == latest_as_of)

    # === Value range checks ===
    for col, (low, high) in FEATURE_RANGES.items():
        if col not in df.columns:
            continue

        values = df[col].to_list()
        for v in values:
            # Allow None, NaN, and inf (skip check)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                continue

            if not (low <= v <= high):
                raise ValueError(f"[{ticker}] {col} out-of-bounds: {v} not in ({low}, {high})")

    # === Historical change checker (if previous data is given) ===
    if prev_df is not None and not prev_df.is_empty():
        prev = (
            prev_df
            .filter(pl.col("as_of") < latest_as_of)
            .sort("as_of")
            .tail(1)
        )
        if prev.is_empty():
            return

        for col, max_ratio in TREND_SENSITIVE.items():
            if col not in current.columns or col not in prev.columns:
                continue

            curr_val = current[col][0]
            prev_val = prev[col][0]

            if prev_val is None or curr_val is None:
                continue
            if abs(prev_val) < 1e-6:  # Avoid division by tiny value
                continue

            ratio = abs(curr_val / prev_val)
            if ratio > max_ratio:
                raise ValueError(
                    f"[{ticker}] {col} abnormal change: {prev_val:.4f} → {curr_val:.4f} (×{ratio:.2f})"
                )
