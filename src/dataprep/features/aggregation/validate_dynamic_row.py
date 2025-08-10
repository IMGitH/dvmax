from __future__ import annotations

from typing import Optional, Tuple, List, Dict
import polars as pl

SECTOR_BOUNDS: Dict[str, Dict[str, float]] = {
    "Financial Services":      {"pfcf_ratio_hi": 600.0, "nde_hi": 30.0},
    "Financials":              {"pfcf_ratio_hi": 600.0, "nde_hi": 30.0},
    "Industrials":             {"pfcf_ratio_hi": 500.0, "nde_hi": 25.0},
    "Technology":              {"pfcf_ratio_hi": 800.0, "nde_hi": 20.0},
    "Communication Services":  {"pfcf_ratio_hi": 700.0, "nde_hi": 20.0},
    "Consumer Discretionary":  {"pfcf_ratio_hi": 700.0, "nde_hi": 20.0},
    "Consumer Staples":        {"pfcf_ratio_hi": 500.0, "nde_hi": 15.0},
    "Health Care":             {"pfcf_ratio_hi": 700.0, "nde_hi": 20.0},
    "Energy":                  {"pfcf_ratio_hi": 400.0, "nde_hi": 15.0},
    "Materials":               {"pfcf_ratio_hi": 500.0, "nde_hi": 20.0},
    "Real Estate":             {"pfcf_ratio_hi": 500.0, "nde_hi": 25.0},
    "Utilities":               {"pfcf_ratio_hi": 400.0, "nde_hi": 15.0},
    "_default":                {"pfcf_ratio_hi": 600.0, "nde_hi": 20.0},
}

EPS = {"fcf": 1.0, "ebitda": 1.0}

REL_CHANGE_MAX = {
    "pfcf_ratio": 8.0,
    "net_debt_to_ebitda": 15.0,
}

def _get_bounds(sector: Optional[str]) -> Dict[str, float]:
    return SECTOR_BOUNDS.get(sector or "", SECTOR_BOUNDS["_default"])

def _near_zero_series(s: pl.Series, eps: float) -> pl.Series:
    return s.abs().is_between(-eps, eps)

def _relative_jump(cur: float, prev: float, eps: float = 1.0) -> float:
    return abs(cur - prev) / max(abs(prev), eps)

def validate_dynamic_row(
    df: pl.DataFrame,
    ticker: str,
    prev_df: Optional[pl.DataFrame] = None,
    sector: Optional[str] = None,
) -> Tuple[str, List[str], pl.DataFrame]:
    """
    Soft validation only:
      - Returns (status, violations, df)
      - status ∈ {'ok','flagged'}
      - Never raises; never marks fatal
      - If denominators are tiny, set the derived ratio to None and flag
    """
    violations: List[str] = []

    # Denominator guards
    if "pfcf_ratio" in df.columns and "free_cash_flow" in df.columns:
        tiny_fcf = _near_zero_series(df["free_cash_flow"], EPS["fcf"])
        if tiny_fcf.any():
            df = df.with_columns([
                pl.when(tiny_fcf).then(pl.lit(None)).otherwise(pl.col("pfcf_ratio")).alias("pfcf_ratio")
            ])
            violations.append("pfcf_ratio_nullified_tiny_fcf")

    if "net_debt_to_ebitda" in df.columns and "ebitda" in df.columns:
        tiny_ebitda = _near_zero_series(df["ebitda"], EPS["ebitda"])
        if tiny_ebitda.any():
            df = df.with_columns([
                pl.when(tiny_ebitda).then(pl.lit(None)).otherwise(pl.col("net_debt_to_ebitda")).alias("net_debt_to_ebitda")
            ])
            violations.append("nde_nullified_tiny_ebitda")

    # Sector-aware soft bounds (flags only)
    bounds = _get_bounds(sector)

    def _flag_oob(col: str, hi_key: str, label: str):
        if col in df.columns:
            s = df[col].drop_nulls()
            if len(s) > 0 and (s > bounds[hi_key]).any():
                try:
                    v = float(s.filter(s > bounds[hi_key]).to_list()[0])
                    violations.append(f"{label}_oob>{bounds[hi_key]}:{v:.4f}")
                except Exception:
                    violations.append(f"{label}_oob>{bounds[hi_key]}")

    _flag_oob("pfcf_ratio", "pfcf_ratio_hi", "pfcf_ratio")
    _flag_oob("net_debt_to_ebitda", "nde_hi", "net_debt_to_ebitda")

    # Relative change flags vs previous row
    if prev_df is not None and prev_df.height > 0:
        prev_sorted = prev_df.sort("as_of")
        for col, max_jump in REL_CHANGE_MAX.items():
            if col in df.columns and col in prev_sorted.columns:
                prev_series = prev_sorted[col].drop_nulls()
                cur_series  = df[col].drop_nulls()
                if len(prev_series) > 0 and len(cur_series) > 0:
                    prev_last = float(prev_series[-1])
                    cur = float(cur_series[0])
                    rel = _relative_jump(cur, prev_last, eps=1.0)
                    if rel > max_jump:
                        violations.append(f"{col}_jump:{rel:.2f}x")

    status = "flagged" if violations else "ok"
    return status, violations, df
