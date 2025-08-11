from __future__ import annotations
import polars as pl
import math
from typing import Optional


# Ranges (conservative, but less noisy)
FEATURE_RANGES = {
    "6m_return": (-1.0, 10.0),
    "12m_return": (-1.0, 20.0),
    "volatility": (0.0, 3.0),
    "max_drawdown_1y": (0.0, 1.0),
    "sector_relative_6m": (-1.0, 1.0),
    "sma_50_200_delta": (-1.0, 1.0),

    "net_debt_to_ebitda": (-10.0, 20.0),

    # NOTE: we no longer range-check raw ebit_interest_cover
    "ebit_interest_cover_capped": (0.0, 200.0),

    "eps_cagr_3y": (-1.0, 5.0),
    "fcf_cagr_3y": (-1.0, 5.0),

    "dividend_yield": (0.0, 0.25),
    "dividend_cagr_3y": (-1.0, 3.0),
    "dividend_cagr_5y": (-1.0, 3.0),
    "yield_vs_5y_median": (-0.75, 0.75),

    "pe_ratio": (0.0, 300.0),
    "pfcf_ratio": (0.0, 500.0), 
    "payout_ratio": (0.0, 2.0),
}

# Lower-bound inclusive
_LOWER_INCLUSIVE = {
    "dividend_yield", "max_drawdown_1y", "volatility",
    "payout_ratio", "pe_ratio", "pfcf_ratio",
    "ebit_interest_cover_capped",
}

# Features where missing (NaN/Inf) is OK and should NOT flag
_ALLOW_MISSING = {
    "eps_cagr_3y", "fcf_cagr_3y",
    "dividend_cagr_3y", "dividend_cagr_5y",
    "yield_vs_5y_median",
}

_TINY = 1e-6
_TINY_FCF = 1.0
_TINY_EBITDA = 1.0
_TINY_INTEREST_EXP = 1.0

# Relative jump limits with prev floors (skip if prev < floor)
_REL_JUMP_CFG = {
    "pfcf_ratio": (15.0, 1.0),
    "net_debt_to_ebitda": (25.0, 0.5),
    "dividend_yield": (10.0, 0.005),
    "pe_ratio": (12.0, 5.0),
    "payout_ratio": (5.0, 0.2),
    "volatility": (3.0, 0.05),
}


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _maybe_nullify_unstable_ratios(df: pl.DataFrame, violations: list[str]) -> pl.DataFrame:
    out = df

    # pfcf: tiny FCF -> nullify
    if "free_cash_flow" in out.columns and "pfcf_ratio" in out.columns:
        fcf = _num(out["free_cash_flow"].to_list()[0] if not out.is_empty() else None)
        if fcf is not None and abs(fcf) <= _TINY_FCF:
            out = out.with_columns(pl.lit(None).alias("pfcf_ratio"))
            violations.append("pfcf_ratio_nullified_tiny_fcf")

    # nde: tiny EBITDA -> nullify
    if "ebitda" in out.columns and "net_debt_to_ebitda" in out.columns:
        e = _num(out["ebitda"].to_list()[0] if not out.is_empty() else None)
        if e is not None and abs(e) <= _TINY_EBITDA:
            out = out.with_columns(pl.lit(None).alias("net_debt_to_ebitda"))
            violations.append("nde_nullified_tiny_ebitda")

    # interest cover: tiny interest expense or non-finite raw -> nullify RAW; keep capped for range check
    if "ebit_interest_cover" in out.columns:
        raw = _num(out["ebit_interest_cover"].to_list()[0] if not out.is_empty() else None)
        ie = _num(out["interest_expense"].to_list()[0]) if "interest_expense" in out.columns and not out.is_empty() else None
        if (ie is not None and abs(ie) <= _TINY_INTEREST_EXP) or raw is None:
            out = out.with_columns(pl.lit(None).alias("ebit_interest_cover"))
            violations.append("eic_nullified_unstable_or_nonfinite")

    return out


def _check_ranges(df: pl.DataFrame, violations: list[str]) -> None:
    if df.is_empty():
        return
    row = {c: (df[c].to_list()[0] if c in df.columns else None) for c in FEATURE_RANGES.keys()}
    for col, (lo, hi) in FEATURE_RANGES.items():
        v = _num(row.get(col))
        if v is None:
            # Missing is OK for these; do not flag
            if col in _ALLOW_MISSING:
                continue
            # For others, we stay silent too — upstream may truly lack data
            continue
        lo_ok = (v >= lo) if col in _LOWER_INCLUSIVE else (v > lo)
        hi_ok = (v < hi)
        if not (lo_ok and hi_ok):
            rng = f"[{lo}, {hi})" if col in _LOWER_INCLUSIVE else f"({lo}, {hi})"
            violations.append(f"{col} out-of-bounds: {v} not in {rng}")


def _check_relative_jumps(df: pl.DataFrame, prev_df: Optional[pl.DataFrame], violations: list[str]) -> None:
    if prev_df is None or prev_df.is_empty() or df.is_empty():
        return
    for col, (limit, prev_floor) in _REL_JUMP_CFG.items():
        if col not in df.columns or col not in prev_df.columns:
            continue
        cur = _num(df[col].to_list()[0])
        prev = _num(prev_df[col].to_list()[-1])
        if cur is None or prev is None:
            continue
        if abs(prev) < max(prev_floor, _TINY):
            continue
        ratio = abs(cur / prev)
        if ratio > limit:
            violations.append(f"{col} abnormal change: {prev:.4f} → {cur:.4f} (×{ratio:.2f})")


def _check_internal_consistency(df: pl.DataFrame, violations: list[str]) -> None:
    if {"ebit_interest_cover", "ebit_interest_cover_capped"} <= set(df.columns) and not df.is_empty():
        raw = df["ebit_interest_cover"].to_list()[0]
        capped = df["ebit_interest_cover_capped"].to_list()[0]
        if raw is not None and capped is not None:
            if capped > raw + 1e-9:
                violations.append("eic_capped_gt_raw")


def validate_dynamic_row(dynamic_df: pl.DataFrame, ticker: str, prev_df: pl.DataFrame | None = None, sector: str | None = None):
    if not isinstance(dynamic_df, pl.DataFrame) or dynamic_df.is_empty():
        return "ok", [], dynamic_df

    violations: list[str] = []
    out = _maybe_nullify_unstable_ratios(dynamic_df, violations)
    _check_ranges(out, violations)
    _check_relative_jumps(out, prev_df, violations)
    _check_internal_consistency(out, violations)

    status = "flagged" if violations else "ok"
    return status, violations, out
