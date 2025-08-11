from __future__ import annotations
import polars as pl


FEATURE_RANGES = {
    # Prices & momentum
    "6m_return": (-1.0, 10.0),           # -100% .. 1000%
    "12m_return": (-1.0, 20.0),          # -100% .. 2000%
    "volatility": (0.0, 3.0),            # annualized stdev (0..300%)
    "max_drawdown_1y": (0.0, 1.0),       # 0..100%
    "sector_relative_6m": (-1.0, 1.0),   # rel perf vs sector (-100%..+100%)
    "sma_50_200_delta": (-1.0, 1.0),     # (SMA50-SMA200)/SMA200

    # Balance sheet / profitability
    "net_debt_to_ebitda": (-10.0, 20.0),
    "ebit_interest_cover": (0.0, 200.0),         # huge values possible; see nullifier below
    "ebit_interest_cover_capped": (0.0, 200.0),  # must be ≤ raw cover

    # Growth
    "eps_cagr_3y": (-1.0, 5.0),   # -100% .. 500% CAGR
    "fcf_cagr_3y": (-1.0, 5.0),

    # Dividends
    "dividend_yield": (0.0, 0.25),         # 0..25% (0 is fine for non-payers)
    "dividend_cagr_3y": (-1.0, 3.0),
    "dividend_cagr_5y": (-1.0, 3.0),
    "yield_vs_5y_median": (-0.2, 0.2),     # -20pp .. +20pp (absolute delta)

    # Valuation
    "pe_ratio": (0.0, 300.0),
    "pfcf_ratio": (0.0, 300.0),
    "payout_ratio": (0.0, 2.0),            # allow occasional >100% payout
}

# Lower-bound inclusive columns (0 is valid)
_LOWER_INCLUSIVE = {
    "dividend_yield", "max_drawdown_1y", "volatility",
    "payout_ratio", "pe_ratio", "pfcf_ratio",
    "ebit_interest_cover", "ebit_interest_cover_capped"
}

# tiny-denominator guards
_TINY = 1e-6
_TINY_FCF = 1.0
_TINY_EBITDA = 1.0
_TINY_INTEREST_EXP = 1.0

# gentle relative jump limits (ratio of current/prev magnitudes)
_REL_JUMP_LIMIT = {
    "pfcf_ratio": 15.0,
    "net_debt_to_ebitda": 25.0,
    "dividend_yield": 10.0,
    "pe_ratio": 12.0,
    "payout_ratio": 5.0,
    "volatility": 3.0,
}

def _in_range(val, lo, hi, lower_inclusive: bool) -> bool:
    if val is None:
        return True
    try:
        v = float(val)
    except Exception:
        return True
    lo_ok = (v >= lo) if lower_inclusive else (v > lo)
    hi_ok = (v < hi)  # keep hi exclusive
    return lo_ok and hi_ok

def _maybe_nullify_unstable_ratios(df: pl.DataFrame, violations: list[str]) -> pl.DataFrame:
    out = df

    # pfcf: tiny FCF -> nullify
    if "free_cash_flow" in out.columns and "pfcf_ratio" in out.columns:
        fcf = out["free_cash_flow"].to_list()[0] if not out.is_empty() else None
        if fcf is not None and abs(float(fcf)) <= _TINY_FCF:
            out = out.with_columns(pl.lit(None).alias("pfcf_ratio"))
            violations.append("pfcf_ratio_nullified_tiny_fcf")

    # nde: tiny EBITDA -> nullify
    if "ebitda" in out.columns and "net_debt_to_ebitda" in out.columns:
        e = out["ebitda"].to_list()[0] if not out.is_empty() else None
        if e is not None and abs(float(e)) <= _TINY_EBITDA:
            out = out.with_columns(pl.lit(None).alias("net_debt_to_ebitda"))
            violations.append("nde_nullified_tiny_ebitda")

    # interest cover: tiny interest expense -> nullify both raw & capped
    if {"ebit_interest_cover", "ebit_interest_cover_capped", "interest_expense"} <= set(out.columns):
        ie = out["interest_expense"].to_list()[0]
        if ie is not None and abs(float(ie)) <= _TINY_INTEREST_EXP:
            out = out.with_columns([
                pl.lit(None).alias("ebit_interest_cover"),
                pl.lit(None).alias("ebit_interest_cover_capped"),
            ])
            violations.append("eic_nullified_tiny_interest_expense")

    return out

def _check_ranges(df: pl.DataFrame, violations: list[str]) -> None:
    if df.is_empty():
        return
    row = {c: (df[c].to_list()[0] if c in df.columns else None) for c in FEATURE_RANGES.keys()}
    for col, (lo, hi) in FEATURE_RANGES.items():
        val = row.get(col, None)
        if not _in_range(val, lo, hi, lower_inclusive=(col in _LOWER_INCLUSIVE)):
            rng = f"[{lo}, {hi})" if col in _LOWER_INCLUSIVE else f"({lo}, {hi})"
            violations.append(f"{col} out-of-bounds: {val} not in {rng}")

def _check_relative_jumps(df: pl.DataFrame, prev_df: pl.DataFrame | None, violations: list[str]) -> None:
    if prev_df is None or prev_df.is_empty() or df.is_empty():
        return
    for col, limit in _REL_JUMP_LIMIT.items():
        if col not in df.columns or col not in prev_df.columns:
            continue
        cur = df[col].to_list()[0]
        prev = prev_df[col].to_list()[-1]
        if cur is None or prev is None:
            continue
        try:
            curf, prevf = float(cur), float(prev)
        except Exception:
            continue
        if abs(prevf) < _TINY:
            continue
        ratio = abs(curf / prevf)
        if ratio > limit:
            violations.append(f"{col} abnormal change: {prevf:.4f} → {curf:.4f} (×{ratio:.2f})")

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
