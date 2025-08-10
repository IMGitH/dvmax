# src/dataprep/features/aggregation/validate_dynamic_row.py
from __future__ import annotations
from typing import Iterable, Tuple, List, Optional
import polars as pl

# Minimal, conservative ranges; expand if needed
FEATURE_RANGES = {
    "dividend_yield": (0.0, 0.25),            # 0% – 25%
    "pfcf_ratio": (0.0, 300.0),               # 0 – 300
    "net_debt_to_ebitda": (-10.0, 20.0),      # broad, allows negs for net cash
}

# thresholds under which a ratio is unreliable -> nullify instead of flagging
# tuned to tests (FCF=0.2, EBITDA=0.4 should nullify)
_TINY_FCF = 1.0
_TINY_EBITDA = 1.0
_TINY = 1e-6


def _in_range(val: Optional[float], lo: float, hi: float) -> bool:
    if val is None:
        return True  # None is handled elsewhere; not a violation here
    try:
        return (val > lo) and (val < hi)
    except Exception:
        return True


def _maybe_nullify_unstable_ratios(df: pl.DataFrame, violations: List[str]) -> pl.DataFrame:
    out = df

    # pfcf: tiny free_cash_flow -> nullify pfcf_ratio
    if "free_cash_flow" in out.columns and "pfcf_ratio" in out.columns:
        fcf = out["free_cash_flow"].to_list()[0] if not out.is_empty() else None
        if fcf is not None and abs(float(fcf)) <= _TINY_FCF:
            out = out.with_columns(pl.lit(None).alias("pfcf_ratio"))
            violations.append("pfcf_ratio_nullified_tiny_fcf")

    # net_debt_to_ebitda: tiny ebitda -> nullify net_debt_to_ebitda
    if "ebitda" in out.columns and "net_debt_to_ebitda" in out.columns:
        e = out["ebitda"].to_list()[0] if not out.is_empty() else None
        if e is not None and abs(float(e)) <= _TINY_EBITDA:
            out = out.with_columns(pl.lit(None).alias("net_debt_to_ebitda"))
            violations.append("nde_nullified_tiny_ebitda")

    return out


def _check_ranges(df: pl.DataFrame, violations: List[str]) -> None:
    # Check each configured feature when present
    row = {c: (df[c].to_list()[0] if c in df.columns and not df.is_empty() else None) for c in FEATURE_RANGES.keys()}
    for col, (lo, hi) in FEATURE_RANGES.items():
        val = row.get(col, None)
        if val is None:
            continue
        try:
            v = float(val)
        except Exception:
            continue
        if not _in_range(v, lo, hi):
            violations.append(f"{col} out-of-bounds: {v} not in ({lo}, {hi})")


def _check_relative_jumps(df: pl.DataFrame, prev_df: Optional[pl.DataFrame], violations: List[str]) -> None:
    if prev_df is None or prev_df.is_empty():
        return

    def _rel_jump(col: str, limit: float) -> None:
        if col not in df.columns or col not in prev_df.columns:
            return
        cur = df[col].to_list()[0]
        prev = prev_df[col].to_list()[-1]
        if cur is None or prev is None:
            return
        try:
            curf, prevf = float(cur), float(prev)
        except Exception:
            return
        if abs(prevf) < _TINY:
            return
        ratio = abs(curf / prevf)
        if ratio > limit:
            violations.append(f"{col} abnormal change: {prevf:.4f} → {curf:.4f} (×{ratio:.2f})")

    # gentle limits; tune as needed
    _rel_jump("pfcf_ratio", 15.0)
    _rel_jump("net_debt_to_ebitda", 25.0)
    _rel_jump("dividend_yield", 10.0)   # <-- add yield trend check



def validate_dynamic_row(
    dynamic_df: pl.DataFrame,
    ticker: str,
    prev_df: Optional[pl.DataFrame] = None,
    sector: Optional[str] = None,
) -> Tuple[str, List[str], pl.DataFrame]:
    """
    Soft validator:
      - never raises
      - returns ("ok" | "flagged", violations, possibly_mutated_df)

    Mutations:
      - nullifies unstable ratios when denominators are tiny
    Flags:
      - range violations (FEATURE_RANGES)
      - large relative jumps vs prev_df (lightweight)
    """
    if not isinstance(dynamic_df, pl.DataFrame) or dynamic_df.is_empty():
        return "ok", [], dynamic_df

    violations: List[str] = []

    # 1) Nullify unstable ratios based on tiny denominators
    out = _maybe_nullify_unstable_ratios(dynamic_df, violations)

    # 2) Hard ranges (soft-flag)
    _check_ranges(out, violations)

    # 3) Relative jumps w.r.t previous row (soft-flag)
    _check_relative_jumps(out, prev_df, violations)

    status = "flagged" if violations else "ok"
    return status, violations, out
