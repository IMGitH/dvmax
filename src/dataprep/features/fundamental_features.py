import polars as pl
import logging
import numpy as np

def compute_net_debt_to_ebitda(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes Net Debt / EBITDA ratio.
    EBITDA is approximated as:
        EBITDA = incomeBeforeTax + interestExpense + depreciationAndAmortization

    Returns the original DataFrame with a new column: 'net_debt_to_ebitda'.
    """
    def safe_col(name: str):
        return pl.col(name) if name in df.columns else pl.lit(0)

    # Approximate EBITDA
    ebitda_expr = (
        safe_col("incomeBeforeTax") +
        safe_col("interestExpense") +
        safe_col("depreciationAndAmortization")
    )

    net_debt_expr = safe_col("totalDebt") - safe_col("cashAndShortTermInvestments")

    ratio_expr = net_debt_expr / ebitda_expr

    return df.with_columns([
        pl.when(ratio_expr.is_finite())
          .then(ratio_expr)
          .otherwise(None)
          .alias("net_debt_to_ebitda")
    ])


def compute_ebit_interest_cover(df: pl.DataFrame, cap: float = 1000.0) -> pl.DataFrame:
    # Determine which EBIT proxy to use
    if "operatingIncome" in df.columns:
        ebit_col = pl.col("operatingIncome")
    elif "incomeBeforeTax" in df.columns:
        logging.warning("[EBIT] Falling back to 'incomeBeforeTax' due to missing 'operatingIncome'")
        ebit_col = pl.col("incomeBeforeTax")
    else:
        raise ValueError("Missing both 'operatingIncome' and 'incomeBeforeTax'. Cannot compute EBIT.")

    # Interest column (fallback to 0 if missing)
    interest_col = pl.col("interestExpense") if "interestExpense" in df.columns else pl.lit(0.0)

    # Raw EBIT / interestExpense ratio (null if division by zero)
    raw_expr = pl.when(interest_col != 0).then(ebit_col / interest_col).otherwise(None)

    # Apply cap
    capped_expr = pl.when(raw_expr < cap).then(raw_expr).otherwise(np.inf)

    # Cap flag: True if cap was applied (either null or above threshold)
    cap_applied_expr = pl.when(raw_expr.is_null() | (raw_expr >= cap)).then(True).otherwise(False)

    return df.with_columns([
        raw_expr.alias("ebit_interest_cover_raw"),
        capped_expr.alias("ebit_interest_cover"),
        cap_applied_expr.alias("ebit_interest_cover_capped")
    ])
