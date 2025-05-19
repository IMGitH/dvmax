import polars as pl
import logging
from datetime import timedelta
from src.dataprep.features.utils import adjust_series_for_splits


def compute_cagr(df: pl.DataFrame, column: str, years: int, grace_days: int = 45) -> float:
    df = df.sort("date")

    if column not in df.columns or df.height < 2:
        return 0.0

    end_date = df[-1, "date"]
    end_val = df[-1, column]

    start_date = end_date - timedelta(days=365 * years)
    grace = timedelta(days=grace_days)

    past_df = df.filter(
        (pl.col("date") >= start_date - grace) & 
        (pl.col("date") <= start_date + grace)
    )

    if past_df.is_empty():
        return 0.0

    start_val = past_df[-1, column]
    if start_val <= 0 or end_val <= 0:
        return 0.0

    return (end_val / start_val) ** (1 / years) - 1


def compute_eps_cagr(df: pl.DataFrame, years: int) -> float:
    return compute_cagr(df, "eps", years)


def compute_fcf_cagr(df: pl.DataFrame, years: int) -> float:
    if "freeCashFlowPerShare" in df.columns:
        col = "freeCashFlowPerShare"
    elif "fcf" in df.columns:
        col = "fcf"
    else:
        logging.warning("[FCF] No FCF or FCF/share column found.")
        return 0.0

    return compute_cagr(df, col, years)