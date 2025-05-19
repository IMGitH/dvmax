from datetime import timedelta
from dateutil.relativedelta import relativedelta
import polars as pl
import datetime
import logging
from src.dataprep.features.utils import adjust_series_for_splits


def find_value_near_date(
    df: pl.DataFrame,
    target_date: datetime.date,
    column: str,
    grace_days: int = None,
    grace_months: int = None,
) -> float | None:
    if grace_days:
        lower = target_date - timedelta(days=grace_days)
        upper = target_date + timedelta(days=grace_days)
    elif grace_months:
        lower = target_date - relativedelta(months=grace_months)
        upper = target_date + relativedelta(months=grace_months)
    else:
        raise ValueError("You must specify either grace_days or grace_months")

    window = df.filter((pl.col("date") >= lower) & (pl.col("date") <= upper))
    return window[-1, column] if not window.is_empty() else None


def compute_cagr_generic(
    df: pl.DataFrame,
    column: str,
    years: int,
    grace_days: int = 90,
    grace_months: int = None,
) -> float:
    if column not in df.columns or df.height < 2:
        return 0.0

    df = df.sort("date")
    end_date = df[-1, "date"]
    end_val = df[-1, column]

    start_date = end_date - timedelta(days=365 * years)
    start_val = find_value_near_date(
        df, start_date, column, grace_days=grace_days, grace_months=grace_months
    )

    if start_val is None or start_val <= 0 or end_val <= 0:
        return 0.0

    return (end_val / start_val) ** (1 / years) - 1

def compute_dividend_cagr(
    df: pl.DataFrame,
    splits_df: pl.DataFrame,
    years: int,
    grace_months: int = 3
) -> float:
    if splits_df is None:
        raise ValueError("Split DataFrame cannot be None.")

    df = df.with_columns(pl.col("date").cast(pl.Date)).sort("date")

    if df.height < 2:
        return 0.0

    df = adjust_series_for_splits(df, splits_df, "dividend", skip_warning=True)

    return compute_cagr_generic(df, column="dividend", years=years, grace_months=grace_months)


def compute_eps_cagr(df: pl.DataFrame, years: int) -> float:
    return compute_cagr_generic(df, "eps", years)


def compute_fcf_cagr(df: pl.DataFrame, years: int) -> float:
    if "freeCashFlowPerShare" in df.columns:
        col = "freeCashFlowPerShare"
    elif "fcf" in df.columns:
        col = "fcf"
    else:
        logging.warning("[FCF] No FCF or FCF/share column found.")
        return 0.0

    return compute_cagr_generic(df, col, years)