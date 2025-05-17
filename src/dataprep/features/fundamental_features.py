import polars as pl

def compute_net_debt_to_ebitda(df: pl.DataFrame) -> pl.DataFrame:
    def safe_col(name):
        return pl.col(name) if name in df.columns else pl.lit(0)

    ebitda_expr = (
        safe_col("incomeBeforeTax") +
        safe_col("interestExpense") +
        safe_col("depreciationAndAmortization")
    )

    net_debt_expr = pl.col("totalDebt") - pl.col("cashAndShortTermInvestments")

    ratio_expr = (net_debt_expr / ebitda_expr)

    return df.with_columns([
        pl.when(ratio_expr.is_finite())
        .then(ratio_expr)
        .otherwise(None)
        .alias("net_debt_to_ebitda")
    ])


def compute_ebit_interest_cover(df: pl.DataFrame) -> pl.DataFrame:
    ratio = pl.when(pl.col("interestExpense") != 0) \
        .then(pl.col("incomeBeforeTax") / pl.col("interestExpense")) \
        .otherwise(None)  # or 0.0 if preferred

    capped = pl.when(ratio.is_not_null() & (ratio < 1000.0)) \
        .then(ratio) \
        .otherwise(1000.0)

    return df.with_columns([capped.alias("ebit_interest_cover")])