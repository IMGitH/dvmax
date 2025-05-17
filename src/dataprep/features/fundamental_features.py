import polars as pl

def compute_net_debt_to_ebitda(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (
            (pl.col("totalDebt") - pl.col("cashAndShortTermInvestments")) /
            pl.col("incomeBeforeTax")
        ).alias("net_debt_to_ebitda")
    ])

def compute_ebit_interest_cover(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (
            (pl.col("incomeBeforeTax") / pl.col("interestExpense"))
            .cast(pl.Float64)
            .map_elements(
                lambda x: min(x, 1000.0) if x != float("inf") else 1000.0,
                return_dtype=pl.Float64
            )
        ).alias("ebit_interest_cover")
    ])
