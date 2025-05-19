import polars as pl
from datetime import date

from src.dataprep.features.price_features import (
    compute_6m_return, compute_12m_return, compute_volatility, compute_max_drawdown
)
from src.dataprep.features import (
    compute_net_debt_to_ebitda, compute_ebit_interest_cover,
    compute_dividend_cagr, compute_yield_vs_median,
    compute_eps_cagr, compute_fcf_cagr,
    extract_latest_pe_pfcf, compute_payout_ratio, compute_sector_relative_return,
    encode_sector
)

def safe_get(df: pl.DataFrame, col: str, default: float = 0.0) -> float:
    return df[-1, col] if col in df.columns and df.height > 0 else default

def build_feature_table_from_inputs(ticker: str, inputs: dict, as_of: date) -> pl.DataFrame:
    # Filter all DataFrames to as_of
    inputs = {
        k: v.filter(pl.col("date") <= as_of).sort("date")
        if isinstance(v, pl.DataFrame) and "date" in v.columns else v
        for k, v in inputs.items()
    }

    prices    = inputs["prices"]
    dividends = inputs["dividends"]
    ratios    = inputs["ratios"]
    income    = inputs["income"]
    # "incomeBeforeTax", "interestExpense", "eps", "netIncome", "revenue", "operatingIncome", "grossProfitRatio", 
    #     "ebitdaratio", "operatingIncomeRatio", "netIncomeRatio", "interestExpense", "depreciationAndAmortization", "weightedAverageShsOut"
    balance   = inputs["balance"]
    profile   = inputs["profile"]
    splits    = inputs["splits"]
    sector_df = inputs.get("sector_index", None)

    df_fundamentals = income.join(balance, on="date", how="inner")
    df_fundamentals = compute_net_debt_to_ebitda(df_fundamentals)
    df_fundamentals = compute_ebit_interest_cover(df_fundamentals)

    pe, pfcf = extract_latest_pe_pfcf(ratios)
    if sector_df is not None and not sector_df.is_empty():
        rel_return = compute_sector_relative_return(prices, sector_df, 180, as_of)
    else:
        rel_return = 0.0
    features_price = {
        "6m_return": compute_6m_return(prices, as_of),
        "12m_return": compute_12m_return(prices, as_of),
        "volatility": compute_volatility(prices),
        "max_drawdown": compute_max_drawdown(prices),
        "sector_relative_6m": rel_return
    }

    features_fundamentals = {
        "net_debt_to_ebitda": safe_get(df_fundamentals, "net_debt_to_ebitda"),
        "ebit_interest_cover": safe_get(df_fundamentals, "ebit_interest_cover"),
        "ebit_interest_cover_capped": safe_get(df_fundamentals, "ebit_interest_cover_capped")
    }

    features_growth = {
        "eps_cagr_3y": compute_eps_cagr(income, years=3),
        "fcf_cagr_3y": compute_fcf_cagr(ratios, years=3),
    }
    
    features_dividends = {
        "dividend_yield": safe_get(ratios, "dividendYield"), # dividends?
        "dividend_cagr_3y": compute_dividend_cagr(dividends, splits, years=3),
        "dividend_cagr_5y": compute_dividend_cagr(dividends, splits, years=5),
        "yield_vs_median": compute_yield_vs_median(ratios, lookback_years=5)
    }

    features_valuation = {
        "pe_ratio": pe,
        "pfcf_ratio": pfcf,
        "payout_ratio": compute_payout_ratio(ratios)
    }

    features_sector = encode_sector(profile.get("sector", ""))

    all_features = {
        "ticker": ticker,
        **features_price,
        **features_fundamentals,
        **features_growth,
        **features_dividends,
        **features_valuation,
        **features_sector
    }

    return pl.DataFrame([all_features])
