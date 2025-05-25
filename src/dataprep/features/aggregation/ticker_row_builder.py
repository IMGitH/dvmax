import polars as pl
from datetime import date
import numpy as np

from src.dataprep.features.engineering import (
    compute_6m_return, compute_12m_return, compute_volatility, compute_max_drawdown
)
from src.dataprep.features.engineering import (
    compute_net_debt_to_ebitda, compute_ebit_interest_cover,
    compute_dividend_cagr, compute_yield_vs_median,
    compute_eps_cagr, compute_fcf_cagr,
    extract_latest_pe_pfcf, compute_payout_ratio,
    compute_sector_relative_return,
    encode_sector, compute_sma_delta_50_250
)
from src.dataprep.fetcher.sector import extract_sector_name


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
    balance   = inputs["balance"]
    profile   = inputs["profile"]
    splits    = inputs["splits"]
    sector_df = inputs.get("sector_index", None)
    macro     = inputs.get("macro", None)

    df_fundamentals = income.join(balance, on="date", how="inner")
    df_fundamentals = compute_net_debt_to_ebitda(df_fundamentals)
    df_fundamentals = compute_ebit_interest_cover(df_fundamentals)

    pe, pfcf = extract_latest_pe_pfcf(ratios)
    if sector_df is not None and not sector_df.is_empty():
        rel_return = compute_sector_relative_return(prices, sector_df, 365, as_of)
    else:
        rel_return = 0.0

    features_price = {
        "6m_return": compute_6m_return(prices, as_of),
        "12m_return": compute_12m_return(prices, as_of),
        "volatility": compute_volatility(prices),
        "max_drawdown_1y": compute_max_drawdown(df=prices, lookback_years=1),
        "sector_relative_6m": rel_return,
        "sma_50_200_delta": compute_sma_delta_50_250(prices)
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
        "dividend_yield": safe_get(ratios, "dividendYield"),
        "dividend_cagr_3y": compute_dividend_cagr(dividends, splits, years=3),
        "dividend_cagr_5y": compute_dividend_cagr(dividends, splits, years=5),
        "yield_vs_5y_median": compute_yield_vs_median(ratios, lookback_years=5)
    }

    features_valuation = {
        "pe_ratio": pe,
        "pfcf_ratio": pfcf,
        "payout_ratio": compute_payout_ratio(ratios)
    }

    sector = extract_sector_name(profile)
    country = profile.get("country", "N/A")
    features_sector = encode_sector(sector)

    features_macro = {}
    if isinstance(macro, pl.DataFrame) and not macro.empty:
        latest_macro = macro.sort_index().iloc[-1]
        features_macro = {
            "gdp_usd": latest_macro.get("GDP (USD)", np.nan),
            "inflation_pct": latest_macro.get("Inflation (%)", np.nan),
            "unemployment_pct": latest_macro.get("Unemployment (%)", np.nan),
            "exports_pct_gdp": latest_macro.get("Exports (% GDP)", np.nan),
            "private_cons_pct_gdp": latest_macro.get("Private Consumption (% GDP)", np.nan)
        }

    all_features = {
        "ticker": ticker,
        **features_price,
        **features_fundamentals,
        **features_growth,
        **features_dividends,
        **features_valuation,
        **features_macro,
        **features_sector,
        "country": country
    }

    nullable_keys = [
        "eps_cagr_3y", "fcf_cagr_3y",
        "dividend_yield", "dividend_cagr_3y", "dividend_cagr_5y",
        "ebit_interest_cover"
    ]
    all_features = add_has_flags(all_features, nullable_keys)

    return pl.DataFrame([all_features])


def add_has_flags(feature_row: dict, nullable_keys: list[str]) -> dict:
    """
    Adds binary flags to the input dictionary indicating presence (not NaN) of each nullable key.

    Args:
        feature_row (dict): A dictionary of features with possible NaN values.
        nullable_keys (list[str]): List of keys to check for presence.

    Returns:
        dict: Updated dictionary with 'has_' flags added.
    """
    for key in nullable_keys:
        feature_row[f"has_{key}"] = int(not np.isnan(feature_row.get(key, np.nan)))
    return feature_row
