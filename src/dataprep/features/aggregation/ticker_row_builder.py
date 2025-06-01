import polars as pl
from datetime import date
import numpy as np

from src.dataprep.features.engineering import (
    compute_6m_return, compute_12m_return, compute_volatility, compute_max_drawdown,
    compute_net_debt_to_ebitda, compute_ebit_interest_cover,
    compute_dividend_cagr, compute_yield_vs_median,
    compute_eps_cagr, compute_fcf_cagr,
    extract_latest_pe_pfcf, compute_payout_ratio,
    compute_sector_relative_return,
    encode_sector, compute_sma_delta_50_250
)
from src.dataprep.fetcher.ticker_params.sector import extract_sector_name


def safe_get(df: pl.DataFrame, col: str, default: float = 0.0) -> float:
    return df[-1, col] if col in df.columns and df.height > 0 else default


def add_has_flags(feature_row: dict, nullable_keys: list[str]) -> dict:
    for key in nullable_keys:
        feature_row[f"has_{key}"] = int(not np.isnan(feature_row.get(key, np.nan)))
    return feature_row


def build_feature_table_from_inputs(ticker: str, inputs: dict, as_of: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    inputs = {
        k: v.filter(pl.col("date") <= as_of).sort("date")
        if isinstance(v, pl.DataFrame) and "date" in v.columns else v
        for k, v in inputs.items()
    }

    prices = inputs["prices"]
    dividends = inputs["dividends"]
    ratios = inputs["ratios"]
    income = inputs["income"]
    balance = inputs["balance"]
    profile = inputs["profile"]
    splits = inputs["splits"]
    sector_df = inputs.get("sector_index", None)

    df_fundamentals = income.join(balance, on="date", how="inner")
    df_fundamentals = compute_net_debt_to_ebitda(df_fundamentals)
    df_fundamentals = compute_ebit_interest_cover(df_fundamentals)

    pe, pfcf = extract_latest_pe_pfcf(ratios)
    if sector_df is not None and not sector_df.is_empty():
        rel_return = compute_sector_relative_return(prices, sector_df, 365, as_of)
    else:
        rel_return = pl.Series("sector_relative_6m", [np.nan] * prices.height)


    dynamic_features = {
        "as_of": as_of,
        "6m_return": compute_6m_return(prices, as_of),
        "12m_return": compute_12m_return(prices, as_of),
        "volatility": compute_volatility(prices),
        "max_drawdown_1y": compute_max_drawdown(df=prices, lookback_years=1),
        "sector_relative_6m": rel_return,
        "sma_50_200_delta": compute_sma_delta_50_250(prices),

        "net_debt_to_ebitda": safe_get(df_fundamentals, "net_debt_to_ebitda"),
        "ebit_interest_cover": safe_get(df_fundamentals, "ebit_interest_cover"),
        "ebit_interest_cover_capped": safe_get(df_fundamentals, "ebit_interest_cover_capped"),

        "eps_cagr_3y": compute_eps_cagr(income, years=3),
        "fcf_cagr_3y": compute_fcf_cagr(ratios, years=3),

        "dividend_yield": safe_get(ratios, "dividendYield"),
        "dividend_cagr_3y": compute_dividend_cagr(dividends, splits, years=3),
        "dividend_cagr_5y": compute_dividend_cagr(dividends, splits, years=5),
        "yield_vs_5y_median": compute_yield_vs_median(ratios, lookback_years=5),

        "pe_ratio": pe,
        "pfcf_ratio": pfcf,
        "payout_ratio": compute_payout_ratio(ratios)
    }

    nullable_keys = [
        "eps_cagr_3y", "fcf_cagr_3y",
        "dividend_yield", "dividend_cagr_3y", "dividend_cagr_5y",
        "ebit_interest_cover"
    ]
    dynamic_features = add_has_flags(dynamic_features, nullable_keys)
    dynamic_features["ticker"] = ticker

    sector = extract_sector_name(profile)
    country = profile.get("country", "N/A")
    static_features = {
        "ticker": ticker,
        "country": country,
        **encode_sector(sector)
    }

    return pl.DataFrame([dynamic_features]), pl.DataFrame([static_features])
