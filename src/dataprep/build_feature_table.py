import polars as pl
from datetime import date

from src.dataprep.fetcher.prices import fetch_prices
from src.dataprep.fetcher.dividends import fetch_dividends
from src.dataprep.fetcher.ratios import fetch_ratios
from src.dataprep.fetcher.fundamentals import fetch_balance_sheet_fund, fetch_income_statement_fund
from src.dataprep.fetcher.company import fetch_company_profile

from src.dataprep.features.price_features import (
    compute_6m_return, compute_12m_return,
    compute_volatility, compute_max_drawdown
)

from src.dataprep.features.fundamental_features import (
    compute_net_debt_to_ebitda, compute_ebit_interest_cover
)

from src.dataprep.features.dividend_features import (
    compute_dividend_cagr, compute_yield_vs_median
)

from src.dataprep.features.valuation_features import extract_latest_pe_pfcf
from src.dataprep.features.metadata_features import encode_sector


def build_feature_table(ticker: str,
                        div_lookback_years: int,
                        other_lookback_years: int,
                        as_of: date = date.today()) -> pl.DataFrame:
    prices = fetch_prices(ticker, lookback_years=div_lookback_years)
    dividends = fetch_dividends(ticker, lookback_years=div_lookback_years)
    ratios = fetch_ratios(ticker, limit=other_lookback_years)
    balance = fetch_balance_sheet_fund(ticker, limit=other_lookback_years)
    income = fetch_income_statement_fund(ticker, limit=other_lookback_years)
    profile = fetch_company_profile(ticker)

    price_feats = {
        "6m_return": compute_6m_return(prices, as_of),
        "12m_return": compute_12m_return(prices, as_of),
        "volatility": compute_volatility(prices),
        "max_drawdown": compute_max_drawdown(prices)
    }

    df_fundamentals = income.join(balance, on="date", how="inner")
    df_fundamentals = compute_net_debt_to_ebitda(df_fundamentals)
    df_fundamentals = compute_ebit_interest_cover(df_fundamentals)

    fundamental_feats = {
        "net_debt_to_ebitda": df_fundamentals[-1, "net_debt_to_ebitda"],
        "ebit_interest_cover": df_fundamentals[-1, "ebit_interest_cover"],
        "ebit_interest_cover_capped": df_fundamentals[-1, "ebit_interest_cover_capped"]
    }

    dividend_feats = {
        "dividend_cagr_5y": compute_dividend_cagr(dividends, years=5),
        "yield_vs_median": compute_yield_vs_median(ratios, lookback_years=5)
    }

    pe, pfcf = extract_latest_pe_pfcf(ratios)
    valuation_feats = {
        "pe_ratio": pe,
        "pfcf_ratio": pfcf
    }

    sector_str = profile.get("sector", "")
    sector_feats = encode_sector(sector_str)

    all_features = {
        "ticker": ticker,
        **price_feats,
        **fundamental_feats,
        **dividend_feats,
        **valuation_feats,
        **sector_feats
    }

    return pl.DataFrame([all_features])


def print_feature_report(ticker: str, 
                         div_lookback_years: int, 
                         other_lookback_years: int, 
                         as_of: date = date.today()) -> None:
    print(f"\n=== Feature Report for {ticker.upper()} ===\n")
    df = build_feature_table(ticker, div_lookback_years, other_lookback_years, as_of)
    row = df.row(0, named=True)

    print(f"- As of: {as_of.isoformat()}")
    print(f"- Shape: {df.shape}")
    print(f"\n=== Column Groups ===")

    prices = fetch_prices(ticker, lookback_years=div_lookback_years)
    dividends = fetch_dividends(ticker, lookback_years=div_lookback_years)
    ratios = fetch_ratios(ticker, limit=other_lookback_years)
    balance = fetch_balance_sheet_fund(ticker, limit=other_lookback_years)
    income = fetch_income_statement_fund(ticker, limit=other_lookback_years)
    profile = fetch_company_profile(ticker)
    sector_str = profile.get("sector", "")
    df_fundamentals = income.join(balance, on="date", how="inner")
    df_fundamentals = compute_net_debt_to_ebitda(df_fundamentals)
    df_fundamentals = compute_ebit_interest_cover(df_fundamentals)

    def print_group(title: str, keys: list[str], source_df: pl.DataFrame | None = None):
        print(f"\n→ {title}")
        for key in keys:
            val = row.get(key, "N/A")
            print(f"{key:25}: {val}")
        if source_df is not None:
            print("\nDataFrame used:")
            print(source_df)

    print_group("Price-Based Features", [
        "6m_return", "12m_return", "volatility", "max_drawdown"
    ], prices)

    print_group("Fundamentals", [
        "net_debt_to_ebitda",
        "ebit_interest_cover",
        "ebit_interest_cover_capped"
    ], df_fundamentals)

    print_group("Dividends", [
        "dividend_cagr_5y", "yield_vs_median"
    ], dividends)

    print_group("Valuation", [
        "pe_ratio", "pfcf_ratio"
    ], ratios)

    sector_df = pl.DataFrame([{"sector": sector_str}])
    print_group("Sector Encoding", sorted([k for k in row if k.startswith("sector_")]), sector_df)


if __name__ == "__main__":
    print_feature_report("AAPL", div_lookback_years=5, other_lookback_years=3)

