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

def build_feature_table(ticker: str, as_of: date = date.today()) -> pl.DataFrame:
    # --- Fetch raw data ---
    prices = fetch_prices(ticker)
    dividends = fetch_dividends(ticker)
    ratios = fetch_ratios(ticker)
    balance = fetch_balance_sheet_fund(ticker)
    income = fetch_income_statement_fund(ticker)
    profile = fetch_company_profile(ticker)

    # --- Price features ---
    price_feats = {
        "6m_return": compute_6m_return(prices, as_of),
        "12m_return": compute_12m_return(prices, as_of),
        "volatility": compute_volatility(prices),
        "max_drawdown": compute_max_drawdown(prices)
    }

    # --- Fundamental features ---
    df_fundamentals = income.join(balance, on="date", how="inner")
    df_fundamentals = compute_net_debt_to_ebitda(df_fundamentals)
    df_fundamentals = compute_ebit_interest_cover(df_fundamentals)

    fundamental_feats = {
        "net_debt_to_ebitda": df_fundamentals[-1, "net_debt_to_ebitda"],
        "ebit_interest_cover": df_fundamentals[-1, "ebit_interest_cover"]
    }

    # --- Dividend features ---
    dividend_feats = {
        "dividend_cagr_5y": compute_dividend_cagr(dividends, years=5),
        "yield_vs_median": compute_yield_vs_median(ratios, lookback_years=5)
    }

    # --- Valuation features ---
    pe, pfcf = extract_latest_pe_pfcf(ratios)
    valuation_feats = {
        "pe_ratio": pe,
        "pfcf_ratio": pfcf
    }

    # --- Sector one-hot encoding ---
    sector_str = profile.get("sector", "")
    sector_feats = encode_sector(sector_str)

    # --- Merge all into one row ---
    all_features = {
        "ticker": ticker,
        **price_feats,
        **fundamental_feats,
        **dividend_feats,
        **valuation_feats,
        **sector_feats
    }

    return pl.DataFrame([all_features])


def print_feature_report(ticker: str, as_of: date = date.today()) -> None:
    print(f"\n=== Feature Report for {ticker.upper()} ===\n")
    df = build_feature_table(ticker, as_of=as_of)
    row = df.row(0, named=True)

    print(f"- As of: {as_of.isoformat()}")
    print(f"- Shape: {df.shape}")
    print(f"\n=== Column Groups ===")

    def print_group(title: str, keys: list[str]):
        print(f"\n→ {title}")
        for key in keys:
            val = row.get(key, "N/A")
            print(f"{key:25}: {val}")

    # Logical groups
    print_group("Price-Based Features", [
        "6m_return", "12m_return", "volatility", "max_drawdown"
    ])

    print_group("Fundamentals", [
        "net_debt_to_ebitda", "ebit_interest_cover"
    ])

    print_group("Dividends", [
        "dividend_cagr_5y", "yield_vs_median"
    ])

    print_group("Valuation", [
        "pe_ratio", "pfcf_ratio"
    ])

    print_group("Sector Encoding", sorted([k for k in row if k.startswith("sector_")]))

if __name__ == "__main__":
    print_feature_report("AAPL")
    