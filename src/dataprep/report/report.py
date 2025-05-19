from datetime import date
import polars as pl


def print_feature_report_from_df(df: pl.DataFrame, inputs: dict, as_of: date):
    row = df.row(0, named=True)
    sector_str = inputs["profile"].get("sector", "")

    def print_group(title: str, keys: list[str], source_df: pl.DataFrame | None = None):
        print(f"\n→ {title}")
        for key in keys:
            print(f"{key:25}: {row.get(key, 'N/A')}")
        if source_df is not None:
            print("\nDataFrame used:")
            print(source_df)

    print(f"\n=== Feature Report for {row['ticker']} ===")
    print(f"- As of: {as_of.isoformat()}")
    print(f"- Shape: {df.shape}")

    print_group("Price-Based Features", [
        "6m_return", "12m_return", "volatility", "max_drawdown"
    ], inputs["prices"])

    print_group("Fundamentals", [
        "net_debt_to_ebitda", "ebit_interest_cover", "ebit_interest_cover_capped"
    ])  # removed df_fundamentals

    print_group("Growth", [
        "eps_cagr_3y", "fcf_cagr_3y"
    ])

    print_group("Dividends", [
        "dividend_cagr_5y", "yield_vs_median"
    ], inputs["dividends"])

    print_group("Valuation", [
        "pe_ratio", "pfcf_ratio"
    ], inputs["ratios"])

    print_group("Sector Encoding", sorted(k for k in row if k.startswith("sector_")),
                pl.DataFrame([{"sector": sector_str}]))


# if __name__ == "__main__":
