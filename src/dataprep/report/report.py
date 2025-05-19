from datetime import date
import polars as pl
from src.dataprep.report.feature_table import build_feature_table_from_inputs

GROUP_PREFIXES = {
    "Price-Based Features": ["6m_", "12m_", "volatility", "max_drawdown"],
    "Fundamentals": ["net_debt", "ebit_"],
    "Growth": ["eps_cagr", "fcf_cagr"],
    "Dividends": ["dividend_", "yield_"],
    "Valuation": ["pe_ratio", "pfcf_ratio"],
    "Sector Encoding": ["sector_"]
}

SOURCE_HINTS = {
    "Price-Based Features": "prices",
    "Dividends": "dividends",
    "Valuation": "ratios",
    "Sector Encoding": "profile"
}

def print_feature_report_from_df(df: pl.DataFrame, inputs: dict, as_of: date):
    row = df.row(0, named=True)
    used_keys = set()
    sector_str = inputs.get("profile", {}).get("sector", "")

    def print_group(title: str, keys: list[str], source_df: pl.DataFrame | None = None):
        print(f"\n→ {title}")
        for key in keys:
            val = row.get(key, 'N/A')
            print(f"{key:25}: {val}")
        if source_df is not None:
            print("\nDataFrame used:")
            print(source_df)

    print(f"\n=== Feature Report for {row['ticker']} ===")
    print(f"- As of: {as_of.isoformat()}")
    print(f"- Shape: {df.shape}")

    for group_name, prefixes in GROUP_PREFIXES.items():
        keys = sorted([
            k for k in row if any(k.startswith(p) for p in prefixes)
        ])
        used_keys.update(keys)

        source_key = SOURCE_HINTS.get(group_name)
        source_df = None
        if source_key == "profile":
            source_df = pl.DataFrame([{"sector": sector_str}])
        elif source_key and source_key in inputs:
            source_df = inputs[source_key]

        print_group(group_name, keys, source_df)

    # Catch any remaining keys
    other_keys = sorted(set(row.keys()) - used_keys - {"ticker"})
    if other_keys:
        print_group("Other Features", other_keys)


if __name__ == "__main__":
    from src.dataprep.fetcher.fetch_all import fetch_all
    ticker = "AAPL"
    inputs = fetch_all(ticker, div_lookback_years=5, other_lookback_years=4)
    df = build_feature_table_from_inputs(ticker, inputs, as_of=date.today())
    print_feature_report_from_df(df, inputs, date.today())

