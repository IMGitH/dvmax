SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Basic Materials": "XLB",  # sometimes shown as "Materials"
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

SECTOR_NORMALIZATION = {
    "Consumer Staples": "Consumer Defensive",
    "Financials": "Financial Services",
    "Communication": "Communication Services",
    "Telecommunication Services": "Communication Services",
    "Consumer Services": "Consumer Cyclical",
    "Basic Materials": "Materials",
}

ALL_SECTORS = set(SECTOR_TO_ETF.keys())

GROUP_PREFIXES = {
    "Price-Based Features": ["6m_", "12m_", "volatility", "max_drawdown_1y"],
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

EXPECTED_COLUMNS = [
    "ticker", "6m_return", "12m_return", "volatility", "max_drawdown_1y",
    "sector_relative_6m", "sma_50_200_delta", "net_debt_to_ebitda",
    "ebit_interest_cover", "ebit_interest_cover_capped", "eps_cagr_3y",
    "fcf_cagr_3y", "dividend_yield", "dividend_cagr_3y", "dividend_cagr_5y",
    "yield_vs_5y_median", "pe_ratio", "pfcf_ratio", "payout_ratio", "country",

    # Binary presence indicators for nullable metrics
    "has_eps_cagr_3y", "has_fcf_cagr_3y",
    "has_dividend_yield", "has_dividend_cagr_3y", "has_dividend_cagr_5y",
    "has_ebit_interest_cover"

    # Note: sector one-hot columns are still dynamic and handled separately
]