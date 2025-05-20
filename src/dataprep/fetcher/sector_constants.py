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
