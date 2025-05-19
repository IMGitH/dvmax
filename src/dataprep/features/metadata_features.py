def encode_sector(sector: str) -> dict:
    all_sectors = [
        "Technology", "Energy", "Utilities", "Financials",
        "Consumer Staples", "Healthcare", "Industrials", "Materials", "Real Estate"
    ]
    return {
        f"sector_{s.replace(' ', '_').lower()}": int(sector == s)
        for s in all_sectors
    }
