from src.dataprep.features import encode_sector
from src.dataprep.fetcher import fetch_company_profile


def test_encode_sector_one_hot_correctness():
    sector = "Utilities"
    encoded = encode_sector(sector)

    print("\n=== Encoded One-Hot ===")
    print(f"Input Sector: {sector}")
    print(f"Encoded Dict: {encoded}")

    assert encoded["sector_utilities"] == 1
    assert sum(encoded.values()) == 1
    assert all(key.startswith("sector_") for key in encoded)


def test_encode_sector_from_fmp_profile():
    profile = {"sector": "Technology"}
    sector = profile.get("sector", "")

    print("\n=== FMP Profile Sector ===")
    print(f"Extracted Sector: {sector}")

    assert isinstance(sector, str)
    assert len(sector) > 0

    encoded = encode_sector(sector)

    print("Encoded Sector Vector:")
    print(encoded)

    expected_key = f"sector_{sector.replace(' ', '_').lower()}"
    assert isinstance(encoded, dict)
    assert sum(encoded.values()) == 1
    assert expected_key in encoded
