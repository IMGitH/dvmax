from src.dataprep.constants import ALL_SECTORS


def encode_sector(sector: str) -> dict:
    return {
        f"sector_{s.replace(' ', '_').lower()}": int(sector == s)
        for s in ALL_SECTORS
    }
