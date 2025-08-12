import re

from src.dataprep.constants import ALL_SECTORS, ALL_COUNTRIES


def _slug(s: str) -> str:
    if not s:
        return "unknown"
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.strip()).strip("_").lower()
    return s or "unknown"

def _encode_one_hot(value: str | None, vocab: list[str], prefix: str) -> dict[str, int]:
    val = (value or "").strip()
    # normalize some common aliases
    if prefix == "sector" and val.lower() in {"technology", "it"}:
        val = "Information Technology"
    cols = {}
    hit = False
    for item in vocab:
        col = f"{prefix}_{_slug(item)}"
        is_one = (val == item)
        cols[col] = 1 if is_one else 0
        hit = hit or is_one
    cols[f"{prefix}_other"] = 0 if hit else 1
    return cols

def encode_sector(sector: str | None) -> dict[str, int]:
    return _encode_one_hot(sector, ALL_SECTORS, "sector")

def encode_country(country: str | None) -> dict[str, int]:
    return _encode_one_hot(country, ALL_COUNTRIES, "country")
