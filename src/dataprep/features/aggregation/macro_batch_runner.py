import polars as pl
import os
from datetime import date
from src.dataprep.fetcher.macro import WorldBankAPI
from src.dataprep.constants import MACRO_INDICATORS
import numpy as np
import logging
from datetime import date as _pyd
import re

def _normalize_date_column(df: pl.DataFrame) -> pl.DataFrame:
    if "date" not in df.columns:
        return df
    try:
        # Fast-path
        return df.with_columns(pl.col("date").cast(pl.Date, strict=False))
    except Exception:
        pass

    # Slow-path: rebuild from Python dates
    vals = df["date"].to_list()
    parsed: list[_pyd] = []
    pat = re.compile(r"(\d{4}).*?(\d{1,2}).*?(\d{1,2})")
    for v in vals:
        if isinstance(v, _pyd):
            parsed.append(v)
        elif isinstance(v, str):
            try:
                parsed.append(_pyd.fromisoformat(v[:10]))
            except Exception:
                m = pat.search(v)
                if not m:
                    raise ValueError(f"Cannot parse date value: {v!r}")
                y, mth, d = map(int, m.groups())
                parsed.append(_pyd(y, mth, d))
        else:
            # Handles pl.Expr or other objects by stringifying
            s = str(v)
            m = pat.search(s)
            if not m:
                raise ValueError(f"Cannot parse date value: {s!r}")
            y, mth, d = map(int, m.groups())
            parsed.append(_pyd(y, mth, d))

    return df.with_columns(pl.Series("date", parsed).cast(pl.Date))

# then at the top of engineer_macro_features:
def engineer_macro_features(df: pl.DataFrame, as_of: date, country: str, output_root: str) -> str:
    # If 'as_of' is current year, backfill to previous year and warn once
    if as_of.year == date.today().year:
        logging.warning(f"Using data for {as_of.year - 1} as the latest available year for {country}.")
        as_of = date(as_of.year - 1, 12, 31)

    # Dates: normalize once and sort
    df = _normalize_date_column(df).sort("date")

    selected_year = as_of.year
    reference_year = selected_year - 1

    def extract_by_year(df: pl.DataFrame, col: str, year: int) -> float | None:
        subset = df.filter((pl.col("date").dt.year() == year) & pl.col(col).is_not_null())
        if subset.is_empty():
            return None
        return float(subset[0, col])

    # GDP yoy
    gdp_now  = extract_by_year(df, "GDP (USD)", reference_year)
    gdp_prev = extract_by_year(df, "GDP (USD)", reference_year - 1)
    gdp_yoy_backfilled = (
        (gdp_now - gdp_prev) / gdp_prev
        if (gdp_now is not None and gdp_prev not in (None, 0.0))
        else np.nan
    )

    # GDP per capita yoy
    gdp_pc_now  = extract_by_year(df, "GDP per Capita (const USD)", reference_year)
    gdp_pc_prev = extract_by_year(df, "GDP per Capita (const USD)", reference_year - 1)
    gdp_pc_yoy_backfilled = (
        (gdp_pc_now - gdp_pc_prev) / gdp_pc_prev
        if (gdp_pc_now is not None and gdp_pc_prev not in (None, 0.0))
        else np.nan
    )

    # Inflation + yoy (levels are in %, convert later)
    infl_now  = extract_by_year(df, "Inflation (%)", selected_year)
    infl_prev = extract_by_year(df, "Inflation (%)", selected_year - 1)
    inflation_latest = infl_now if infl_now is not None else np.nan
    inflation_yoy = (
        (infl_now - infl_prev) if (infl_now is not None and infl_prev is not None) else np.nan
    )

    # Unemployment (%, convert later)
    unemp_latest = extract_by_year(df, "Unemployment (%)", selected_year)
    if unemp_latest is None:
        unemp_latest = np.nan

    # Backfilled consumption & exports (%, convert later)
    consumption = extract_by_year(df, "Private Consumption (% GDP)", reference_year)
    exports     = extract_by_year(df, "Exports (% GDP)", reference_year)

    # % fields -> proportions
    def to_prop(x):
        if x is None:
            return np.nan
        try:
            return (x / 100.0) if not np.isnan(x) else np.nan
        except TypeError:
            return np.nan

    inflation_latest = to_prop(inflation_latest)
    inflation_yoy    = to_prop(inflation_yoy)
    unemp_latest     = to_prop(unemp_latest)
    consumption_backfilled = to_prop(consumption)
    exports_backfilled     = to_prop(exports)

    df_feat = pl.DataFrame([{
        "as_of_year": int(as_of.year),
        "backfilled_year": int(reference_year),
        "country": country,
        "gdp_yoy_backfilled": float(gdp_yoy_backfilled) if gdp_yoy_backfilled is not None else np.nan,
        "gdp_pc_yoy_backfilled": float(gdp_pc_yoy_backfilled) if gdp_pc_yoy_backfilled is not None else np.nan,
        "inflation_latest": float(inflation_latest) if inflation_latest is not None else np.nan,
        "inflation_yoy": float(inflation_yoy) if inflation_yoy is not None else np.nan,
        "unemployment_latest": float(unemp_latest) if unemp_latest is not None else np.nan,
        "consumption_backfilled": float(consumption_backfilled) if consumption_backfilled is not None else np.nan,
        "exports_backfilled": float(exports_backfilled) if exports_backfilled is not None else np.nan,
    }])

    # NaN validation
    float_cols = [c for c, t in df_feat.schema.items() if t in (pl.Float32, pl.Float64)]
    nan_mask = df_feat.select([pl.col(c).is_nan().alias(c) for c in float_cols])
    if any(nan_mask.row(0)):
        bads = [c for c, is_nan in zip(nan_mask.columns, nan_mask.row(0)) if is_nan]
        raise ValueError(f"❌ Some macro features are NaN for {country} (as_of={as_of}) -> {', '.join(bads)}")

    # Save
    output_dir = os.path.join(output_root, "macro_history")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{country.replace(' ', '_').lower()}.parquet")

    if os.path.exists(output_path):
        df_existing = pl.read_parquet(output_path)
        if "as_of_year" in df_existing.columns:
            df_existing = df_existing.filter(pl.col("as_of_year") != as_of.year)
        df_feat = pl.concat([df_existing, df_feat], how="vertical").sort("as_of_year")

    df_feat.write_parquet(output_path)
    print(f"✅ Saved macro features for {country} (as_of={as_of.year}) to {output_path}")
    return output_path

def fetch_and_save_macro(
    country: str,
    start_year: int,
    end_year: int = _pyd.today().year,
    output_root: str = "features_data"
) -> str:
    macro_api = WorldBankAPI()
    df_raw = macro_api.fetch_macro_indicators(
        indicator_map=MACRO_INDICATORS,
        country_name=country,
        start=start_year,
        end=end_year
    )

    df = pl.from_pandas(df_raw.reset_index())
    df = df.with_columns(pl.col("date").cast(pl.Date))

    last_output_path = None

    # 👇 Loop over multiple years
    for year in range(start_year + 2, end_year + 1):
        as_of = _pyd(year, 12, 31) 
        try:
            last_output_path = engineer_macro_features(
                df=df,
                as_of=as_of,
                country=country,
                output_root=output_root
            )
        except ValueError as e:
            print(f"[SKIP] {country}@{as_of.year}: {e}")
    
    return last_output_path


if __name__ == "__main__":
    # Example usage
    country = "United States"
    start_year = 2000
    output_path = fetch_and_save_macro(country, start_year)
