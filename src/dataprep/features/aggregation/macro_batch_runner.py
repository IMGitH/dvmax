import polars as pl
import os
from datetime import date
from src.dataprep.fetcher.macro import WorldBankAPI
from src.dataprep.constants import MACRO_INDICATORS
import numpy as np
import logging


def engineer_macro_features(df: pl.DataFrame, as_of: date, country: str, output_root: str) -> str:
    # if as of year is current year - show a warning saying that one year previous data will be used
    if as_of.year == date.today().year:
        logging.warning(f"Using data for {as_of.year - 1} as the latest available year for {country}.")
        as_of = date(as_of.year - 1, 12, 31) 
    df = df.sort("date")
    selected_year = as_of.year
    reference_year = selected_year - 1

    # === Helper to extract value for a specific column and year ===
    def extract_by_year(df: pl.DataFrame, col: str, year: int) -> float | None:
        subset = df.filter((pl.col("date").dt.year() == year) & pl.col(col).is_not_null())
        if subset.is_empty():
            print(f"⚠️ No data found for {col} in {year}")
            return None
        try:
            return subset[0, col]
        except Exception as e:
            print(f"❌ Failed to extract {col} for year {year}: {e}")
            return None


    # === GDP and GDP per capita YoY ===
    gdp_now = extract_by_year(df, "GDP (USD)", reference_year)
    gdp_prev = extract_by_year(df, "GDP (USD)", reference_year - 1)
    gdp_yoy_backfilled = ((gdp_now - gdp_prev) / gdp_prev) if gdp_now and gdp_prev else np.nan

    gdp_pc_now = extract_by_year(df, "GDP per Capita (const USD)", reference_year)
    gdp_pc_prev = extract_by_year(df, "GDP per Capita (const USD)", reference_year - 1)
    gdp_pc_yoy_backfilled = ((gdp_pc_now - gdp_pc_prev) / gdp_pc_prev) if gdp_pc_now and gdp_pc_prev else np.nan

    # === Inflation and Unemployment ===
    infl_now = extract_by_year(df, "Inflation (%)", selected_year)
    infl_prev = extract_by_year(df, "Inflation (%)", selected_year - 1)
    inflation_latest = infl_now if infl_now is not None else np.nan
    inflation_yoy = (infl_now - infl_prev) if infl_now and infl_prev else np.nan

    unemp_latest = extract_by_year(df, "Unemployment (%)", selected_year)
    unemp_latest = unemp_latest if unemp_latest is not None else np.nan

    # === Backfilled Consumption and Exports ===
    consumption = extract_by_year(df, "Private Consumption (% GDP)", reference_year)
    exports = extract_by_year(df, "Exports (% GDP)", reference_year)
    consumption_backfilled = consumption if consumption is not None else np.nan
    exports_backfilled = exports if exports is not None else np.nan

    inflation_latest = inflation_latest / 100 if inflation_latest is not np.nan else np.nan
    inflation_yoy = inflation_yoy / 100 if inflation_yoy is not np.nan else np.nan
    unemp_latest = unemp_latest / 100 if unemp_latest is not np.nan else np.nan
    consumption_backfilled = consumption_backfilled / 100 if consumption_backfilled is not np.nan else np.nan
    exports_backfilled = exports_backfilled / 100 if exports_backfilled is not np.nan else np.nan


    # === Build feature DataFrame ===
    df_feat = pl.DataFrame([{
        "as_of_year": as_of.year,
        "backfilled_year": reference_year,
        "country": country,
        "gdp_yoy_backfilled": gdp_yoy_backfilled,
        "gdp_pc_yoy_backfilled": gdp_pc_yoy_backfilled,
        "inflation_latest": inflation_latest,
        "inflation_yoy": inflation_yoy,
        "unemployment_latest": unemp_latest,
        "consumption_backfilled": consumption_backfilled,
        "exports_backfilled": exports_backfilled
    }])

    # === Validation: Ensure no NaNs (except explicitly accepted) ===
    numeric_cols = [col for col in df_feat.columns if df_feat.schema[col] in [pl.Float32, pl.Float64]]
    nan_check_df = df_feat.select([pl.col(col).is_nan().alias(col) for col in numeric_cols])
    nan_status = nan_check_df.row(0)  # tuple of bools
    nan_cols = nan_check_df.columns   # list of col names

    if any(nan_status):
        print("❌ Missing values detected:")
        for col, is_nan in zip(nan_cols, nan_status):
            if is_nan:
                print(f"   - {col} is NaN")
        raise ValueError(f"❌ Some macro features are NaN for {country} (as_of={as_of})")


    # === Save to file ===
    today_str = date.today().strftime('%d-%m-%Y')
    output_dir = os.path.join(output_root, today_str, "macro")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{country.replace(' ', '_').lower()}.parquet")
    df_feat.write_parquet(output_path)
    print(f"✅ Saved macro features for {country} to {output_path}")
    return output_path


def fetch_and_save_macro(
    country: str,
    start_year: int,
    end_year: int = date.today().year,
    output_root: str = "features_parquet"
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

    return engineer_macro_features(df, as_of=date(end_year, 12, 31), country=country, output_root=output_root)


if __name__ == "__main__":
    # Example usage
    country = "United States"
    start_year = 2000
    output_path = fetch_and_save_macro(country, start_year)
