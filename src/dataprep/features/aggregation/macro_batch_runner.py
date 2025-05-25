import polars as pl
import os
from datetime import date, datetime
from src.dataprep.fetcher.macro import WorldBankAPI
from src.dataprep.constants import MACRO_INDICATORS

def engineer_macro_features(df: pl.DataFrame, as_of: date, country: str, output_dir: str) -> str:
    """
    Computes and saves engineered macroeconomic features from raw time series.

    Returns:
        str: path to the saved Parquet file
    """
    df = df.sort("date")
    latest_date = df.filter(pl.col("date") <= pl.lit(as_of)).select(pl.col("date").max()).item()
    prev_date = latest_date.replace(year=latest_date.year - 1)

    def safe(col: str, d: date) -> float | None:
        try:
            return df.filter(pl.col("date") == d).select(col).item()
        except:
            return None

    gdp_now = safe("GDP (USD)", latest_date)
    gdp_prev = safe("GDP (USD)", prev_date)

    gdp_pc_now = safe("GDP per Capita (const USD)", latest_date)
    gdp_pc_prev = safe("GDP per Capita (const USD)", prev_date)

    infl_now = safe("Inflation (%)", latest_date)
    infl_prev = safe("Inflation (%)", prev_date)

    unemp_now = safe("Unemployment (%)", latest_date)
    cons_now = safe("Private Consumption (% GDP)", latest_date)
    exports_now = safe("Exports (% GDP)", latest_date)

    infl_vol_3y = (
        df.filter((pl.col("date").dt.year() >= latest_date.year - 2) & (pl.col("date") <= latest_date))
          .select(pl.col("Inflation (%)").std())
          .item()
    )

    df_feat = pl.DataFrame([{
        "as_of": latest_date,
        "gdp_yoy": ((gdp_now - gdp_prev) / gdp_prev) if gdp_now and gdp_prev else None,
        "gdp_pc_yoy": ((gdp_pc_now - gdp_pc_prev) / gdp_pc_prev) if gdp_pc_now and gdp_pc_prev else None,
        "inflation_latest": infl_now,
        "inflation_yoy": (infl_now - infl_prev) if infl_now and infl_prev else None,
        "inflation_vol_3y": infl_vol_3y,
        "unemployment_latest": unemp_now,
        "consumption_latest": cons_now,
        "exports_latest": exports_now
    }])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{country.replace(' ', '_').lower()}.parquet")
    df_feat.write_parquet(output_path)
    return output_path

def fetch_and_save_macro(
    country: str,
    start_year: int,
    end_year: int = date.today().year,
    output_dir: str = "features_parquet/macro"
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

    return engineer_macro_features(df, as_of=date(end_year, 12, 31), country=country, output_dir=output_dir)
