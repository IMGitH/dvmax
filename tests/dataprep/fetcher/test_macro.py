from src.dataprep.fetcher.macro import WorldBankAPI

def test_fetch_core_macro_indicators():
    api = WorldBankAPI()
    
    indicators = {
        "NY.GDP.MKTP.CD": "GDP (USD)",  # Total economic output
        "FP.CPI.TOTL.ZG": "Inflation (%)",  # CPI year-over-year
        "SL.UEM.TOTL.ZS": "Unemployment (%)",  # Labor market stress
        "NE.EXP.GNFS.ZS": "Exports (% GDP)",  # External demand
        "NE.CON.PRVT.ZS": "Private Consumption (% GDP)"  # Internal demand
    }

    df = api.fetch_macro_indicators(indicators, "United States", start=2015, end=2022)

    assert not df.empty, "DataFrame should not be empty"
    assert set(indicators.values()).issubset(df.columns), "Missing expected columns"
    assert df.index.min().year >= 2015, "Start year is earlier than requested"
    assert df.index.max().year <= 2022, "End year is later than requested"
    assert df.dropna().shape[0] > 0, "No rows with complete data"
