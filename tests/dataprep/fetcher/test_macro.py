from src.dataprep.fetcher.macro import WorldBankAPI
from src.dataprep.constants import MACRO_INDICATORS

def test_fetch_core_macro_indicators():
    api = WorldBankAPI()
    
    df = api.fetch_macro_indicators(MACRO_INDICATORS, "United States", start=2015, end=2022)

    assert not df.empty, "DataFrame should not be empty"
    assert set(MACRO_INDICATORS.values()).issubset(df.columns), "Missing expected columns"
    assert df.index.min().year >= 2015, "Start year is earlier than requested"
    assert df.index.max().year <= 2022, "End year is later than requested"
    assert df.dropna().shape[0] > 0, "No rows with complete data"
