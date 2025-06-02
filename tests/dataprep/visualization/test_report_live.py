import pytest
import os
from datetime import date

from src.dataprep.fetcher.ticker_data_sources import fetch_all_per_ticker
from src.dataprep.features.aggregation.ticker_row_builder import build_feature_table_from_inputs
from src.dataprep.visualization.report import print_feature_report_from_df

@pytest.mark.skipif(not os.getenv("FMP_API_KEY"), reason="FMP_API_KEY not set")
def test_print_report_live():
    ticker = "AAPL"
    inputs = fetch_all_per_ticker(ticker, div_lookback_years=5, other_lookback_years=5)
    df_dynamic, _ = build_feature_table_from_inputs(ticker, inputs, as_of=date.today())
    
    # ✅ Pass the correct variable
    print_feature_report_from_df(df_dynamic, inputs, date.today())  # visually inspect output for now
