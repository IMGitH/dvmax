import pytest
import os
from datetime import date

from src.dataprep.fetcher.fetch_all import fetch_all
from src.dataprep.features.aggregation.feature_table import build_feature_table_from_inputs
from src.dataprep.visualization.report import print_feature_report_from_df

@pytest.mark.skipif(not os.getenv("FMP_API_KEY"), reason="FMP_API_KEY not set")
def test_print_report_live():
    ticker = "AAPL"
    inputs = fetch_all(ticker, div_lookback_years=5, other_lookback_years=5)
    df = build_feature_table_from_inputs(ticker, inputs, as_of=date.today())
    print_feature_report_from_df(df, inputs, date.today())  # visually check output for now
