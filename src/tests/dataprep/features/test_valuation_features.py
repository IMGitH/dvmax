from src.dataprep.features.valuation_features import extract_latest_pe_pfcf
import polars as pl


def test_extract_latest_pe_pfcf_returns_expected_values():
    df = pl.DataFrame({
        "date": ["2023-01-01", "2023-12-31"],
        "priceEarningsRatio": [22.1, 24.0],
        "priceToFreeCashFlowsRatio": [18.0, 19.5]
    })
    pe, pfcf = extract_latest_pe_pfcf(df)
    print("\n=== test_extract_latest_pe_pfcf_returns_expected_values ===")
    print(df)
    print(f"Extracted PE: {pe}, PFCF: {pfcf}")
    assert pe == 24.0
    assert pfcf == 19.5


def test_extract_latest_pe_pfcf_from_2018():
    df = pl.DataFrame({
        "date": [
            "2018-12-31", "2019-12-31", "2020-12-31",
            "2021-12-31", "2022-12-31", "2023-12-31", "2024-06-30"
        ],
        "priceEarningsRatio": [10.2, 12.5, 15.0, 18.3, 21.7, 24.0, 25.6],
        "priceToFreeCashFlowsRatio": [8.1, 9.4, 11.2, 13.8, 16.9, 19.5, 21.3]
    })
    pe, pfcf = extract_latest_pe_pfcf(df)
    print("\n=== test_extract_latest_pe_pfcf_from_2018 ===")
    print(df)
    print(f"Extracted PE: {pe}, PFCF: {pfcf}")
    assert pe == 25.6
    assert pfcf == 21.3
