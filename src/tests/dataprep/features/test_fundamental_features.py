from src.dataprep.features.fundamental_features import (
    compute_net_debt_to_ebitda, 
    compute_ebit_interest_cover
)
import polars as pl

def test_compute_net_debt_to_ebitda():
    df = pl.DataFrame({
        "date": ["2023-12-31", "2022-12-31"],
        "totalDebt": [1000, 1200],
        "cashAndShortTermInvestments": [200, 300],
        "incomeBeforeTax": [400, 300],
    })
    expected = [(1000 - 200) / 400, (1200 - 300) / 300]
    out = compute_net_debt_to_ebitda(df)
    print("\n=== test_compute_net_debt_to_ebitda ===")
    print("Input:")
    print(df)
    print("Output:")
    print(out.select(["date", "net_debt_to_ebitda"]))
    print(f"Expected: {expected}")
    assert "net_debt_to_ebitda" in out.columns
    assert out["net_debt_to_ebitda"].to_list() == expected


def test_compute_ebit_interest_cover():
    df = pl.DataFrame({
        "date": ["2023-12-31", "2022-12-31"],
        "incomeBeforeTax": [500, 400],
        "interestExpense": [100, 200],
    })
    expected = [5.0, 2.0]
    out = compute_ebit_interest_cover(df)
    print("\n=== test_compute_ebit_interest_cover ===")
    print("Input:")
    print(df)
    print("Output:")
    print(out.select(["date", "ebit_interest_cover"]))
    print(f"Expected: {expected}")
    assert "ebit_interest_cover" in out.columns
    assert out["ebit_interest_cover"].to_list() == expected
