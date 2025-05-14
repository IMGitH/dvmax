from src.dataprep.features.fundamental_features import compute_net_debt_to_ebitda, compute_ebit_interest_cover
import polars as pl

def test_compute_net_debt_to_ebitda():
    df = pl.DataFrame({
        "date": ["2023-12-31", "2022-12-31"],
        "totalDebt": [1000, 1200],
        "cashAndShortTermInvestments": [200, 300],
        "incomeBeforeTax": [400, 300],
    })
    out = compute_net_debt_to_ebitda(df)
    assert "net_debt_to_ebitda" in out.columns
    assert out["net_debt_to_ebitda"].to_list() == [(1000 - 200) / 400, (1200 - 300) / 300]

def test_compute_ebit_interest_cover():
    df = pl.DataFrame({
        "date": ["2023-12-31", "2022-12-31"],
        "incomeBeforeTax": [500, 400],
        "interestExpense": [100, 200],
    })
    out = compute_ebit_interest_cover(df)
    assert "ebit_interest_cover" in out.columns
    assert out["ebit_interest_cover"].to_list() == [5.0, 2.0]
