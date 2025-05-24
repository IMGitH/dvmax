import polars as pl
from datetime import date
import random
from datetime import timedelta
from src.dataprep.features.aggregation.row_builder import \
     build_feature_table_from_inputs

def test_build_feature_table_from_inputs_minimal():
    # Prices with enough history for 6m and 12m return
    # Create a daily range from 2024-01-01 to 2025-05-17
    prices = get_random_prices()

    # Dividends for 5Y CAGR (1 per year)
    dividends = pl.DataFrame({
        "date": [
            date(2020, 6, 1),
            date(2021, 6, 1),
            date(2022, 6, 1),
            date(2023, 6, 1),
            date(2024, 6, 1)
        ],
        "dividend": [0.8, 1.0, 1.1, 1.2, 1.3]
    })

    splits = pl.DataFrame({
        "date": [],
        "numerator": [],
        "denominator": []
    })

    ratios = pl.DataFrame({
        "date": [date(2025, 5, 17)],
        "priceEarningsRatio": [20.0],
        "priceToFreeCashFlowsRatio": [25.0],
        "dividendYield": [0.012],
        "payoutRatio": [0.3]
    })

    income = pl.DataFrame({
        "date": [date(2023, 5, 17)],
        "incomeBeforeTax": [500],
        "interestExpense": [50],
        "depreciationAndAmortization": [100],
        "eps": [2.0]
    })

    balance = pl.DataFrame({
        "date": [date(2023, 5, 17)],
        "totalDebt": [1000],
        "cashAndShortTermInvestments": [200],
        "freeCashFlow": [300]
    })

    profile = {"sector": "Technology", "country": "'United States'"}

    inputs = {
        "prices": prices,
        "dividends": dividends,
        "splits": splits,
        "ratios": ratios,
        "income": income,
        "balance": balance,
        "profile": profile
    }

    result = build_feature_table_from_inputs("AAPL", inputs, as_of=date(2025, 5, 17))

    # --- Assertions ---
    assert result.shape[0] == 1

    expected_columns = {
        "ticker", "6m_return", "12m_return", "volatility", "max_drawdown_1y",
        "net_debt_to_ebitda", "ebit_interest_cover", "ebit_interest_cover_capped",
        "eps_cagr_3y", "fcf_cagr_3y",
        "dividend_cagr_5y", "yield_vs_5y_median",
        "pe_ratio", "pfcf_ratio",
        "sector_technology"
    }

    missing = expected_columns - set(result.columns)
    assert not missing, f"Missing expected columns: {missing}"

    assert result[0, "ticker"] == "AAPL"
    assert isinstance(result[0, "pe_ratio"], float)
    assert isinstance(result[0, "6m_return"], float)

def get_random_prices():
    start_date = date(2024, 1, 1)
    end_date = date(2025, 5, 17)
    n_days = (end_date - start_date).days + 1

    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Simulate prices with slight random walk from a starting value
    price = 90.0
    closes = []
    for _ in range(n_days):
        price *= 1 + random.uniform(-0.002, 0.003)  # small daily change
        closes.append(round(price, 2))

    # Create DataFrame
    prices = pl.DataFrame({
        "date": dates,
        "close": closes
    })
    
    return prices
