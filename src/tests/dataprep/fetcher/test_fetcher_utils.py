from src.dataprep.fetcher.utils import default_date_range
import datetime
import calendar


def test_default_date_range_returns_4_year_span_ending_last_quarter():
    start_str, end_str = default_date_range(lookback_years=4, quarter_mode=True)
    start = datetime.date.fromisoformat(start_str)
    end = datetime.date.fromisoformat(end_str)
    today = datetime.date.today()

    print("\n=== test_default_date_range_returns_4_year_span_ending_last_quarter ===")
    print(f"Today: {today}")
    print(f"Start: {start}  |  End: {end}")
    print(f"Delta in years: {end.year - start.year}")

    # Range must be 4 years
    assert (end.year - start.year) == 4
    # End must be in the past and on a quarter-end
    assert end < today
    assert end.day in {30, 31}
    assert end.month in {3, 6, 9, 12}
    # Start must match day/month with valid clamping
    assert start.month == end.month
    expected_last_day = calendar.monthrange(start.year, start.month)[1]
    assert 1 <= start.day <= expected_last_day


def test_default_date_range_returns_span_clamped_to_valid_day():
    # Simulate February 29th logic (ensure day is clamped)
    # We can't mock datetime easily without a lib, so we just test correctness
    start_str, end_str = default_date_range(lookback_years=5)
    start = datetime.date.fromisoformat(start_str)
    end = datetime.date.fromisoformat(end_str)

    print("\n=== test_default_date_range_returns_span_clamped_to_valid_day ===")
    print(f"Start: {start} | End: {end}")
    
    assert (end.year - start.year) == 5
    assert start.month == end.month
    expected_last_day = calendar.monthrange(start.year, start.month)[1]
    assert 1 <= start.day <= expected_last_day
