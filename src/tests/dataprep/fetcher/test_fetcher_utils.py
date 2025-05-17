from src.dataprep.fetcher.utils import default_date_range
import datetime

def test_default_date_range_returns_4_year_span_ending_last_quarter():
    start_str, end_str = default_date_range()
    start = datetime.date.fromisoformat(start_str)
    end = datetime.date.fromisoformat(end_str)
    today = datetime.date.today()
    print("\n=== test_default_date_range_returns_4_year_span_ending_last_quarter ===")
    print(f"Today: {today}")
    print(f"Start: {start}  |  End: {end}")
    print(f"Delta in years: {end.year - start.year}")
    print(f"End month: {end.month}, End day: {end.day}")
    # Check that range is ~4 years
    assert (end.year - start.year) == 4
    # Check that end is the last full quarter (end of quarter - not current month)
    assert end < today
    assert end.day in {30, 31}  # not 1st of month
    assert end.month in {3, 6, 9, 12}  # end of a quarter
    # Ensure start matches day/month of end, but 4 years earlier
    assert start.day == end.day
    assert start.month == end.month
