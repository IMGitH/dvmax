import datetime
import calendar

def default_date_range(
    lookback_years: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    quarter_mode: bool = False
) -> tuple[str, str]:
    """
    Generate a safe (start_date, end_date) pair.
    - If both dates are provided: returns them.
    - If neither provided: uses `lookback_years` and current date or last full quarter.
    - If only one is provided: raises an error.

    Ensures start date exists on calendar (e.g., no Feb 30).
    """
    if (start_date and not end_date) or (end_date and not start_date):
        raise ValueError("Either provide both start_date and end_date, or neither (use lookback_years).")

    if start_date and end_date:
        return start_date, end_date

    if lookback_years is None:
        raise ValueError("lookback_years is required when start_date and end_date are not provided.")

    today = datetime.date.today()
    year, month = today.year, today.month

    if quarter_mode:
        if month <= 3:
            end = datetime.date(year - 1, 12, 31)
        elif month <= 6:
            end = datetime.date(year, 3, 31)
        elif month <= 9:
            end = datetime.date(year, 6, 30)
        else:
            end = datetime.date(year, 9, 30)
    else:
        end = today

    # Clamp day to last valid day of that month in target year
    target_year = end.year - lookback_years
    last_day = calendar.monthrange(target_year, end.month)[1]
    start_day = min(end.day, last_day)
    start = datetime.date(target_year, end.month, start_day)

    return start.isoformat(), end.isoformat()
