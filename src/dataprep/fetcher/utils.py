import datetime

def default_date_range() -> tuple[str, str]:
    today = datetime.date.today()
    year = today.year
    month = today.month

    # Determine the last full quarter's end date
    if month <= 3:
        end = datetime.date(year - 1, 12, 31)
    elif month <= 6:
        end = datetime.date(year, 3, 31)
    elif month <= 9:
        end = datetime.date(year, 6, 30)
    else:
        end = datetime.date(year, 9, 30)

    start = datetime.date(end.year - 4, end.month, end.day)
    return start.isoformat(), end.isoformat()
