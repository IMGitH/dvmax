import datetime

def default_date_range() -> tuple[str, str]:
    today = datetime.date.today()
    quarter_month = (today.month - 1) // 3 * 3
    last_quarter_end = datetime.date(today.year, quarter_month, 1) - datetime.timedelta(days=1)
    start = datetime.date(last_quarter_end.year - 4, last_quarter_end.month, last_quarter_end.day)
    return start.isoformat(), last_quarter_end.isoformat()
