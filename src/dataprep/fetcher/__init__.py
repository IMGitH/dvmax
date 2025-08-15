def fetch_dividends(*args, **kwargs):
    from .ticker_params.dividends import fetch_dividends as _impl
    return _impl(*args, **kwargs)

def fetch_prices(*args, **kwargs):
    from .ticker_params.prices import fetch_prices as _impl
    return _impl(*args, **kwargs)

def fetch_ratios(*args, **kwargs):
    from .ticker_params.ratios import fetch_ratios as _impl
    return _impl(*args, **kwargs)

def fetch_company_profile(*args, **kwargs):
    from .ticker_params.company import fetch_company_profile as _impl
    return _impl(*args, **kwargs)

def fetch_balance_sheet_fund(*args, **kwargs):
    from .ticker_params.fundamentals import fetch_balance_sheet_fund as _impl
    return _impl(*args, **kwargs)

def fetch_cashflow_statement_fund(*args, **kwargs):
    from .ticker_params.fundamentals import fetch_cashflow_statement_fund as _impl
    return _impl(*args, **kwargs)

def fetch_income_statement_fund(*args, **kwargs):
    from .ticker_params.fundamentals import fetch_income_statement_fund as _impl
    return _impl(*args, **kwargs)

def fmp_get(*args, **kwargs):
    from ._fmp_client import fmp_get as _impl
    return _impl(*args, **kwargs)

__all__ = [
    "fetch_dividends", "fetch_prices", "fetch_ratios", "fetch_company_profile",
    "fetch_balance_sheet_fund", "fetch_cashflow_statement_fund",
    "fetch_income_statement_fund", "fmp_get",
]
