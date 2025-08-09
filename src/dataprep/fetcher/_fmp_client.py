# src/dataprep/fetcher/_fmp_client.py
import os, time, requests

class FMPAuthError(RuntimeError): pass        # 401 invalid/missing key
class FMPPlanError(RuntimeError): pass        # 403/402 feature not in plan
class FMPRateLimitError(RuntimeError): pass   # 429 too many requests
class FMPServerError(RuntimeError): pass      # 5xx

_API_KEY = os.getenv("FMP_API_KEY")
if not _API_KEY:
    raise FMPAuthError("FMP_API_KEY not set")

_s = requests.Session()

def fmp_get(path: str, params=None, max_retries=3):
    url = f"https://financialmodelingprep.com{path}"
    params = dict(params or {})
    params["apikey"] = _API_KEY

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        r = _s.get(url, params=params, timeout=30)
        if r.status_code == 401:
            raise FMPAuthError("401 Unauthorized (bad/missing key)")
        if r.status_code in (402, 403):
            raise FMPPlanError(f"{r.status_code} Plan does not cover endpoint")
        if r.status_code == 429:
            if attempt == max_retries:
                raise FMPRateLimitError("429 rate limit after retries")
            time.sleep(backoff); backoff *= 2; continue
        if 500 <= r.status_code < 600:
            if attempt == max_retries:
                raise FMPServerError(f"{r.status_code} server error")
            time.sleep(backoff); backoff *= 2; continue
        r.raise_for_status()
        return r.json()
