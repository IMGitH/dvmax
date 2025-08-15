import os
import time
import json
import logging
from typing import Any, Dict, Optional

import requests

# Typed errors so your runners / workflow can branch on cause
class FMPAuthError(RuntimeError): pass        # 401 bad/missing key
class FMPPlanError(RuntimeError): pass        # 402/403 plan/forbidden
class FMPRateLimitError(RuntimeError): pass   # 429 after retries
class FMPServerError(RuntimeError): pass      # 5xx after retries

class FMPClient:
    def __init__(self, base_url: str = "https://financialmodelingprep.com/api/v3", timeout: int = 30):
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            raise FMPAuthError("FMP_API_KEY is not set in environment or .env")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.request_count = 0

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "dvmax/feature-fetcher"})

    def _sleep_backoff(self, attempt: int, retry_after: Optional[str]) -> None:
        if retry_after:
            try:
                # Honor server hint first
                sleep_s = int(retry_after)
                time.sleep(max(1, sleep_s))
                return
            except Exception:
                pass
        # Exponential backoff with jitter
        base = 1.5 ** attempt
        jitter = 0.25 + (time.time() % 0.5)  # small pseudo-jitter
        time.sleep(min(30, base + jitter))

    def fetch(self, endpoint: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Any:
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        url = f"{self.base_url}/{endpoint}"

        params = dict(params or {})
        params["apikey"] = self.api_key

        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.Timeout as e:
                last_exc = e
                if attempt == max_retries:
                    raise FMPServerError(f"Timeout contacting FMP: {e}") from e
                self._sleep_backoff(attempt, None)
                continue
            except requests.RequestException as e:
                # Network hiccup
                last_exc = e
                if attempt == max_retries:
                    raise FMPServerError(f"Network error contacting FMP: {e}") from e
                self._sleep_backoff(attempt, None)
                continue

            # Classify status codes early
            code = resp.status_code
            if code == 200:
                self.request_count += 1
                # FMP sometimes returns [] or {} — both valid
                ctype = resp.headers.get("Content-Type", "")
                if "application/json" in ctype or resp.text.strip().startswith(("{", "[")):
                    return resp.json()
                # Fallback parse
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    return resp.text

            if code == 401:
                raise FMPAuthError("401 Unauthorized: bad/missing API key")

            if code in (402, 403):
                # Try to give a helpful message
                try:
                    msg = resp.json().get("Error Message", "") or resp.text
                except Exception:
                    msg = resp.text
                raise FMPPlanError(f"{code} Forbidden/Plan limit for '{endpoint}': {msg}")

            if code == 404:
                # Many FMP endpoints 404 for delisted tickers; treat as empty
                logging.warning("FMP 404 for %s — treating as empty response", endpoint)
                return []

            if code == 429 or 500 <= code < 600:
                # Retryable: rate limit or server error
                if attempt == max_retries:
                    if code == 429:
                        raise FMPRateLimitError("429 Too Many Requests after retries")
                    raise FMPServerError(f"{code} server error after retries")
                self._sleep_backoff(attempt, resp.headers.get("Retry-After"))
                continue

            # Anything else: raise with body for diagnostics
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            resp.raise_for_status()  # will raise HTTPError including code
            return body  # Unreachable, but keeps type checkers happy
