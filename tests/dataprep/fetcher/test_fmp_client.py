import requests
import pytest
from src.dataprep.fetcher._fmp_client import fmp_get, FMPAuthError, FMPRateLimitError

class R:  # tiny fake Response
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data
    def json(self):
        return self._json_data
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def test_fmp_auth_error(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return R(401, {"error":"bad key"})
    monkeypatch.setattr("src.dataprep.fetcher._fmp_client._s.get", fake_get)
    with pytest.raises(FMPAuthError):
        fmp_get("/api/v3/ping", {})

def test_fmp_rate_limit(monkeypatch):
    calls={"n":0}
    def fake_get(url, params=None, timeout=None):
        calls["n"]+=1
        return R(429, {})
    monkeypatch.setattr("src.dataprep.fetcher._fmp_client._s.get", fake_get)
    with pytest.raises(FMPRateLimitError):
        fmp_get("/api/v3/ratios/AAPL", {"limit":1})

def test_fmp_ok(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return R(200, [{"ok":1}])
    monkeypatch.setattr("src.dataprep.fetcher._fmp_client._s.get", fake_get)
    assert fmp_get("/x", {}) == [{"ok":1}]
