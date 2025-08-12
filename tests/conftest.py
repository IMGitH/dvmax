import os
import pytest

@pytest.fixture(autouse=True)
def _disable_fmp_preflight(monkeypatch):
    # Avoid network/API at import & runtime during unit tests
    monkeypatch.setenv("FMP_PREFLIGHT", "0")
