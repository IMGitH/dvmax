import os
import tempfile
from src.dataprep.universe.populate_tickers import is_valid_ticker, save_tickers_to_file

def test_is_valid_ticker():
    assert is_valid_ticker("AAPL")
    assert is_valid_ticker("BRK.B")
    assert not is_valid_ticker("^GSPC")
    assert not is_valid_ticker("1234567")
    assert not is_valid_ticker("aapl")  # must be uppercase

def test_save_tickers_to_file():
    tickers = ["AAPL", "MSFT", "GOOGL"]

    with tempfile.NamedTemporaryFile(mode='r+', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_tickers_to_file.__globals__["TICKERS_FILE"] = tmp_path
        save_tickers_to_file(tickers)

        with open(tmp_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        assert sorted(set(tickers)) == sorted(lines)

    finally:
        os.remove(tmp_path)
