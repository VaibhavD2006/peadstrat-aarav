import polars as pl
import pytest
from aria.data.store.lake import DataLake

def test_write_and_read_prices(tmp_path):
    lake = DataLake(base_dir=str(tmp_path))
    df = pl.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "ticker": ["AAPL", "AAPL"],
        "close": [185.0, 186.0],
        "volume": [50_000_000, 48_000_000],
    })
    lake.write("prices/AAPL", df)
    result = lake.read("prices/AAPL")
    assert result.shape == (2, 4)
    assert result["close"].to_list() == [185.0, 186.0]

def test_exists_returns_false_before_write(tmp_path):
    lake = DataLake(base_dir=str(tmp_path))
    assert not lake.exists("prices/NOTHERE")

def test_exists_returns_true_after_write(tmp_path):
    lake = DataLake(base_dir=str(tmp_path))
    df = pl.DataFrame({"x": [1, 2]})
    lake.write("test/data", df)
    assert lake.exists("test/data")

def test_roundtrip_preserves_types(tmp_path):
    import datetime
    lake = DataLake(base_dir=str(tmp_path))
    df = pl.DataFrame({
        "date": [datetime.date(2024, 1, 2)],
        "close": [185.5],
        "volume": [50_000_000],
        "ticker": ["AAPL"],
    })
    lake.write("prices/test", df)
    result = lake.read("prices/test")
    assert result["close"][0] == pytest.approx(185.5)
    assert result["ticker"][0] == "AAPL"
