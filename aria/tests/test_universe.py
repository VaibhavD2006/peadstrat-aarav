import polars as pl
import pytest
from aria.data.universe import UniverseFilter

def test_filter_excludes_low_adv():
    filt = UniverseFilter(min_adv_usd=5_000_000, min_price=5.0)
    prices = pl.DataFrame({
        "ticker": ["AAPL", "TINY", "MID"],
        "close": [185.0, 3.0, 50.0],
        "volume": [50_000_000, 100_000, 200_000],
        "adv_20d_usd": [9_250_000_000.0, 300_000.0, 10_000_000.0],
    })
    result = filt.apply(prices)
    assert "TINY" not in result["ticker"].to_list()
    assert "AAPL" in result["ticker"].to_list()
    assert "MID" in result["ticker"].to_list()

def test_filter_excludes_low_price():
    filt = UniverseFilter(min_adv_usd=1_000, min_price=5.0)
    prices = pl.DataFrame({
        "ticker": ["CHEAP", "OK"],
        "close": [2.0, 10.0],
        "volume": [1_000_000, 1_000_000],
        "adv_20d_usd": [2_000_000.0, 10_000_000.0],
    })
    result = filt.apply(prices)
    assert result["ticker"].to_list() == ["OK"]

def test_compute_adv_adds_column():
    import datetime
    filt = UniverseFilter()
    dates = [datetime.date(2024, 1, i+1) for i in range(25)]
    prices = pl.DataFrame({
        "date": dates,
        "ticker": ["AAPL"] * 25,
        "close": [100.0 + i for i in range(25)],
        "volume": [1_000_000] * 25,
    })
    result = filt.compute_adv(prices)
    assert "adv_20d_usd" in result.columns
    # First 19 rows will be null (not enough history), row 20+ should be non-null
    non_null = result["adv_20d_usd"].drop_nulls()
    assert non_null.shape[0] > 0
