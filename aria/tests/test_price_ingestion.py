import polars as pl
import pytest
from unittest.mock import patch
from aria.data.ingestion.price import PriceLoader

def _mock_df():
    return pl.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "open": [184.0, 185.0],
        "high": [186.0, 187.0],
        "low":  [183.0, 184.0],
        "adj_close": [185.0, 186.0],
        "volume": [50_000_000, 48_000_000],
        "ticker": ["AAPL", "AAPL"],
    }).with_columns(pl.col("date").cast(pl.Date))

def test_price_loader_returns_expected_schema(tmp_path):
    loader = PriceLoader(cache_dir=str(tmp_path))
    with patch.object(loader, '_fetch_from_yfinance', return_value=_mock_df()):
        result = loader.load("AAPL", start="2024-01-01", end="2024-01-05")
    assert "ticker" in result.columns
    assert "adj_close" in result.columns
    assert result.shape[0] == 2

def test_price_loader_caches_on_second_call(tmp_path):
    loader = PriceLoader(cache_dir=str(tmp_path))
    with patch.object(loader, '_fetch_from_yfinance', return_value=_mock_df()) as mock:
        loader.load("AAPL", start="2024-01-01", end="2024-01-05")
        loader.load("AAPL", start="2024-01-01", end="2024-01-05")
    assert mock.call_count == 1

def test_load_many_concatenates(tmp_path):
    loader = PriceLoader(cache_dir=str(tmp_path))
    def mock_fetch(ticker, start, end):
        return pl.DataFrame({
            "date": ["2024-01-02"], "open": [100.0], "high": [101.0], "low": [99.0],
            "adj_close": [100.5], "volume": [1_000_000], "ticker": [ticker],
        }).with_columns(pl.col("date").cast(pl.Date))
    with patch.object(loader, '_fetch_from_yfinance', side_effect=mock_fetch):
        result = loader.load_many(["AAPL", "MSFT"], start="2024-01-01", end="2024-01-05")
    assert result.shape[0] == 2
    assert set(result["ticker"].to_list()) == {"AAPL", "MSFT"}
