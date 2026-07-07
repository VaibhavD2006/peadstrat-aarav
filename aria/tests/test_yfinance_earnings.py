"""Tests for the yfinance earnings date loader."""
import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from aria.data.ingestion.yfinance_earnings import (
    fetch_earnings_dates,
    load_earnings_dates,
    build_announce_map,
    _load_from_cache,
    _save_to_cache,
    CACHE_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yf_df(dates_reported: list[tuple[str, float | None]]) -> pd.DataFrame:
    """Build a minimal fake yfinance earnings DataFrame."""
    index = pd.to_datetime([d for d, _ in dates_reported], utc=True)
    reported = [r for _, r in dates_reported]
    return pd.DataFrame({"Reported EPS": reported}, index=index)


# ---------------------------------------------------------------------------
# fetch_earnings_dates
# ---------------------------------------------------------------------------

def test_fetch_returns_confirmed_dates_only():
    fake_df = _make_yf_df([
        ("2023-10-26", 1.46),   # confirmed
        ("2023-07-27", 1.26),   # confirmed
        ("2024-02-01", None),   # future estimate → excluded
    ])
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates.return_value = fake_df

    with patch("aria.data.ingestion.yfinance_earnings.yf.Ticker", return_value=mock_ticker):
        dates = fetch_earnings_dates("AAPL")

    assert dates == [date(2023, 7, 27), date(2023, 10, 26)]


def test_fetch_returns_empty_on_error():
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates.side_effect = Exception("network error")

    with patch("aria.data.ingestion.yfinance_earnings.yf.Ticker", return_value=mock_ticker):
        dates = fetch_earnings_dates("AAPL")

    assert dates == []


def test_fetch_returns_empty_when_no_confirmed_rows():
    fake_df = _make_yf_df([("2025-01-01", None), ("2025-04-01", None)])
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates.return_value = fake_df

    with patch("aria.data.ingestion.yfinance_earnings.yf.Ticker", return_value=mock_ticker):
        dates = fetch_earnings_dates("AAPL")

    assert dates == []


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path):
    expected = [date(2022, 1, 28), date(2022, 7, 29)]
    with patch("aria.data.ingestion.yfinance_earnings.CACHE_DIR", tmp_path):
        _save_to_cache("AAPL", expected)
        loaded = _load_from_cache("AAPL")
    assert loaded == expected


def test_load_from_cache_returns_none_when_missing(tmp_path):
    with patch("aria.data.ingestion.yfinance_earnings.CACHE_DIR", tmp_path):
        result = _load_from_cache("MISSING")
    assert result is None


def test_load_earnings_dates_uses_cache(tmp_path):
    cached_dates = [date(2022, 10, 27), date(2023, 2, 2)]
    (tmp_path / "AAPL.json").write_text(json.dumps([str(d) for d in cached_dates]))

    with patch("aria.data.ingestion.yfinance_earnings.CACHE_DIR", tmp_path):
        with patch("aria.data.ingestion.yfinance_earnings.fetch_earnings_dates") as mock_fetch:
            result = load_earnings_dates(["AAPL"], verbose=False)

    # Should NOT have called the network
    mock_fetch.assert_not_called()
    assert result["AAPL"] == cached_dates


def test_load_earnings_dates_fetches_on_cache_miss(tmp_path):
    fetched = [date(2022, 10, 27)]
    with patch("aria.data.ingestion.yfinance_earnings.CACHE_DIR", tmp_path):
        with patch("aria.data.ingestion.yfinance_earnings.fetch_earnings_dates",
                   return_value=fetched) as mock_fetch:
            result = load_earnings_dates(["AAPL"], delay=0, verbose=False)

    mock_fetch.assert_called_once_with("AAPL", limit=40, request_timeout=5.0)
    assert result["AAPL"] == fetched


# ---------------------------------------------------------------------------
# build_announce_map
# ---------------------------------------------------------------------------

def test_map_picks_closest_yf_date_before_pubdate():
    # yfinance shows Oct 26; SimFin publishes Oct 29 (3 days later)
    yf_dates = {"AAPL": [date(2023, 10, 26)]}
    pairs = [("AAPL", date(2023, 10, 29))]
    mapping = build_announce_map(pairs, yf_dates, window_days=10)
    assert mapping[("AAPL", date(2023, 10, 29))] == date(2023, 10, 26)


def test_map_falls_back_to_pubdate_when_no_match():
    # yfinance has dates far from the publish date
    yf_dates = {"AAPL": [date(2023, 6, 1)]}
    pairs = [("AAPL", date(2023, 10, 29))]
    mapping = build_announce_map(pairs, yf_dates, window_days=10)
    assert mapping[("AAPL", date(2023, 10, 29))] == date(2023, 10, 29)


def test_map_falls_back_when_ticker_not_in_yf():
    pairs = [("XYZ", date(2023, 10, 29))]
    mapping = build_announce_map(pairs, yf_dates={}, window_days=10)
    assert mapping[("XYZ", date(2023, 10, 29))] == date(2023, 10, 29)


def test_map_ignores_yf_dates_after_pubdate():
    # yfinance date is AFTER publish date — should not match
    yf_dates = {"AAPL": [date(2023, 11, 5)]}
    pairs = [("AAPL", date(2023, 10, 29))]
    mapping = build_announce_map(pairs, yf_dates, window_days=10)
    assert mapping[("AAPL", date(2023, 10, 29))] == date(2023, 10, 29)


def test_map_picks_closest_when_multiple_candidates():
    # Two yfinance dates in window; closer one should win
    yf_dates = {"AAPL": [date(2023, 10, 20), date(2023, 10, 27)]}
    pairs = [("AAPL", date(2023, 10, 29))]
    mapping = build_announce_map(pairs, yf_dates, window_days=20)
    assert mapping[("AAPL", date(2023, 10, 29))] == date(2023, 10, 27)  # 2 days before


def test_map_handles_multiple_tickers():
    yf_dates = {
        "AAPL": [date(2023, 10, 26)],
        "MSFT": [date(2023, 10, 24)],
    }
    pairs = [("AAPL", date(2023, 10, 28)), ("MSFT", date(2023, 10, 27))]
    mapping = build_announce_map(pairs, yf_dates)
    assert mapping[("AAPL", date(2023, 10, 28))] == date(2023, 10, 26)
    assert mapping[("MSFT", date(2023, 10, 27))] == date(2023, 10, 24)
