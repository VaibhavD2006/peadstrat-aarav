"""Tests for PEADSignal."""
import polars as pl
import pytest
from datetime import date, timedelta
from aria.signals.pead import PEADSignal


def _make_prices(ticker_closes: dict[str, list[tuple[date, float]]]) -> pl.DataFrame:
    rows = []
    for ticker, daily in ticker_closes.items():
        for d, c in daily:
            rows.append({"ticker": ticker, "date": d, "close": c})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_pead_positive_and_negative():
    """Ticker A gaps up, B gaps down — PEAD_z signs should differ."""
    announce = date(2024, 1, 15)
    prices = _make_prices({
        "A": [(date(2024, 1, 12), 100.0), (announce, 110.0), (date(2024, 1, 16), 111.0)],
        "B": [(date(2024, 1, 12), 100.0), (announce, 90.0),  (date(2024, 1, 16), 89.0)],
        "C": [(date(2024, 1, 12), 100.0), (announce, 100.5), (date(2024, 1, 16), 100.5)],
    })
    event_by_date = {announce: ["A", "B", "C"]}
    result = PEADSignal().compute_batch(prices, event_by_date)

    assert result.shape[0] == 3
    a_z = float(result.filter(pl.col("ticker") == "A")["PEAD_z"][0])
    b_z = float(result.filter(pl.col("ticker") == "B")["PEAD_z"][0])
    assert a_z > 0, "gap-up ticker should have positive PEAD_z"
    assert b_z < 0, "gap-down ticker should have negative PEAD_z"


def test_pead_entry_date_is_next_weekday():
    """Entry date should be the next weekday after announce_date."""
    announce = date(2024, 1, 12)  # Friday
    prices = _make_prices({
        "A": [(date(2024, 1, 11), 100.0), (announce, 105.0)],
        "B": [(date(2024, 1, 11), 100.0), (announce, 95.0)],
        "C": [(date(2024, 1, 11), 100.0), (announce, 100.0)],
    })
    event_by_date = {announce: ["A", "B", "C"]}
    result = PEADSignal().compute_batch(prices, event_by_date)
    expected_entry = date(2024, 1, 15)  # Monday
    assert all(result["entry_date"].to_list()[i] == expected_entry for i in range(result.shape[0]))


def test_pead_returns_empty_on_too_few_tickers():
    """Fewer than 3 tickers in cohort — skip (cannot z-score reliably)."""
    announce = date(2024, 1, 15)
    prices = _make_prices({
        "A": [(date(2024, 1, 12), 100.0), (announce, 105.0)],
        "B": [(date(2024, 1, 12), 100.0), (announce, 90.0)],
    })
    event_by_date = {announce: ["A", "B"]}
    result = PEADSignal().compute_batch(prices, event_by_date)
    assert result.shape[0] == 0


def test_pead_missing_prior_price_skipped():
    """Tickers without a prior-day price are silently skipped."""
    announce = date(2024, 1, 15)
    prices = _make_prices({
        "A": [(announce, 105.0)],  # no prior day price
        "B": [(date(2024, 1, 12), 100.0), (announce, 90.0)],
        "C": [(date(2024, 1, 12), 100.0), (announce, 100.0)],
        "D": [(date(2024, 1, 12), 100.0), (announce, 102.0)],
    })
    event_by_date = {announce: ["A", "B", "C", "D"]}
    result = PEADSignal().compute_batch(prices, event_by_date)
    assert "A" not in result["ticker"].to_list()
    assert result.shape[0] == 3


def test_pead_earnings_reaction_correct():
    """Verify earnings_reaction = (close_day0 - prior_close) / prior_close."""
    announce = date(2024, 1, 15)
    prices = _make_prices({
        "A": [(date(2024, 1, 12), 200.0), (announce, 210.0)],
        "B": [(date(2024, 1, 12), 100.0), (announce, 90.0)],
        "C": [(date(2024, 1, 12), 50.0),  (announce, 50.0)],
    })
    event_by_date = {announce: ["A", "B", "C"]}
    result = PEADSignal().compute_batch(prices, event_by_date)
    a_row = result.filter(pl.col("ticker") == "A")
    assert abs(float(a_row["earnings_reaction"][0]) - 0.05) < 1e-9


def test_pead_multiple_event_dates():
    """Two separate announce dates produce independent z-score cohorts."""
    d1 = date(2024, 1, 15)
    d2 = date(2024, 2, 15)
    prices = _make_prices({
        "A": [(date(2024, 1, 12), 100.0), (d1, 110.0)],
        "B": [(date(2024, 1, 12), 100.0), (d1, 90.0)],
        "C": [(date(2024, 1, 12), 100.0), (d1, 100.0)],
        "X": [(date(2024, 2, 12), 200.0), (d2, 220.0)],
        "Y": [(date(2024, 2, 12), 200.0), (d2, 180.0)],
        "Z": [(date(2024, 2, 12), 200.0), (d2, 200.0)],
    })
    event_by_date = {d1: ["A", "B", "C"], d2: ["X", "Y", "Z"]}
    result = PEADSignal().compute_batch(prices, event_by_date)
    assert result.shape[0] == 6
    d1_rows = result.filter(pl.col("announce_date") == d1)
    d2_rows = result.filter(pl.col("announce_date") == d2)
    assert d1_rows.shape[0] == 3
    assert d2_rows.shape[0] == 3
