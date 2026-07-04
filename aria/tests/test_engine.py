import polars as pl
import numpy as np
import pytest
from datetime import date
from aria.backtest.engine import BacktestEngine, BacktestConfig, Position

def _make_prices(tickers=None, start=date(2024, 1, 2), end=date(2024, 3, 31)):
    if tickers is None:
        tickers = ["AAPL", "MSFT", "AMZN", "GOOG"]
    # Generate all calendar days then filter to weekdays (Mon=0 .. Fri=4)
    all_dates = pl.date_range(start, end, interval="1d", eager=True)
    dates = [d for d in all_dates.to_list() if d.weekday() < 5]
    rng = np.random.default_rng(42)
    rows = []
    for ticker in tickers:
        returns = rng.normal(0.001, 0.015, len(dates))
        prices  = 100.0 * np.cumprod(1 + returns)
        for d, p in zip(dates, prices):
            rows.append({
                "date": d, "ticker": ticker,
                "open": float(p * 0.999), "close": float(p),
                "adj_close": float(p), "adv_20d_usd": 5e9,
            })
    return pl.DataFrame(rows)

def test_engine_produces_records():
    prices = _make_prices()
    config = BacktestConfig(hold_days=5, initial_capital=1_000_000)
    engine = BacktestEngine(config=config)
    signals = pl.DataFrame({
        "ticker":      ["AAPL", "MSFT"],
        "entry_date":  [date(2024, 1, 11), date(2024, 1, 11)],
        "side":        ["long", "short"],
        "weight":      [0.5, 0.5],
    })
    results = engine.run(signals=signals, prices=prices)
    assert results.shape[0] == 2
    assert "pnl" in results.columns

def test_long_and_short_pnl_opposite_sign_on_same_return():
    """If price goes up, long gains and short loses (approximately)."""
    # Use deterministic prices: AAPL goes up 5% over hold period
    all_dates = pl.date_range(date(2024, 1, 2), date(2024, 1, 31), interval="1d", eager=True)
    dates = [d for d in all_dates.to_list() if d.weekday() < 5]
    prices_arr = [100.0 * (1.005 ** i) for i in range(len(dates))]
    rows = []
    for d, p in zip(dates, prices_arr):
        rows.append({"date": d, "ticker": "AAPL", "open": p, "close": p,
                     "adj_close": p, "adv_20d_usd": 5e9})
    prices = pl.DataFrame(rows)

    config = BacktestConfig(hold_days=5, initial_capital=100_000)
    engine = BacktestEngine(config=config)

    long_signals  = pl.DataFrame({"ticker": ["AAPL"], "entry_date": [date(2024, 1, 3)],
                                   "side": ["long"],  "weight": [1.0]})
    short_signals = pl.DataFrame({"ticker": ["AAPL"], "entry_date": [date(2024, 1, 3)],
                                   "side": ["short"], "weight": [1.0]})
    r_long  = engine.run(signals=long_signals,  prices=prices)
    r_short = engine.run(signals=short_signals, prices=prices)
    assert r_long["pnl"][0] > 0
    assert r_short["pnl"][0] < 0

def test_engine_empty_when_no_prices():
    prices = _make_prices(tickers=["AAPL"])
    config = BacktestConfig(hold_days=5, initial_capital=1_000_000)
    engine = BacktestEngine(config=config)
    signals = pl.DataFrame({
        "ticker": ["NOTEXIST"], "entry_date": [date(2024, 1, 11)],
        "side": ["long"], "weight": [1.0],
    })
    results = engine.run(signals=signals, prices=prices)
    assert results.shape[0] == 0
