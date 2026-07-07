"""Tests for IVRS signal."""
import polars as pl
import numpy as np
import pytest
from datetime import date, timedelta
from aria.signals.ivrs import IVRSignal


def _make_price_df(tickers: list[str], start: date, n_days: int, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    all_dates = [start + timedelta(days=i) for i in range(n_days * 2)]
    biz_dates = [d for d in all_dates if d.weekday() < 5][:n_days]
    rows = []
    for ticker in tickers:
        # Different volatility regimes per ticker
        base_vol = 0.01 + hash(ticker) % 3 * 0.005
        returns = rng.normal(0.0005, base_vol, len(biz_dates))
        prices = 100.0 * np.cumprod(1 + returns)
        for d, p in zip(biz_dates, prices):
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": float(p * 0.999),
                "close": float(p),
                "adj_close": float(p),
                "volume": 1_000_000
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_ivrs_computes_zscore():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    prices = _make_price_df(tickers, date(2024, 1, 1), 100)

    sig = IVRSignal(short_window=20, long_window=60)
    result = sig.compute(prices)

    assert result.shape[0] > 0
    assert "IVRS_z" in result.columns
    assert "ticker" in result.columns
    assert "date" in result.columns

    # Check z-score properties (approximately mean 0, std 1 per date)
    for d in result["date"].unique()[:3]:
        day_data = result.filter(pl.col("date") == d)["IVRS_z"].to_numpy()
        assert len(day_data) >= 3
        assert abs(day_data.mean()) < 1.5  # roughly centered
        assert day_data.std() < 3.5  # winsorized


def test_ivrs_high_vol_ticker_has_higher_ratio():
    """Ticker with higher short-term vol should have higher vol_ratio."""
    tickers = ["LOW_VOL", "HIGH_VOL"]
    start = date(2024, 1, 1)
    n_days = 100

    # LOW_VOL: stable
    rng = np.random.default_rng(1)
    dates = [start + timedelta(days=i) for i in range(n_days * 2)]
    biz_dates = [d for d in dates if d.weekday() < 5][:n_days]
    rows = []
    for ticker, vol in [("LOW_VOL", 0.005), ("HIGH_VOL", 0.025)]:
        returns = rng.normal(0.0005, vol, len(biz_dates))
        prices = 100.0 * np.cumprod(1 + returns)
        for d, p in zip(biz_dates, prices):
            rows.append({"date": d, "ticker": ticker, "adj_close": float(p)})
    prices = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

    sig = IVRSignal(short_window=20, long_window=60)
    result = sig.compute(prices)

    # Get latest IVRS for each
    latest = result.sort("date").group_by("ticker").tail(1)
    low_vol_ivrs = latest.filter(pl.col("ticker") == "LOW_VOL")["IVRS_z"].item()
    high_vol_ivrs = latest.filter(pl.col("ticker") == "HIGH_VOL")["IVRS_z"].item()

    # High vol ticker should have higher IVRS (more stress)
    assert high_vol_ivrs > low_vol_ivrs


def test_ivrs_latest_returns_single_row_per_ticker():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    prices = _make_price_df(tickers, date(2024, 1, 1), 100)

    sig = IVRSignal()
    latest = sig.compute_latest(prices)

    assert latest.shape[0] == 3
    assert set(latest["ticker"].to_list()) == set(tickers)
    assert "IVRS_z" in latest.columns


def test_ivrs_winsorization():
    """Test that extreme values are clipped."""
    tickers = [f"T{i}" for i in range(10)]
    prices = _make_price_df(tickers, date(2024, 1, 1), 100)

    sig = IVRSignal()
    result = sig.compute(prices)

    # All values should be within [-3, 3] due to winsorization
    ivrs_vals = result["IVRS_z"].to_numpy()
    assert ivrs_vals.max() <= 3.0 + 1e-10
    assert ivrs_vals.min() >= -3.0 - 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])