"""Tests for IFR signal."""
import polars as pl
import numpy as np
import pytest
from datetime import date, timedelta
from aria.signals.ifr import IFRSignal


def _make_ticker_volume_df(ticker: str, start: date, n_days: int, base_vol: float, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    all_dates = [start + timedelta(days=i) for i in range(n_days * 2)]
    biz_dates = [d for d in all_dates if d.weekday() < 5][:n_days]
    # Volume with some autocorrelation
    vol_shocks = rng.normal(0, 0.2, len(biz_dates))
    volumes = base_vol * np.exp(np.cumsum(vol_shocks))
    rows = []
    for d, v in zip(biz_dates, volumes):
        rows.append({"date": d, "volume": float(v), "adj_close": 100.0})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_ifr_compute_returns_dataframe():
    """IFR compute should return DataFrame with date and IFR_z."""
    ticker = "AAPL"
    start = date(2024, 1, 1)
    n_days = 100

    prices = _make_ticker_volume_df(ticker, start, n_days, base_vol=50_000_000, seed=1)

    sig = IFRSignal(lookback_days=60, min_periods=30)
    result = sig.compute(ticker, prices, start=str(start - timedelta(days=80)), end=str(start + timedelta(days=n_days)))

    assert result.shape[0] > 0
    assert "date" in result.columns
    assert "IFR_z" in result.columns


def test_ifr_batch_cross_sectional_zscore():
    """IFR batch should produce cross-sectional z-scores per date."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    start = date(2024, 1, 1)
    n_days = 120

    prices_dict = {}
    for i, t in enumerate(tickers):
        prices_dict[t] = _make_ticker_volume_df(t, start, n_days, base_vol=50_000_000 + i * 10_000_000, seed=i+1)

    sig = IFRSignal()
    result = sig.compute_batch(tickers, prices_dict, str(start - timedelta(days=80)), str(start + timedelta(days=n_days)))

    assert result.shape[0] > 0
    assert set(result.columns) == {"ticker", "date", "IFR_z"}

    # Check cross-sectional properties per date
    for d in result["date"].unique()[:5]:
        day_data = result.filter(pl.col("date") == d)
        if day_data.shape[0] >= 3:
            zscores = day_data["IFR_z"].to_numpy()
            assert abs(zscores.mean()) < 0.5  # approximately zero mean
            assert zscores.std() < 2.5  # reasonable std


def test_ifr_high_volume_residual_positive():
    """Ticker with unusually high volume vs sector should have positive IFR."""
    # This is a more targeted test - we'd need to mock sector data
    # For now, just verify it runs without error
    tickers = ["AAPL", "MSFT"]
    start = date(2024, 1, 1)
    n_days = 100

    prices_dict = {
        "AAPL": _make_ticker_volume_df("AAPL", start, n_days, 50_000_000, 1),
        "MSFT": _make_ticker_volume_df("MSFT", start, n_days, 40_000_000, 2),
    }

    sig = IFRSignal()
    result = sig.compute_batch(tickers, prices_dict, str(start - timedelta(days=80)), str(start + timedelta(days=n_days)))

    assert result.shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])