"""Tests for HMM regime classifier."""
import polars as pl
import numpy as np
import pytest
from datetime import date, timedelta
from aria.signals.hmm import RegimeHMM, create_regime_filter


def _make_regime_price_data(n_days: int = 500, seed: int = 42) -> pl.DataFrame:
    """Generate price data with distinct regime-like behavior."""
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 1)
    all_dates = [start + timedelta(days=i) for i in range(n_days * 2)]
    biz_dates = [d for d in all_dates if d.weekday() < 5][:n_days]

    # Create 4 regime segments with different characteristics
    n_per_regime = n_days // 4
    returns = []

    # Regime 0: Bull - positive drift, low vol
    returns.extend(rng.normal(0.001, 0.008, n_per_regime))
    # Regime 1: Bear - negative drift, moderate vol
    returns.extend(rng.normal(-0.0005, 0.015, n_per_regime))
    # Regime 2: Crisis - large negative, high vol
    returns.extend(rng.normal(-0.003, 0.03, n_per_regime))
    # Regime 3: Recovery - positive drift, high vol
    returns.extend(rng.normal(0.0015, 0.02, n_per_regime))

    # If more days needed, repeat pattern
    while len(returns) < n_days:
        returns.extend(rng.normal(0.0005, 0.012, min(n_per_regime, n_days - len(returns))))

    returns = returns[:n_days]
    prices = 100.0 * np.cumprod(1 + np.array(returns))

    rows = []
    for d, p in zip(biz_dates, prices):
        rows.append({"date": d, "adj_close": float(p), "volume": 1_000_000})

    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_hmm_fits_and_predicts():
    """HMM should fit and produce regime predictions."""
    prices = _make_regime_price_data(600)

    hmm = RegimeHMM(n_states=4)
    hmm.fit(prices, window=20)

    regimes = hmm.predict(prices, window=20)

    assert regimes.shape[0] > 100  # Should have predictions for most days
    assert "regime" in regimes.columns
    assert "date" in regimes.columns
    assert regimes["regime"].min() >= 0
    assert regimes["regime"].max() <= 3


def test_hmm_regime_stats():
    """HMM should produce sensible regime statistics."""
    prices = _make_regime_price_data(600)

    hmm = RegimeHMM(n_states=4)
    hmm.fit(prices, window=20)

    stats = hmm.get_regime_stats(prices, window=20)

    assert len(stats) == 4
    for regime in range(4):
        assert "name" in stats[regime]
        assert "mean_return" in stats[regime]
        assert "ann_return" in stats[regime]
        assert "ann_vol" in stats[regime]
        assert "sharpe" in stats[regime]

    # Check regime ordering: Bull (0) should have highest Sharpe
    bull_sharpe = stats[0]["sharpe"]
    bear_sharpe = stats[1]["sharpe"]
    crisis_sharpe = stats[2]["sharpe"]
    recovery_sharpe = stats[3]["sharpe"]

    # Bull should be best or close to best
    assert bull_sharpe >= crisis_sharpe
    # Crisis should be worst
    assert crisis_sharpe <= bear_sharpe
    assert crisis_sharpe <= recovery_sharpe


def test_hmm_transition_matrix():
    """HMM should produce valid transition matrix."""
    prices = _make_regime_price_data(600)

    hmm = RegimeHMM(n_states=4)
    hmm.fit(prices, window=20)

    transmat = hmm.get_transition_matrix()

    assert transmat.shape == (4, 4)
    # Rows should sum to 1
    row_sums = transmat.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    # All probabilities should be non-negative
    assert (transmat >= 0).all()


def test_hmm_predict_proba():
    """HMM should produce probability predictions."""
    prices = _make_regime_price_data(600)

    hmm = RegimeHMM(n_states=4)
    hmm.fit(prices, window=20)

    probas = hmm.predict_proba(prices, window=20)

    assert probas.shape[1] == 4
    # Each row should sum to 1
    row_sums = probas.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    # All probabilities should be in [0, 1]
    assert (probas >= 0).all() and (probas <= 1).all()


def test_regime_filter():
    """create_regime_filter should produce binary filter."""
    prices = _make_regime_price_data(600)

    filter_df = create_regime_filter(prices, allowed_regimes=[0, 3], window=20)

    assert filter_df.shape[0] > 100
    assert "regime" in filter_df.columns
    assert "regime_filter" in filter_df.columns
    assert set(filter_df["regime_filter"].unique().to_list()).issubset({0, 1})

    # Should have some 1s and some 0s
    assert filter_df["regime_filter"].sum() > 0
    assert filter_df["regime_filter"].sum() < filter_df.shape[0]


def test_hmm_regime_labels():
    """Regime labels should match expected names."""
    hmm = RegimeHMM()
    labels = hmm.regime_labels()

    assert labels == ["Bull", "Bear", "Crisis", "Recovery"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])