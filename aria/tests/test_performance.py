import numpy as np
import pytest
from aria.backtest.performance import PerformanceAnalytics

def test_sharpe_positive_for_positive_returns():
    returns = np.full(252, 0.001)
    pa = PerformanceAnalytics()
    assert pa.sharpe(returns, rf_annual=0.0) > 0.0

def test_sharpe_roughly_correct():
    # daily return = 0.1%, vol = 0 → infinite sharpe capped by std check
    # daily return = 0%, vol = 1% → sharpe ~= 0
    returns = np.random.default_rng(99).normal(0, 0.01, 252)
    pa = PerformanceAnalytics()
    sharpe = pa.sharpe(returns, rf_annual=0.0)
    assert abs(sharpe) < 3.0   # random walk ~ 0 Sharpe

def test_max_drawdown_correct():
    equity = np.array([100.0, 110.0, 90.0, 95.0])
    pa = PerformanceAnalytics()
    mdd = pa.max_drawdown(equity)
    expected = (90.0 - 110.0) / 110.0
    assert abs(mdd - expected) < 1e-6

def test_max_drawdown_zero_for_monotone_rising():
    equity = np.array([100.0, 110.0, 120.0, 130.0])
    pa = PerformanceAnalytics()
    assert pa.max_drawdown(equity) == pytest.approx(0.0)

def test_ic_perfect_rank_correlation():
    signals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    pa = PerformanceAnalytics()
    ic = pa.information_coefficient(signals, returns)
    assert ic == pytest.approx(1.0)

def test_ic_nan_for_too_few_observations():
    pa = PerformanceAnalytics()
    ic = pa.information_coefficient(np.array([1.0, 2.0]), np.array([0.01, 0.02]))
    assert np.isnan(ic)

def test_summarize_returns_all_keys():
    returns = np.random.default_rng(0).normal(0.001, 0.01, 252)
    pa = PerformanceAnalytics()
    summary = pa.summarize(returns)
    for key in ["sharpe", "annual_return", "annual_vol", "max_drawdown", "n_periods"]:
        assert key in summary
