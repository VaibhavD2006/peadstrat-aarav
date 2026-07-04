import pytest
import numpy as np
from aria.backtest.costs import TransactionCostModel

def test_large_cap_spread_is_5bps():
    model = TransactionCostModel(spread_bps_large=5, spread_bps_mid=10)
    assert model.spread_cost_bps(is_large_cap=True) == 5.0

def test_mid_cap_spread_is_10bps():
    model = TransactionCostModel(spread_bps_large=5, spread_bps_mid=10)
    assert model.spread_cost_bps(is_large_cap=False) == 10.0

def test_total_cost_large_cap_reasonable_range():
    model = TransactionCostModel(spread_bps_large=5, market_impact_mult=10, participation_cap=0.10)
    # 1% participation → sqrt(0.1) * 10 ≈ 3.16 bps impact + 5 bps spread = ~8.16
    cost = model.total_cost_bps(order_usd=1_000_000, adv_20d_usd=100_000_000, is_large_cap=True)
    assert 5 < cost < 20

def test_mid_cap_total_higher_than_large_cap():
    model = TransactionCostModel(spread_bps_large=5, spread_bps_mid=10, market_impact_mult=10)
    cost_large = model.total_cost_bps(1_000_000, 100_000_000, is_large_cap=True)
    cost_mid   = model.total_cost_bps(1_000_000, 100_000_000, is_large_cap=False)
    assert cost_mid > cost_large

def test_daily_borrow_cost():
    model = TransactionCostModel(borrow_cost_annual_bps=30)
    daily = model.daily_borrow_cost_bps()
    assert abs(daily - 30 / 252) < 0.001

def test_higher_participation_means_higher_impact():
    model = TransactionCostModel(market_impact_mult=10, participation_cap=0.10)
    small = model.market_impact_bps(100_000, 100_000_000)
    large = model.market_impact_bps(10_000_000, 100_000_000)
    assert large > small
