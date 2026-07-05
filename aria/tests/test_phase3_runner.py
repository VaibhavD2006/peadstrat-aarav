"""Unit tests for Phase3Runner helper methods."""
import pytest
import numpy as np
from aria.research.phase3_runner import Phase3Runner


def test_vol_target_scale_single_position():
    """1 long + 1 short at vol=0.35 should give port_vol ≈ 0.414, scale ≈ 0.362."""
    runner = Phase3Runner.__new__(Phase3Runner)  # no __init__ needed
    ticker_vols = {"AAPL": 0.35, "MSFT": 0.35}
    scale = runner._vol_target_scale(["AAPL"], ["MSFT"], ticker_vols, target_vol=0.15)
    # port_vol = sqrt(vol_L^2 + vol_S^2 - 2*rho*vol_L*vol_S)
    # With 1 long, 1 short, rho_within=0.30, rho_ls=0.30:
    # vol_L = 0.35 * sqrt(0.30 + 0.70/1) = 0.35
    # vol_S = 0.35
    # port_vol = sqrt(0.35^2 + 0.35^2 - 2*0.30*0.35*0.35) = sqrt(0.2450 - 0.0735) = sqrt(0.1715) ≈ 0.414
    # scale = 0.15 / 0.414 ≈ 0.362
    assert 0.35 <= scale <= 0.40, f"Expected scale ~0.36, got {scale:.3f}"


def test_vol_target_scale_diversified_book():
    """With 3 longs + 3 shorts, port vol is lower → scale is higher."""
    runner = Phase3Runner.__new__(Phase3Runner)
    vols = {f"L{i}": 0.35 for i in range(3)}
    vols.update({f"S{i}": 0.35 for i in range(3)})
    longs = [f"L{i}" for i in range(3)]
    shorts = [f"S{i}" for i in range(3)]
    scale_1x1 = runner._vol_target_scale(["L0"], ["S0"], {"L0": 0.35, "S0": 0.35}, 0.15)
    scale_3x3 = runner._vol_target_scale(longs, shorts, vols, 0.15)
    assert scale_3x3 > scale_1x1, "More positions should allow larger scale"


def test_vol_target_scale_capped_at_two():
    """Very low vol tickers: scale is capped at 2.0."""
    runner = Phase3Runner.__new__(Phase3Runner)
    ticker_vols = {"A": 0.05, "B": 0.05}
    scale = runner._vol_target_scale(["A"], ["B"], ticker_vols, target_vol=0.15)
    assert scale == 2.0
