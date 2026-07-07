"""Tests for BSQSignal."""
import polars as pl
import pytest
from aria.signals.bsq import BSQSignal


def _make_rows(data: list[dict]) -> list[dict]:
    return [{"ticker": d["ticker"], **{k: v for k, v in d.items() if k != "ticker"}} for d in data]


def test_bsq_scores_and_eligibility():
    """High-quality firm should be long eligible; low-quality short eligible."""
    rows = [
        {"ticker": "GOOD", "accruals": 0.05,  "cfo_margin": 0.20, "debt_burden": 0.10, "cash_quality": 1.0},
        {"ticker": "MED",  "accruals": 0.00,  "cfo_margin": 0.10, "debt_burden": 0.00, "cash_quality": 0.5},
        {"ticker": "BAD",  "accruals": -0.10, "cfo_margin": -0.05,"debt_burden": -0.30,"cash_quality": 0.0},
        {"ticker": "BAD2", "accruals": -0.08, "cfo_margin": -0.03,"debt_burden": -0.20,"cash_quality": 0.0},
        {"ticker": "BAD3", "accruals": -0.09, "cfo_margin": -0.04,"debt_burden": -0.25,"cash_quality": 0.0},
    ]
    result = BSQSignal().compute_batch(rows)
    good = result.filter(pl.col("ticker") == "GOOD")
    bad = result.filter(pl.col("ticker") == "BAD")
    assert bool(good["long_eligible"][0]) is True
    assert float(good["BSQ_score"][0]) > float(bad["BSQ_score"][0])


def test_bsq_missing_components_excluded():
    """Tickers with no BSQ components are excluded from results."""
    rows = [
        {"ticker": "EMPTY"},  # no component keys
        {"ticker": "A", "cfo_margin": 0.15, "cash_quality": 1.0},
        {"ticker": "B", "cfo_margin": 0.05, "cash_quality": 0.0},
        {"ticker": "C", "cfo_margin": 0.10, "cash_quality": 0.5},
    ]
    result = BSQSignal().compute_batch(rows)
    assert "EMPTY" not in result["ticker"].to_list()
    assert result.shape[0] == 3


def test_bsq_returns_empty_on_too_few():
    """Fewer than 3 scored tickers returns empty DataFrame."""
    rows = [
        {"ticker": "A", "cfo_margin": 0.10},
        {"ticker": "B", "cfo_margin": 0.05},
    ]
    result = BSQSignal().compute_batch(rows)
    assert result.shape[0] == 0


def test_bsq_apply_filter_passes_unknown_tickers():
    """Tickers not in eligibility_df pass through unchanged."""
    rows = [
        {"ticker": "A", "accruals": 0.05, "cfo_margin": 0.20, "debt_burden": 0.10, "cash_quality": 1.0},
        {"ticker": "B", "accruals": -0.10,"cfo_margin":-0.05,"debt_burden":-0.30,"cash_quality": 0.0},
        {"ticker": "C", "accruals": 0.01, "cfo_margin": 0.12,"debt_burden": 0.05, "cash_quality": 1.0},
    ]
    elig = BSQSignal().compute_batch(rows)
    longs, shorts = BSQSignal().apply_filter(["A", "UNKNOWN_L"], ["B", "UNKNOWN_S"], elig)
    assert "UNKNOWN_L" in longs
    assert "UNKNOWN_S" in shorts


def test_bsq_short_eligibility_requires_negative_pead():
    """Short eligibility requires both low BSQ and negative PEAD_z."""
    rows = [
        {"ticker": "LOWBSQ1", "accruals": -0.10, "cfo_margin": -0.05, "debt_burden": -0.30, "cash_quality": 0.0},
        {"ticker": "LOWBSQ2", "accruals": -0.09, "cfo_margin": -0.04, "debt_burden": -0.25, "cash_quality": 0.0},
        {"ticker": "LOWBSQ3", "accruals": -0.08, "cfo_margin": -0.03, "debt_burden": -0.20, "cash_quality": 0.0},
        {"ticker": "HIGHBSQ", "accruals": 0.05,  "cfo_margin": 0.20,  "debt_burden": 0.10,  "cash_quality": 1.0},
        {"ticker": "HIGHBSQ2","accruals": 0.04,  "cfo_margin": 0.18,  "debt_burden": 0.09,  "cash_quality": 1.0},
    ]
    # LOWBSQ1 has negative PEAD → eligible to short
    # LOWBSQ2 has positive PEAD → NOT eligible to short (stock already re-rated up)
    pead_map = {
        "LOWBSQ1": -1.5,
        "LOWBSQ2": +1.0,
        "LOWBSQ3": -0.5,
        "HIGHBSQ": +0.8,
        "HIGHBSQ2": +0.7,
    }
    result = BSQSignal().compute_batch(rows, pead_z_map=pead_map)
    lowbsq1 = result.filter(pl.col("ticker") == "LOWBSQ1")
    lowbsq2 = result.filter(pl.col("ticker") == "LOWBSQ2")
    assert bool(lowbsq1["short_eligible"][0]) is True
    assert bool(lowbsq2["short_eligible"][0]) is False
