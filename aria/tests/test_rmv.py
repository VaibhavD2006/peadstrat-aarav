import polars as pl
import numpy as np
import pytest
from datetime import date, timedelta
from aria.signals.rmv import RMVSignal

def _make_revision_df(ticker: str, b1_val: float, b2_val: float, b3_val: float,
                       n_analysts: int = 5) -> pl.DataFrame:
    """Create revision rows with given mean values for 3 time buckets."""
    t0 = date(2024, 1, 1)
    rows = []
    for analyst_idx in range(n_analysts):
        aid = f"analyst_{analyst_idx}"
        for bucket_idx, (days_ago, val) in enumerate([
            (75, b1_val),   # bucket 1: 90-60 days ago
            (45, b2_val),   # bucket 2: 60-30 days ago
            (15, b3_val),   # bucket 3: 30-0 days ago
        ]):
            rows.append({
                "ticker": ticker,
                "analyst_id": aid,
                "revision_date": t0 - timedelta(days=days_ago - analyst_idx),
                "revision_pct": val + np.random.default_rng(analyst_idx + bucket_idx).normal(0, 0.001),
                "earnings_date": t0,
            })
    return pl.DataFrame(rows)

def test_rmv_accelerating_upward_is_positive():
    sig = RMVSignal()
    revisions = _make_revision_df("BULL", b1_val=-0.01, b2_val=0.02, b3_val=0.05)
    result = sig.compute(revisions, earnings_date=date(2024, 1, 1))
    assert result["RMV"] > 0.0

def test_rmv_accelerating_more_than_decelerating():
    sig = RMVSignal()
    r_acc = _make_revision_df("ACC", b1_val=-0.01, b2_val=0.02, b3_val=0.05)
    r_dec = _make_revision_df("DEC", b1_val=0.05,  b2_val=0.02, b3_val=-0.01)
    result_acc = sig.compute(r_acc, date(2024, 1, 1))
    result_dec = sig.compute(r_dec, date(2024, 1, 1))
    assert result_acc["RMV"] > result_dec["RMV"]

def test_rmv_nan_when_insufficient_analysts():
    sig = RMVSignal(min_analysts=5)
    revisions = _make_revision_df("FEW", 0.01, 0.02, 0.03, n_analysts=3)
    result = sig.compute(revisions, date(2024, 1, 1))
    assert np.isnan(result["RMV"])

def test_rmv_batch_returns_dataframe():
    sig = RMVSignal()
    r1 = _make_revision_df("AAPL", 0.01, 0.02, 0.03)
    r2 = _make_revision_df("MSFT", -0.01, -0.02, -0.03)
    revisions = pl.concat([r1, r2])
    result = sig.compute_batch(revisions, [("AAPL", date(2024, 1, 1)), ("MSFT", date(2024, 1, 1))])
    assert result.shape[0] == 2
    assert "RMV" in result.columns
