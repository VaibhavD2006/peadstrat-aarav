import polars as pl
import numpy as np
import pytest
from aria.portfolio.scorer import CompositeScorer

def test_equal_weight_ordering():
    scorer = CompositeScorer(weights={"ESQS_z": 0.5, "RMV_z": 0.5})
    df = pl.DataFrame({
        "ticker": ["A", "B", "C"],
        "ESQS_z": [1.0, 0.0, -1.0],
        "RMV_z":  [1.0, 0.0, -1.0],
    })
    result = scorer.score(df)
    assert result["composite"][0] > result["composite"][1]
    assert result["composite"][1] > result["composite"][2]
    assert abs(result["composite"][1]) < 1e-10

def test_winsorization_clips_extreme():
    scorer = CompositeScorer(weights={"ESQS_z": 1.0}, winsorize_sigma=2.0)
    df = pl.DataFrame({
        "ticker": ["A", "EXTREME"],
        "ESQS_z": [1.0, 100.0],
    })
    result = scorer.score(df)
    extreme_val = result.filter(pl.col("ticker") == "EXTREME")["composite"][0]
    assert extreme_val <= 2.0

def test_long_short_selection_count():
    scorer = CompositeScorer(weights={"ESQS_z": 1.0})
    df = pl.DataFrame({
        "ticker": [f"T{i}" for i in range(20)],
        "ESQS_z": list(range(20, 0, -1)),
    })
    result = scorer.score(df)
    longs, shorts = scorer.select_long_short(result, top_pct=0.20, bottom_pct=0.20)
    assert len(longs) == 4
    assert len(shorts) == 4

def test_long_short_correct_names():
    scorer = CompositeScorer(weights={"ESQS_z": 1.0})
    df = pl.DataFrame({
        "ticker": [f"T{i:02d}" for i in range(20)],
        "ESQS_z": list(range(20, 0, -1)),
    })
    result = scorer.score(df)
    longs, shorts = scorer.select_long_short(result, top_pct=0.20, bottom_pct=0.20)
    # T00 has highest score (20), T01=19, T02=18, T03=17 → top 4
    assert set(longs) == {"T00", "T01", "T02", "T03"}

def test_missing_signal_treated_as_zero():
    scorer = CompositeScorer(weights={"ESQS_z": 0.5, "RMV_z": 0.5})
    df = pl.DataFrame({
        "ticker": ["A"],
        "ESQS_z": [2.0],
        # RMV_z column missing entirely
    })
    result = scorer.score(df)
    # Only ESQS contributes: 0.5 * 2.0 = 1.0 (before winsorize)
    assert abs(result["composite"][0] - 1.0) < 0.01
