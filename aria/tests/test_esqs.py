import polars as pl
import numpy as np
import pytest
from aria.signals.esqs import ESQSSignal

def test_positive_beat_scores_positive():
    sig = ESQSSignal()
    df = pl.DataFrame({
        "ticker":                ["AAPL"],
        "revenue_actual":        [120_000.0],
        "revenue_consensus":     [110_000.0],
        "gross_margin_actual":   [0.45],
        "gross_margin_trailing": [0.42],
        "sga_pct_actual":        [0.08],
        "sga_pct_trailing":      [0.10],
        "guidance_score":        [1.0],
    })
    result = sig.compute(df)
    assert result["ESQS"][0] > 0.0

def test_negative_miss_scores_negative():
    sig = ESQSSignal()
    df = pl.DataFrame({
        "ticker":                ["XYZ"],
        "revenue_actual":        [80_000.0],
        "revenue_consensus":     [100_000.0],
        "gross_margin_actual":   [0.30],
        "gross_margin_trailing": [0.38],
        "sga_pct_actual":        [0.15],
        "sga_pct_trailing":      [0.12],
        "guidance_score":        [-1.0],
    })
    result = sig.compute(df)
    assert result["ESQS"][0] < 0.0

def test_missing_guidance_treated_as_neutral():
    sig = ESQSSignal()
    df = pl.DataFrame({
        "ticker":                ["NOOG"],
        "revenue_actual":        [50_000.0],
        "revenue_consensus":     [48_000.0],
        "gross_margin_actual":   [0.40],
        "gross_margin_trailing": [0.40],
        "sga_pct_actual":        [0.10],
        "sga_pct_trailing":      [0.10],
        "guidance_score":        [None],
    })
    result = sig.compute(df)
    assert result["ESQS"][0] is not None
    assert not np.isnan(float(result["ESQS"][0]))

def test_normalized_scores_have_reasonable_distribution():
    rng = np.random.default_rng(42)
    n = 40
    sig = ESQSSignal()
    df = pl.DataFrame({
        "ticker":                [f"T{i}" for i in range(n)],
        "revenue_actual":        rng.uniform(90_000, 110_000, n).tolist(),
        "revenue_consensus":     [100_000.0] * n,
        "gross_margin_actual":   rng.uniform(0.35, 0.45, n).tolist(),
        "gross_margin_trailing": [0.40] * n,
        "sga_pct_actual":        rng.uniform(0.08, 0.12, n).tolist(),
        "sga_pct_trailing":      [0.10] * n,
        "guidance_score":        rng.choice([-1.0, 0.0, 1.0], n).tolist(),
    })
    result = sig.compute_normalized(df)
    scores = result["ESQS_z"].to_numpy()
    assert abs(scores.mean()) < 0.5
    assert 0.5 < scores.std() < 1.5
