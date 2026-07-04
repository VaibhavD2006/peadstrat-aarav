"""
End-to-end smoke test for ARIA Phase 1 pipeline.
Uses entirely synthetic data — no network calls.
"""
import polars as pl
import numpy as np
import pytest
from datetime import date, timedelta
from aria.signals.esqs import ESQSSignal
from aria.signals.rmv import RMVSignal
from aria.portfolio.scorer import CompositeScorer
from aria.backtest.engine import BacktestEngine, BacktestConfig
from aria.backtest.performance import PerformanceAnalytics

def _make_price_df(tickers: list[str], start: date, n_days: int, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    # Generate calendar days, filter to weekdays
    all_dates = [start + timedelta(days=i) for i in range(n_days * 2)]
    biz_dates = [d for d in all_dates if d.weekday() < 5][:n_days]
    rows = []
    for ticker in tickers:
        returns = rng.normal(0.001, 0.015, len(biz_dates))
        prices  = 100.0 * np.cumprod(1 + returns)
        for d, p in zip(biz_dates, prices):
            rows.append({"date": d, "ticker": ticker, "open": float(p * 0.999),
                          "close": float(p), "adj_close": float(p), "adv_20d_usd": 5e9})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

def test_full_pipeline_smoke():
    rng = np.random.default_rng(42)
    tickers = [f"T{i:02d}" for i in range(20)]
    n = len(tickers)

    # Step 1: Compute ESQS for all tickers
    fundamentals = pl.DataFrame({
        "ticker":                tickers,
        "revenue_actual":        rng.uniform(90_000, 110_000, n).tolist(),
        "revenue_consensus":     [100_000.0] * n,
        "gross_margin_actual":   rng.uniform(0.35, 0.45, n).tolist(),
        "gross_margin_trailing": [0.40] * n,
        "sga_pct_actual":        rng.uniform(0.08, 0.12, n).tolist(),
        "sga_pct_trailing":      [0.10] * n,
        "guidance_score":        rng.choice([-1.0, 0.0, 1.0], n).tolist(),
    })
    esqs = ESQSSignal().compute_normalized(fundamentals)

    # Step 2: Attach synthetic RMV scores
    scored = esqs.with_columns(
        pl.Series("RMV_z", rng.normal(0, 1, n).tolist(), dtype=pl.Float64)
    )

    # Step 3: Score compositely and select long/short
    scorer = CompositeScorer(weights={"ESQS_z": 0.5, "RMV_z": 0.5})
    scored = scorer.score(scored)
    longs, shorts = scorer.select_long_short(scored, top_pct=0.20, bottom_pct=0.20)
    assert len(longs) >= 1 and len(shorts) >= 1

    # Step 4: Build signals DataFrame for backtester
    entry_date = date(2024, 1, 15)
    n_long, n_short = len(longs), len(shorts)
    signal_rows = (
        [{"ticker": t, "entry_date": entry_date, "side": "long",  "weight": 1.0/n_long}  for t in longs] +
        [{"ticker": t, "entry_date": entry_date, "side": "short", "weight": 1.0/n_short} for t in shorts]
    )
    signals = pl.DataFrame(signal_rows).with_columns(pl.col("entry_date").cast(pl.Date))

    # Step 5: Run backtest
    prices = _make_price_df(tickers, date(2024, 1, 2), n_days=60)
    config = BacktestConfig(hold_days=10, initial_capital=10_000_000)
    engine = BacktestEngine(config=config)
    results = engine.run(signals=signals, prices=prices)

    assert results.shape[0] > 0, "Engine produced no trade records"
    assert "pnl" in results.columns

    # Step 6: Compute performance metrics
    total_pnl = float(results["pnl"].sum())
    pa = PerformanceAnalytics()
    daily_returns = results.group_by("entry_date").agg(
        (pl.col("pnl").sum() / config.initial_capital).alias("ret")
    )["ret"].to_numpy()
    if len(daily_returns) > 1:
        summary = pa.summarize(daily_returns)
        assert "sharpe" in summary

def test_long_positions_double_short_positions_are_balanced():
    """Long book capital ≈ short book capital (dollar-neutral)."""
    rng = np.random.default_rng(7)
    tickers = [f"T{i:02d}" for i in range(10)]
    n = len(tickers)
    fundamentals = pl.DataFrame({
        "ticker": tickers,
        "revenue_actual": rng.uniform(90_000, 110_000, n).tolist(),
        "revenue_consensus": [100_000.0] * n,
        "gross_margin_actual": rng.uniform(0.35, 0.45, n).tolist(),
        "gross_margin_trailing": [0.40] * n,
        "sga_pct_actual": rng.uniform(0.08, 0.12, n).tolist(),
        "sga_pct_trailing": [0.10] * n,
        "guidance_score": [0.0] * n,
    })
    scored = CompositeScorer(weights={"ESQS_z": 1.0}).score(
        ESQSSignal().compute_normalized(fundamentals)
    )
    longs, shorts = CompositeScorer(weights={"ESQS_z": 1.0}).select_long_short(
        scored, top_pct=0.30, bottom_pct=0.30
    )
    # Dollar-neutral: equal number of names per side (given equal weights)
    assert len(longs) == len(shorts)
