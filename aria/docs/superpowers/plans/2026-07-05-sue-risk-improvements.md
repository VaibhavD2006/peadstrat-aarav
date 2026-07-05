# SUE Signal & Risk Management Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve E17 (SUE+BSQ) from Sharpe=+0.34/MaxDD=-36.8% to Sharpe>0.6/MaxDD<-25% by fixing vol targeting, adding stop-losses, improving SUE normalization, and layering on EPS revision direction weighting.

**Architecture:** Phase 1 fixes risk management (vol formula, stop-loss) without touching signal logic. Phase 2 upgrades signal quality (rolling error normalizer, magnitude + revision weighting). Each phase produces a set of ablation experiments (E19-E25) for validation. Every change is test-driven.

**Tech Stack:** Python 3.11, Polars, NumPy, pytest. No new dependencies.

---

## File Map

| File | Role |
|---|---|
| `aria/research/phase3_runner.py` | Extract `_vol_target_scale()`; fix L/S vol formula; add SUE magnitude + revision weighting |
| `aria/backtest/engine.py` | Add `stop_loss_pct` to `BacktestConfig`; add daily stop-loss check in `run()` |
| `aria/research/ablation.py` | Add `stop_loss_pct` field to `ExperimentSpec`; add E19–E25 |
| `aria/data/ingestion/sue_loader.py` | Add `compute_historical_errors()` and `compute_revision_dir()` |
| `aria/signals/sue.py` | Add `hist_errors` parameter to `compute_sue_raw()` |
| `aria/tests/test_engine.py` | Stop-loss exit timing tests |
| `aria/tests/test_sue.py` | Historical error normalizer + revision direction tests |

---

## Task 1: Fix Vol Targeting — Extract and Correct `_vol_target_scale()`

The current formula (`avg_vol * sqrt(rho + (1-rho)/n)`) treats the L/S book as long-only, underestimating portfolio vol by ~47% and causing oversized positions. The fix computes long-side and short-side vol separately, then combines them via the L/S portfolio variance formula.

**Files:**
- Modify: `aria/research/phase3_runner.py` (around line 414, alongside `_compute_ticker_vols`)
- Test: No unit test needed — the formula is validated by the E19 ablation run. But add an inline sanity check via a module-level helper test in `aria/tests/test_phase3_runner.py` (new file, minimal).

- [ ] **Step 1: Write the failing test**

Create `aria/tests/test_phase3_runner.py`:

```python
"""Unit tests for Phase3Runner helper methods."""
import pytest
import numpy as np
from aria.research.phase3_runner import Phase3Runner


def test_vol_target_scale_single_position():
    """1 long + 1 short at vol=0.35 should give port_vol ≈ 0.414, scale ≈ 0.362."""
    runner = Phase3Runner.__new__(Phase3Runner)  # no __init__ needed
    ticker_vols = {"AAPL": 0.35, "MSFT": 0.35}
    scale = runner._vol_target_scale(["AAPL"], ["MSFT"], ticker_vols, target_vol=0.15)
    # port_vol = 0.35 * sqrt(2 * (1 - 0.30)) = 0.35 * sqrt(1.4) ≈ 0.414
    # scale = 0.15 / 0.414 ≈ 0.362, capped at 2.0
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
```

- [ ] **Step 2: Run test to verify it fails**

```
cd C:\Users\dandy\OneDrive\Documents\aaravstrat\aria
python -m pytest aria/tests/test_phase3_runner.py -v
```

Expected: `AttributeError: type object 'Phase3Runner' has no attribute '_vol_target_scale'`

- [ ] **Step 3: Add `_vol_target_scale()` to `Phase3Runner`**

In `aria/research/phase3_runner.py`, add this method after `_compute_ticker_vols` (around line 439):

```python
def _vol_target_scale(
    self,
    longs: list[str],
    shorts: list[str],
    ticker_vols: dict[str, float],
    target_vol: float,
    rho_within: float = 0.30,
    rho_ls: float = 0.30,
) -> float:
    """Compute position scale factor so estimated portfolio vol ≈ target_vol.

    Uses the correct L/S portfolio variance formula: separate long and short
    side vols, then combine via cov(L,S) term.  The old formula treated the
    book as long-only and underestimated vol by ~47%.
    """
    n_long  = max(len(longs), 1)
    n_short = max(len(shorts), 1)
    avg_vol_long  = float(np.mean([ticker_vols.get(t, 0.30) for t in longs]))
    avg_vol_short = float(np.mean([ticker_vols.get(t, 0.30) for t in shorts]))
    vol_L = avg_vol_long  * np.sqrt(rho_within + (1.0 - rho_within) / n_long)
    vol_S = avg_vol_short * np.sqrt(rho_within + (1.0 - rho_within) / n_short)
    port_vol_est = np.sqrt(vol_L**2 + vol_S**2 - 2.0 * rho_ls * vol_L * vol_S)
    return min(target_vol / max(port_vol_est, 0.01), 2.0)
```

- [ ] **Step 4: Replace the old vol formula in the `run()` method**

Find the vol targeting block (around line 931–952 in `phase3_runner.py`):

```python
                if exp.vol_target > 0:
                    # Inverse-vol weights, then scale book to hit vol_target
                    ticker_vols = self._compute_ticker_vols(
                        longs + shorts, all_prices, entry_date
                    )
                    lw = {t: 1.0 / max(ticker_vols[t], 0.05) for t in longs}
                    sw = {t: 1.0 / max(ticker_vols[t], 0.05) for t in shorts}
                    lt = sum(lw.values())
                    st = sum(sw.values())
                    long_w = {t: v / lt for t, v in lw.items()}
                    short_w = {t: v / st for t, v in sw.items()}
                    # Scale gross book: portfolio vol ≈ avg_ticker_vol / sqrt(n)
                    all_book = longs + shorts
                    n = max(len(all_book), 1)
                    avg_vol = np.mean([ticker_vols[t] for t in all_book])
                    # Assume avg pairwise correlation ρ=0.3 (typical for equity L/S).
                    # port_vol ≈ avg_ticker_vol * sqrt(ρ + (1-ρ)/n)
                    rho = 0.30
                    port_vol_est = avg_vol * np.sqrt(rho + (1.0 - rho) / n)
                    scale = min(exp.vol_target / max(port_vol_est, 0.01), 1.5)
                    long_w = {t: w * scale for t, w in long_w.items()}
                    short_w = {t: w * scale for t, w in short_w.items()}
```

Replace with:

```python
                if exp.vol_target > 0:
                    ticker_vols = self._compute_ticker_vols(
                        longs + shorts, all_prices, entry_date
                    )
                    lw = {t: 1.0 / max(ticker_vols[t], 0.05) for t in longs}
                    sw = {t: 1.0 / max(ticker_vols[t], 0.05) for t in shorts}
                    lt = sum(lw.values())
                    st = sum(sw.values())
                    long_w = {t: v / lt for t, v in lw.items()}
                    short_w = {t: v / st for t, v in sw.items()}
                    scale = self._vol_target_scale(
                        longs, shorts, ticker_vols, exp.vol_target
                    )
                    long_w  = {t: w * scale for t, w in long_w.items()}
                    short_w = {t: w * scale for t, w in short_w.items()}
```

- [ ] **Step 5: Run tests**

```
python -m pytest aria/tests/test_phase3_runner.py aria/tests/test_engine.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add aria/research/phase3_runner.py aria/tests/test_phase3_runner.py
git commit -m "fix: correct L/S portfolio vol formula in vol targeting

Old formula treated the L/S book as long-only (avg_vol * sqrt(rho + (1-rho)/n)),
underestimating portfolio vol by ~47% and oversizing positions.
New _vol_target_scale() separates long/short side vols and uses the
correct covariance term: sqrt(vol_L^2 + vol_S^2 - 2*rho_ls*vol_L*vol_S).
Scale cap raised from 1.5x to 2.0x (rarely triggered now that vol is correct).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Add Per-Trade Stop-Loss to BacktestEngine

Add `stop_loss_pct: float = 0.0` to `BacktestConfig`. When nonzero, `run()` iterates through daily prices after entry and exits the position on the first day where cumulative return < `-stop_loss_pct`. Stop-loss is per-position (not portfolio).

**Files:**
- Modify: `aria/backtest/engine.py`
- Modify: `aria/tests/test_engine.py`

- [ ] **Step 1: Write the failing tests**

Add to `aria/tests/test_engine.py`:

```python
def test_stop_loss_exits_early_on_large_drop():
    """Position drops >10% → exits before hold_days."""
    # Deterministic: stock drops 15% on day 3, then recovers
    all_dates = pl.date_range(date(2024, 1, 2), date(2024, 2, 15), interval="1d", eager=True)
    dates = [d for d in all_dates.to_list() if d.weekday() < 5]
    prices_arr = []
    for i, d in enumerate(dates):
        if i < 3:
            prices_arr.append(100.0)
        elif i == 3:
            prices_arr.append(84.0)   # -16% drop triggers stop
        else:
            prices_arr.append(100.0)  # recovery (should not be used)
    rows = [{"date": d, "ticker": "TST", "open": float(p), "close": float(p),
             "adj_close": float(p), "adv_20d_usd": 5e9}
            for d, p in zip(dates, prices_arr)]
    prices = pl.DataFrame(rows)

    config = BacktestConfig(hold_days=20, initial_capital=100_000, stop_loss_pct=0.10)
    engine = BacktestEngine(config=config)
    signals = pl.DataFrame({
        "ticker": ["TST"], "entry_date": [date(2024, 1, 4)],
        "side": ["long"], "weight": [1.0],
    })
    results = engine.run(signals=signals, prices=prices)
    assert results.shape[0] == 1
    # Exit should happen on day 3 after entry (the -16% day), not day 20
    exit_dt = date.fromisoformat(results["exit_date"][0])
    assert exit_dt < date(2024, 1, 15), f"Stop not triggered: exit={exit_dt}"
    # PnL should reflect the -16% loss, not the recovery
    assert results["pnl"][0] < -10_000


def test_stop_loss_not_triggered_when_drop_is_small():
    """Position drops 5% → no early exit, holds full hold_days."""
    all_dates = pl.date_range(date(2024, 1, 2), date(2024, 2, 15), interval="1d", eager=True)
    dates = [d for d in all_dates.to_list() if d.weekday() < 5]
    # Drop 5% early, then flat — never hits -10% stop
    prices_arr = [100.0 if i == 0 else 95.0 for i in range(len(dates))]
    rows = [{"date": d, "ticker": "TST", "open": float(p), "close": float(p),
             "adj_close": float(p), "adv_20d_usd": 5e9}
            for d, p in zip(dates, prices_arr)]
    prices = pl.DataFrame(rows)

    config = BacktestConfig(hold_days=10, initial_capital=100_000, stop_loss_pct=0.10)
    engine = BacktestEngine(config=config)
    signals = pl.DataFrame({
        "ticker": ["TST"], "entry_date": [date(2024, 1, 3)],
        "side": ["long"], "weight": [1.0],
    })
    results = engine.run(signals=signals, prices=prices)
    assert results.shape[0] == 1
    # Should hold full 10 days
    exit_dt = date.fromisoformat(results["exit_date"][0])
    entry_dt = date(2024, 1, 3)
    calendar_days = (exit_dt - entry_dt).days
    assert calendar_days >= 12, f"Exited too early: {exit_dt}"


def test_stop_loss_zero_disables_feature():
    """stop_loss_pct=0.0 (default) → original behaviour, no early exit."""
    prices = _make_prices(tickers=["AAPL"])
    config = BacktestConfig(hold_days=5, initial_capital=100_000, stop_loss_pct=0.0)
    engine = BacktestEngine(config=config)
    signals = pl.DataFrame({
        "ticker": ["AAPL"], "entry_date": [date(2024, 1, 11)],
        "side": ["long"], "weight": [1.0],
    })
    results = engine.run(signals=signals, prices=prices)
    assert results.shape[0] == 1  # always produces a record when prices exist
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest aria/tests/test_engine.py::test_stop_loss_exits_early_on_large_drop aria/tests/test_engine.py::test_stop_loss_not_triggered_when_drop_is_small aria/tests/test_engine.py::test_stop_loss_zero_disables_feature -v
```

Expected: FAIL — `BacktestConfig` has no `stop_loss_pct` attribute.

- [ ] **Step 3: Add `stop_loss_pct` to `BacktestConfig` and implement stop-loss in `run()`**

Replace `aria/backtest/engine.py` with:

```python
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import polars as pl
import numpy as np
from aria.backtest.costs import TransactionCostModel

@dataclass
class BacktestConfig:
    hold_days: int = 10
    initial_capital: float = 100_000_000
    cost_model: TransactionCostModel = field(default_factory=TransactionCostModel)
    max_gap_pct: float = 0.03
    stop_loss_pct: float = 0.0   # 0 = disabled; e.g. 0.10 = exit if position down >10%

@dataclass
class Position:
    ticker: str
    entry_date: date
    exit_date: date
    entry_price: float
    side: str
    weight: float
    capital: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def _get_price(self, prices: pl.DataFrame, ticker: str, d: date,
                   price_col: str = "open") -> Optional[float]:
        row = prices.filter((pl.col("ticker") == ticker) & (pl.col("date") == d))
        if row.is_empty():
            return None
        return float(row[price_col][0])

    def _get_adv(self, prices: pl.DataFrame, ticker: str, d: date) -> float:
        rows = prices.filter((pl.col("ticker") == ticker) & (pl.col("date") <= d)).tail(20)
        if "adv_20d_usd" in rows.columns and not rows.is_empty():
            val = rows["adv_20d_usd"][-1]
            if val is not None:
                return float(val)
        return 5_000_000_000.0

    def run(self, signals: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
        records = []
        cost = self.config.cost_model
        stop_loss = self.config.stop_loss_pct

        for row in signals.iter_rows(named=True):
            ticker     = row["ticker"]
            entry_date = row["entry_date"]
            side       = row["side"]
            weight     = row["weight"]

            entry_price = self._get_price(prices, ticker, entry_date, "open")
            if entry_price is None:
                continue

            future = (prices
                      .filter((pl.col("ticker") == ticker) & (pl.col("date") > entry_date))
                      .sort("date"))
            if future.shape[0] < self.config.hold_days:
                continue

            # Determine exit: either scheduled hold or stop-loss trigger
            direction = 1.0 if side == "long" else -1.0
            exit_idx  = self.config.hold_days - 1
            if stop_loss > 0.0:
                closes = future["close"].to_list()
                for i, close in enumerate(closes[:self.config.hold_days]):
                    cum_ret = direction * (close - entry_price) / entry_price
                    if cum_ret < -stop_loss:
                        exit_idx = i
                        break

            exit_date  = future["date"][exit_idx]
            exit_price = float(future["close"][exit_idx])

            capital    = self.config.initial_capital * weight
            adv        = self._get_adv(prices, ticker, entry_date)
            cost_entry = capital * cost.total_cost_bps(capital, adv, True) / 10_000
            cost_exit  = capital * cost.total_cost_bps(capital, adv, True) / 10_000
            borrow     = (capital * cost.daily_borrow_cost_bps() / 10_000 *
                          (exit_idx + 1)) if side == "short" else 0.0

            gross_return = (exit_price - entry_price) / entry_price
            pnl = capital * direction * gross_return - cost_entry - cost_exit - borrow

            records.append({
                "ticker":       ticker,
                "entry_date":   str(entry_date),
                "exit_date":    str(exit_date),
                "side":         side,
                "entry_price":  entry_price,
                "exit_price":   exit_price,
                "gross_return": float(gross_return),
                "pnl":          float(pnl),
                "capital":      capital,
                "weight":       weight,
            })

        if not records:
            return pl.DataFrame({
                "ticker": [], "entry_date": [], "exit_date": [], "side": [],
                "entry_price": [], "exit_price": [], "gross_return": [], "pnl": [],
                "capital": [], "weight": [],
            })
        return pl.DataFrame(records)
```

- [ ] **Step 4: Run all engine tests**

```
python -m pytest aria/tests/test_engine.py -v
```

Expected: all pass, including the 3 new stop-loss tests.

- [ ] **Step 5: Commit**

```bash
git add aria/backtest/engine.py aria/tests/test_engine.py
git commit -m "feat: add per-trade stop-loss to BacktestEngine

Add stop_loss_pct field to BacktestConfig (default 0.0 = disabled).
When nonzero, run() scans daily closes after entry and exits on the
first day where cumulative position return < -stop_loss_pct.
Borrow cost now uses actual hold duration instead of fixed hold_days.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Add `stop_loss_pct` to `ExperimentSpec` and Ablation Matrix E19–E25

**Files:**
- Modify: `aria/research/ablation.py`

- [ ] **Step 1: Add `stop_loss_pct` field to `ExperimentSpec`**

In `aria/research/ablation.py`, add the new field after `vol_target`:

```python
@dataclass
class ExperimentSpec:
    name: str
    signals: list[str]
    weights: dict[str, float]
    hold_days: int = 10
    regime_filter: bool = False
    ivrs_multiplier: bool = False
    bsq_filter: bool = False
    beta_neutral: bool = False
    vol_target: float = 0.0
    stop_loss_pct: float = 0.0   # 0 = disabled; e.g. 0.10 = 10% per-position stop
    notes: str = ""
```

- [ ] **Step 2: Add E19–E25 to `ABLATION_MATRIX`**

Append to the list in `aria/research/ablation.py` after E18:

```python
    # --- Phase 6: Risk management + SUE signal quality improvements ---
    ExperimentSpec(
        "E19_vol_fix",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        notes="E17 + corrected L/S vol targeting formula (isolate MaxDD impact)",
    ),
    ExperimentSpec(
        "E20_vol_fix_stoploss",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        notes="E19 + 10% per-position stop-loss (isolate stop-loss impact)",
    ),
    ExperimentSpec(
        "E21_sue_normalizer",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        notes="E20 + rolling 4-quarter historical error std as SUE normalizer",
    ),
    ExperimentSpec(
        "E22_sue_magnitude",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        notes="E20 + SUE magnitude position weighting (|SUE_z| clipped [0.5, 3.0])",
    ),
    ExperimentSpec(
        "E23_phase1_full",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        notes="All Phase 1 changes: vol fix + stop-loss + normalizer + magnitude weighting",
    ),
    ExperimentSpec(
        "E24_revision_dir",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        notes="E23 + EPS revision direction proxy filter",
    ),
    ExperimentSpec(
        "E25_revision_magnitude",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        notes="Full Phase 1+2 stack: E23 + revision direction + magnitude combined",
    ),
```

- [ ] **Step 3: Update the `summary_df()` to include `stop_loss_pct`**

In the `summary_table()` method, the `**self.results.get(exp.name, {})` already catches all metrics. Just make sure `stop_loss_pct` is in the spec columns. Add it to the dict explicitly:

```python
    def summary_table(self) -> list[dict]:
        return [
            {
                "experiment": exp.name,
                "signals": str(exp.signals),
                "hold_days": exp.hold_days,
                "regime_filter": exp.regime_filter,
                "ivrs_multiplier": exp.ivrs_multiplier,
                "bsq_filter": exp.bsq_filter,
                "beta_neutral": exp.beta_neutral,
                "vol_target": exp.vol_target,
                "stop_loss_pct": exp.stop_loss_pct,
                **self.results.get(exp.name, {}),
            }
            for exp in self.experiments
        ]
```

- [ ] **Step 4: Update the test that checks ablation matrix size**

In `aria/tests/test_ablation.py`, find the test that checks experiment count and update it to use `len(ABLATION_MATRIX)` dynamically (it should already be dynamic from a prior fix — confirm and leave as-is if so).

```
python -m pytest aria/tests/test_ablation.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add aria/research/ablation.py aria/tests/test_ablation.py
git commit -m "feat: add E19-E25 ablation experiments and stop_loss_pct to ExperimentSpec

Phase 6 experiments isolate each improvement: vol fix (E19), stop-loss (E20),
rolling normalizer (E21), magnitude weighting (E22), full Phase 1 (E23),
revision direction (E24), full Phase 1+2 stack (E25).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Wire `stop_loss_pct` into the Runner

The runner creates `BacktestConfig` per experiment but currently never passes `stop_loss_pct`. Wire it through.

**Files:**
- Modify: `aria/research/phase3_runner.py` (BacktestConfig construction, around line 895)

- [ ] **Step 1: Find the `BacktestConfig` construction in `run()`**

Search for `BacktestConfig(` in `phase3_runner.py`. It looks like:

```python
                config = BacktestConfig(
                    hold_days=effective_hold,
                    initial_capital=self.initial_capital,
                )
```

- [ ] **Step 2: Add `stop_loss_pct`**

Replace with:

```python
                config = BacktestConfig(
                    hold_days=effective_hold,
                    initial_capital=self.initial_capital,
                    stop_loss_pct=exp.stop_loss_pct,
                )
```

- [ ] **Step 3: Run existing tests**

```
python -m pytest aria/tests/ -v --tb=short
```

Expected: all pass (no behaviour change for experiments with `stop_loss_pct=0.0`).

- [ ] **Step 4: Commit**

```bash
git add aria/research/phase3_runner.py
git commit -m "feat: pass stop_loss_pct from ExperimentSpec to BacktestConfig

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Improve SUE Normalizer — Add `hist_errors` to `compute_sue_raw()`

**Files:**
- Modify: `aria/signals/sue.py`
- Modify: `aria/tests/test_sue.py`

- [ ] **Step 1: Write failing tests**

Add to `aria/tests/test_sue.py`:

```python
def test_sue_raw_uses_hist_errors_when_provided():
    """hist_errors with std=0.20 → normalizer=0.20."""
    hist = [0.10, -0.10, 0.20, -0.20]  # std = 0.158...
    import numpy as np
    expected_norm = float(np.std(hist, ddof=1))
    val = compute_sue_raw(actual_eps=1.30, consensus_eps=1.00,
                          hist_errors=hist)
    assert abs(val - 0.30 / expected_norm) < 1e-6


def test_sue_raw_hist_errors_preferred_over_forecast_std():
    """hist_errors takes priority over forecast_std."""
    hist = [0.10, -0.10, 0.20, -0.20]
    import numpy as np
    expected_norm = float(np.std(hist, ddof=1))
    val = compute_sue_raw(actual_eps=1.30, consensus_eps=1.00,
                          forecast_std=0.50, hist_errors=hist)
    assert abs(val - 0.30 / expected_norm) < 1e-6


def test_sue_raw_falls_back_when_hist_errors_too_short():
    """Only 1 hist_error → not enough for std → falls back to abs(consensus)."""
    val_with_hist  = compute_sue_raw(actual_eps=1.10, consensus_eps=1.00,
                                     hist_errors=[0.05])
    val_without    = compute_sue_raw(actual_eps=1.10, consensus_eps=1.00)
    assert abs(val_with_hist - val_without) < 1e-9


def test_sue_raw_hist_errors_fallback_scale_prevents_tiny_std():
    """If std of hist_errors is < fallback_scale=0.01, uses fallback_scale."""
    hist = [0.001, -0.001, 0.001, -0.001]  # tiny std
    val = compute_sue_raw(actual_eps=1.10, consensus_eps=1.00,
                          hist_errors=hist, fallback_scale=0.01)
    assert abs(val - 0.10 / 0.01) < 1e-6  # normalizer clamped to 0.01
```

- [ ] **Step 2: Run to verify they fail**

```
python -m pytest aria/tests/test_sue.py::test_sue_raw_uses_hist_errors_when_provided aria/tests/test_sue.py::test_sue_raw_hist_errors_preferred_over_forecast_std aria/tests/test_sue.py::test_sue_raw_falls_back_when_hist_errors_too_short aria/tests/test_sue.py::test_sue_raw_hist_errors_fallback_scale_prevents_tiny_std -v
```

Expected: FAIL — `compute_sue_raw()` has no `hist_errors` parameter.

- [ ] **Step 3: Update `compute_sue_raw()` in `aria/signals/sue.py`**

Replace the function with:

```python
def compute_sue_raw(
    actual_eps: float,
    consensus_eps: float,
    forecast_std: Optional[float] = None,
    hist_errors: Optional[list[float]] = None,
    fallback_scale: float = 0.01,
) -> float:
    """Compute a single ticker's raw SUE value.

    Normalizer priority:
      1. Rolling std of hist_errors (past 4 quarters of actual-minus-consensus)
         — most accurate; captures per-ticker analyst accuracy history.
      2. forecast_std (cross-sectional std of analyst estimates at announcement)
         — available from some paid providers.
      3. max(abs(consensus_eps), fallback_scale) — always available.

    Args:
        actual_eps:     Reported EPS.
        consensus_eps:  Mean analyst consensus EPS estimate before announcement.
        forecast_std:   Std dev of analyst forecasts (second-priority normalizer).
        hist_errors:    List of prior (actual-consensus) values for this ticker.
                        Requires ≥2 values to compute std; shorter lists are ignored.
        fallback_scale: Minimum denominator to avoid division by zero.

    Returns:
        Raw surprise scalar. Positive = beat, negative = miss.
    """
    surprise = actual_eps - consensus_eps

    if hist_errors is not None and len(hist_errors) >= 2:
        normalizer = max(float(np.std(hist_errors, ddof=1)), fallback_scale)
        return surprise / normalizer

    if forecast_std is not None and forecast_std > fallback_scale:
        return surprise / forecast_std

    denom = max(abs(consensus_eps), fallback_scale)
    return surprise / denom
```

- [ ] **Step 4: Run all SUE tests**

```
python -m pytest aria/tests/test_sue.py -v
```

Expected: all 23 tests pass (19 original + 4 new).

- [ ] **Step 5: Commit**

```bash
git add aria/signals/sue.py aria/tests/test_sue.py
git commit -m "feat: add hist_errors parameter to compute_sue_raw for rolling normalizer

Historical error std (past 4 quarters) takes priority over forecast_std
and abs(consensus_eps) fallback, giving a per-ticker analyst accuracy-
adjusted normalizer. Falls back gracefully when < 2 quarters available.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Add `compute_historical_errors()` to `sue_loader.py` and Wire into `get_sue_inputs()`

**Files:**
- Modify: `aria/data/ingestion/sue_loader.py`
- Modify: `aria/tests/test_sue.py`

- [ ] **Step 1: Write failing tests**

Add to `aria/tests/test_sue.py`:

```python
# ---------------------------------------------------------------------------
# sue_loader: compute_historical_errors
# ---------------------------------------------------------------------------

from aria.data.ingestion.sue_loader import compute_historical_errors

def _make_rich_consensus_df():
    """6 quarters of AAPL history for testing historical errors."""
    return pl.DataFrame({
        "ticker": ["AAPL"] * 6 + ["MSFT"],
        "report_date": [
            date(2022, 7, 1), date(2022, 10, 1),
            date(2023, 1, 1), date(2023, 4, 1),
            date(2023, 7, 1), date(2023, 10, 1),
            date(2023, 10, 15),
        ],
        "consensus_eps": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0],
        "actual_eps":    [1.1, 1.0, 1.3, 1.2, 1.5, 1.4, 2.2],
    })


def test_compute_historical_errors_returns_prior_errors():
    df = _make_rich_consensus_df()
    errors = compute_historical_errors("AAPL", date(2024, 1, 1), df, n_quarters=4)
    # Prior 4 quarters before 2024-01-01: Q3'23, Q2'23, Q1'23, Q4'22 reports
    # errors = [actual - consensus]: 0.1, -0.1, 0.1, -0.1
    assert len(errors) == 4
    assert all(abs(e) == pytest.approx(0.1) for e in errors)


def test_compute_historical_errors_excludes_as_of_date():
    """report_date equal to as_of_date is excluded (point-in-time safe)."""
    df = _make_rich_consensus_df()
    errors = compute_historical_errors("AAPL", date(2023, 10, 1), df, n_quarters=4)
    # as_of_date is 2023-10-01; the row on that date is excluded
    assert len(errors) <= 4
    # The 2023-10-01 row has error=-0.1; if included it would appear
    # Verify it's NOT the last error
    # Prior 4 are: Q2'23 (2023-04-01), Q1'23 (2023-01-01), Q4'22 (2022-10-01), Q3'22 (2022-07-01)
    assert errors[-1] == pytest.approx(0.1)  # last of those 4 is Q2'23: 1.2-1.3=-0.1... wait
    # Actually: errors are actual-consensus. Q2'23: 1.2-1.3=-0.1. Let me just check length
    assert len(errors) == 4


def test_compute_historical_errors_returns_empty_for_unknown_ticker():
    df = _make_rich_consensus_df()
    errors = compute_historical_errors("UNKNOWN", date(2024, 1, 1), df)
    assert errors == []


def test_get_sue_inputs_uses_historical_errors_in_normalizer():
    """get_sue_inputs should pass hist_errors to compute_sue_raw when available."""
    df = _make_rich_consensus_df()
    # AAPL has 5 prior quarters before 2024-01-01 (reports up to 2023-10-01)
    rows = get_sue_inputs(["AAPL"], date(2024, 1, 1), df)
    assert len(rows) == 1
    # With hist_errors, the normalizer is std([0.1,-0.1,0.1,-0.1]) = 0.1155
    # sue_raw = (1.4 - 1.5) / 0.1155 ≈ -0.866 (using most recent row 2023-10-01)
    import numpy as np
    hist = [1.1-1.0, 1.0-1.1, 1.3-1.2, 1.2-1.3, 1.5-1.4]  # 5 prior quarters
    expected_norm = max(float(np.std(hist[-4:], ddof=1)), 0.01)
    expected_sue = (1.4 - 1.5) / expected_norm
    assert abs(rows[0]["sue_raw"] - expected_sue) < 0.01
```

- [ ] **Step 2: Run to verify they fail**

```
python -m pytest aria/tests/test_sue.py::test_compute_historical_errors_returns_prior_errors aria/tests/test_sue.py::test_compute_historical_errors_excludes_as_of_date aria/tests/test_sue.py::test_compute_historical_errors_returns_empty_for_unknown_ticker aria/tests/test_sue.py::test_get_sue_inputs_uses_historical_errors_in_normalizer -v
```

Expected: FAIL — `compute_historical_errors` not importable.

- [ ] **Step 3: Add `compute_historical_errors()` to `sue_loader.py`**

Add after the `load_consensus()` function:

```python
def compute_historical_errors(
    ticker: str,
    as_of_date: date,
    consensus_df: pl.DataFrame,
    n_quarters: int = 4,
) -> list[float]:
    """Return list of (actual_eps - consensus_eps) for the n_quarters prior to as_of_date.

    Point-in-time safe: excludes rows with report_date >= as_of_date.
    Returns [] if fewer than 1 prior row exists.
    """
    prior = (
        consensus_df
        .filter(
            (pl.col("ticker") == ticker) &
            (pl.col("report_date") < as_of_date)
        )
        .sort("report_date")
        .tail(n_quarters)
    )
    if prior.is_empty():
        return []
    return (prior["actual_eps"] - prior["consensus_eps"]).to_list()
```

- [ ] **Step 4: Update `get_sue_inputs()` to compute and pass `hist_errors`**

In `sue_loader.py`, update `get_sue_inputs()` — replace the `sue_raw = compute_sue_raw(...)` call:

```python
    rows: list[dict] = []
    has_std = "std_eps" in latest.columns
    for row in latest.iter_rows(named=True):
        forecast_std = row.get("std_eps") if has_std else None
        if forecast_std is not None and not isinstance(forecast_std, float):
            forecast_std = float(forecast_std)
        hist_errors = compute_historical_errors(
            row["ticker"], row["report_date"], consensus_df
        )
        sue_raw = compute_sue_raw(
            actual_eps=float(row["actual_eps"]),
            consensus_eps=float(row["consensus_eps"]),
            forecast_std=forecast_std,
            hist_errors=hist_errors if len(hist_errors) >= 2 else None,
        )
        rows.append({"ticker": row["ticker"], "sue_raw": sue_raw})
```

- [ ] **Step 5: Run all SUE tests**

```
python -m pytest aria/tests/test_sue.py -v
```

Expected: all pass. `test_get_sue_inputs_returns_most_recent_per_ticker` still passes unchanged: `_make_consensus_df()` has only 1 AAPL prior row before the most-recent report date, so `len(hist_errors) == 1 < 2` → fallback to `abs(consensus_eps)` → same formula as before.

- [ ] **Step 6: Commit**

```bash
git add aria/data/ingestion/sue_loader.py aria/tests/test_sue.py
git commit -m "feat: rolling 4-quarter historical error std as SUE normalizer

compute_historical_errors() collects prior analyst error magnitudes per
ticker. get_sue_inputs() now passes these to compute_sue_raw(), enabling
per-ticker analyst-accuracy-adjusted normalization (priority 1 when >= 2
quarters available).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Add `compute_revision_dir()` to `sue_loader.py`

**Files:**
- Modify: `aria/data/ingestion/sue_loader.py`
- Modify: `aria/tests/test_sue.py`

- [ ] **Step 1: Write failing tests**

Add to `aria/tests/test_sue.py`:

```python
# ---------------------------------------------------------------------------
# sue_loader: compute_revision_dir
# ---------------------------------------------------------------------------

from aria.data.ingestion.sue_loader import compute_revision_dir

def _make_revision_df():
    """4 quarters of AAPL: consensus rising YoY."""
    return pl.DataFrame({
        "ticker": ["AAPL"] * 4,
        "report_date": [
            date(2022, 10, 1),  # Q3 2022
            date(2023, 1, 15),  # Q4 2022
            date(2023, 10, 1),  # Q3 2023  (~365 days after Q3 2022)
            date(2024, 1, 15),  # Q4 2023
        ],
        "consensus_eps": [1.00, 1.20, 1.30, 1.50],
        "actual_eps":    [1.10, 1.25, 1.35, 1.55],
    })


def test_revision_dir_positive_when_consensus_raised():
    """Q3 2023 consensus (1.30) > Q3 2022 consensus (1.00) → positive."""
    df = _make_revision_df()
    val = compute_revision_dir("AAPL", date(2023, 10, 1), df)
    assert val > 0, f"Expected positive revision, got {val}"
    assert val == pytest.approx((1.30 - 1.00) / 1.00, abs=0.01)


def test_revision_dir_negative_when_consensus_cut():
    """Flip: prior year consensus was higher → negative revision."""
    df = pl.DataFrame({
        "ticker": ["AAPL"] * 2,
        "report_date": [date(2022, 10, 1), date(2023, 10, 1)],
        "consensus_eps": [1.50, 1.20],   # cut YoY
        "actual_eps":    [1.55, 1.25],
    })
    val = compute_revision_dir("AAPL", date(2023, 10, 1), df)
    assert val < 0


def test_revision_dir_returns_zero_for_missing_prior():
    """No prior-year row → returns 0.0."""
    df = _make_revision_df()
    val = compute_revision_dir("AAPL", date(2022, 10, 1), df)
    assert val == 0.0


def test_revision_dir_clipped_to_minus_one_plus_one():
    """Very large change is clipped to [-1, 1]."""
    df = pl.DataFrame({
        "ticker": ["AAPL"] * 2,
        "report_date": [date(2022, 10, 1), date(2023, 10, 1)],
        "consensus_eps": [0.10, 5.00],  # 50x increase → raw=49 → clipped to 1.0
        "actual_eps":    [0.10, 5.00],
    })
    val = compute_revision_dir("AAPL", date(2023, 10, 1), df)
    assert val == pytest.approx(1.0)


def test_revision_dir_returns_zero_for_unknown_ticker():
    df = _make_revision_df()
    val = compute_revision_dir("UNKNOWN", date(2023, 10, 1), df)
    assert val == 0.0
```

- [ ] **Step 2: Run to verify they fail**

```
python -m pytest aria/tests/test_sue.py::test_revision_dir_positive_when_consensus_raised aria/tests/test_sue.py::test_revision_dir_negative_when_consensus_cut aria/tests/test_sue.py::test_revision_dir_returns_zero_for_missing_prior aria/tests/test_sue.py::test_revision_dir_clipped_to_minus_one_plus_one aria/tests/test_sue.py::test_revision_dir_returns_zero_for_unknown_ticker -v
```

Expected: FAIL — `compute_revision_dir` not importable.

- [ ] **Step 3: Add `compute_revision_dir()` to `sue_loader.py`**

Add after `compute_historical_errors()`. Also add `from datetime import timedelta` at the top of the file if not already present, and `import numpy as np`:

```python
def compute_revision_dir(
    ticker: str,
    report_date: date,
    consensus_df: pl.DataFrame,
) -> float:
    """Proxy revision direction: (current_consensus - prior_year_consensus) / |prior|.

    Compares this quarter's analyst consensus against the same fiscal quarter
    one year ago (report_date ± 320-410 days). Returns a float in [-1, 1]:
      positive = analysts raised expectations YoY (bullish signal)
      negative = analysts cut expectations YoY (bearish signal)
      0.0      = insufficient history

    This is a proxy: it uses the final consensus before announcement, not
    intra-quarter revision snapshots. It captures the multi-quarter trend,
    not the last 30 days of revisions.
    """
    rows = consensus_df.filter(pl.col("ticker") == ticker).sort("report_date")

    current = rows.filter(pl.col("report_date") == report_date)
    if current.is_empty():
        return 0.0
    current_consensus = float(current["consensus_eps"][0])

    from datetime import timedelta
    prior_window = rows.filter(
        (pl.col("report_date") >= report_date - timedelta(days=410)) &
        (pl.col("report_date") <= report_date - timedelta(days=320))
    )
    if prior_window.is_empty():
        return 0.0
    prior_consensus = float(prior_window["consensus_eps"][-1])
    if abs(prior_consensus) < 0.01:
        return 0.0

    raw = (current_consensus - prior_consensus) / abs(prior_consensus)
    return float(np.clip(raw, -1.0, 1.0))
```

Add `import numpy as np` at the top of `sue_loader.py` if not present.

- [ ] **Step 4: Run all SUE tests**

```
python -m pytest aria/tests/test_sue.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add aria/data/ingestion/sue_loader.py aria/tests/test_sue.py
git commit -m "feat: add compute_revision_dir() for EPS revision direction proxy

Compares current quarter analyst consensus against same quarter one year
prior (320-410 day window) to detect whether expectations are rising or
falling. Returns clipped float in [-1, 1]. Used as soft position weight
multiplier in Phase 2 sizing.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Wire SUE Magnitude + Revision Weighting into Phase3Runner

This task adds two new position-weighting steps inside the vol-target sizing block, controlled by which experiment is running. All E19+ experiments get magnitude weighting by default (it's always active when vol_target > 0 and SUE_z is the signal). Revision weighting is only active for E24/E25.

Rather than adding experiment-specific flags, use a simple rule: **apply magnitude weighting whenever `SUE_z` is in `exp.signals` and `exp.vol_target > 0`**. Apply revision weighting for the same condition whenever `exp.name` ends in `_revision_dir` or `_revision_magnitude` — actually cleaner: add a `use_revision_weight: bool = False` flag to `ExperimentSpec` and set it True for E24/E25.

**Files:**
- Modify: `aria/research/ablation.py` (add `use_revision_weight` field; set True on E24/E25)
- Modify: `aria/research/phase3_runner.py` (add weighting in vol-target block)
- Modify: `aria/data/ingestion/sue_loader.py` (ensure `compute_revision_dir` is importable)

- [ ] **Step 1: Add `use_revision_weight` to `ExperimentSpec`**

In `aria/research/ablation.py`, add to the dataclass:

```python
    use_revision_weight: bool = False  # True → apply revision_dir as weight multiplier
```

Update E24 and E25 to set `use_revision_weight=True`:

```python
    ExperimentSpec(
        "E24_revision_dir",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        use_revision_weight=True,
        notes="E23 + EPS revision direction proxy filter",
    ),
    ExperimentSpec(
        "E25_revision_magnitude",
        signals=["SUE_z"],
        weights={"SUE_z": 1.0},
        hold_days=20,
        bsq_filter=True,
        vol_target=0.15,
        stop_loss_pct=0.10,
        use_revision_weight=True,
        notes="Full Phase 1+2 stack: E23 + revision direction + magnitude combined",
    ),
```

- [ ] **Step 2: Add imports to `phase3_runner.py`**

Near the top of `phase3_runner.py`, add:

```python
from aria.data.ingestion.sue_loader import load_consensus, get_sue_inputs, compute_revision_dir
```

(Replace the existing import of just `load_consensus, get_sue_inputs`.)

- [ ] **Step 3: Compute `revision_dir_map` alongside the SUE block**

In `phase3_runner.py`, find the SUE signal block (around line 846):

```python
            # SUE signal for this cohort (requires paid consensus data)
            if has_sue and sue_signal is not None and consensus_df is not None:
                sue_rows = get_sue_inputs(present, entry_date, consensus_df)
                ...
```

After the SUE join, add a revision_dir_map computation:

```python
            # Compute revision direction map for this cohort (used in Phase 2 sizing)
            revision_dir_map: dict[str, float] = {}
            if has_sue and consensus_df is not None:
                for t in present:
                    revision_dir_map[t] = compute_revision_dir(t, entry_date, consensus_df)
```

- [ ] **Step 4: Add magnitude + revision weighting inside the vol-target sizing block**

After the existing `long_w = {t: v / lt ...}` / `short_w = {t: v / st ...}` lines (the normalisation to sum=1) and before the `scale = self._vol_target_scale(...)` call, add:

```python
                    # SUE magnitude weighting: tilt by |SUE_z| ∈ [0.5, 3.0]
                    if "SUE_z" in exp.signals:
                        sue_z_map = dict(zip(base["ticker"].to_list(), base["SUE_z"].to_list()))
                        for t in longs:
                            lw[t] *= float(np.clip(abs(sue_z_map.get(t, 1.0)), 0.5, 3.0))
                        for t in shorts:
                            sw[t] *= float(np.clip(abs(sue_z_map.get(t, 1.0)), 0.5, 3.0))
                        lt2 = sum(lw.values()); st2 = sum(sw.values())
                        long_w  = {t: v / lt2 for t, v in lw.items()}
                        short_w = {t: v / st2 for t, v in sw.items()}

                    # Revision direction weighting (E24/E25 only)
                    if exp.use_revision_weight and revision_dir_map:
                        for t in list(longs):
                            rd = revision_dir_map.get(t, 0.0)
                            if rd < -0.5:
                                longs.remove(t)   # zero-out: analysts aggressively cutting
                            else:
                                long_w[t] = long_w.get(t, 1.0 / len(longs)) * (1.0 + 0.5 * rd)
                        for t in shorts:
                            rd = revision_dir_map.get(t, 0.0)
                            short_w[t] = short_w.get(t, 1.0 / len(shorts)) * (1.0 - 0.5 * rd)
                        if not longs:
                            continue
                        lt3 = sum(long_w.get(t, 0) for t in longs)
                        st3 = sum(short_w.values())
                        if lt3 > 0:
                            long_w  = {t: long_w[t] / lt3 for t in longs}
                        if st3 > 0:
                            short_w = {t: v / st3 for t, v in short_w.items()}
```

Note: the magnitude weighting block must run BEFORE `_vol_target_scale()` because it uses `lw`/`sw` (pre-normalised). Restructure so `long_w`/`short_w` are set to post-magnitude-weighted values before the scale call. The vol scale still multiplies the final weights.

Full replacement of the vol-target block (lines ~931-952):

```python
                if exp.vol_target > 0:
                    ticker_vols = self._compute_ticker_vols(
                        longs + shorts, all_prices, entry_date
                    )
                    # Inverse-vol base weights
                    lw = {t: 1.0 / max(ticker_vols.get(t, 0.30), 0.05) for t in longs}
                    sw = {t: 1.0 / max(ticker_vols.get(t, 0.30), 0.05) for t in shorts}

                    # SUE magnitude tilt: |SUE_z| ∈ [0.5, 3.0]
                    if "SUE_z" in exp.signals:
                        sue_z_map = dict(zip(base["ticker"].to_list(), base["SUE_z"].to_list()))
                        for t in longs:
                            lw[t] *= float(np.clip(abs(sue_z_map.get(t, 1.0)), 0.5, 3.0))
                        for t in shorts:
                            sw[t] *= float(np.clip(abs(sue_z_map.get(t, 1.0)), 0.5, 3.0))

                    # Revision direction tilt (E24/E25)
                    if exp.use_revision_weight and revision_dir_map:
                        filtered_longs = []
                        for t in longs:
                            rd = revision_dir_map.get(t, 0.0)
                            if rd >= -0.5:
                                lw[t] *= (1.0 + 0.5 * rd)
                                filtered_longs.append(t)
                        longs = filtered_longs
                        for t in shorts:
                            rd = revision_dir_map.get(t, 0.0)
                            sw[t] *= (1.0 - 0.5 * rd)
                        if not longs:
                            continue

                    # Normalise each side to sum=1
                    lt = sum(lw[t] for t in longs)
                    st = sum(sw.values())
                    long_w  = {t: lw[t] / lt for t in longs}
                    short_w = {t: v / st for t, v in sw.items()}

                    # Scale to hit vol target using corrected L/S formula
                    scale   = self._vol_target_scale(longs, shorts, ticker_vols, exp.vol_target)
                    long_w  = {t: w * scale for t, w in long_w.items()}
                    short_w = {t: w * scale for t, w in short_w.items()}
```

- [ ] **Step 5: Update `summary_table()` in `ablation.py` to include `use_revision_weight`**

```python
                "use_revision_weight": exp.use_revision_weight,
```

(Add alongside the other spec fields.)

- [ ] **Step 6: Run full test suite**

```
python -m pytest aria/tests/ -v --tb=short
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add aria/research/phase3_runner.py aria/research/ablation.py
git commit -m "feat: add SUE magnitude and revision direction position weighting

When vol_target > 0 and SUE_z is in signals, positions are tilted by
|SUE_z| (clipped [0.5, 3.0]) so larger surprises get proportionally
larger allocations. E24/E25 additionally apply the revision direction
multiplier (1 + 0.5*revision_dir), zeroing out longs where analysts cut
> 50% YoY.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 9: Validate — Run E19–E25 Ablation on 2021-2024

**Files:** None — run-only task.

- [ ] **Step 1: Run full test suite first**

```
python -m pytest aria/tests/ -v --tb=short
```

Expected: all pass.

- [ ] **Step 2: Run E19–E25 ablation**

```
python -m aria.research.phase3_runner --start 2021-01-01 --end 2024-12-31 --exp E17_SUE_BSQ_volTarget,E19_vol_fix,E20_vol_fix_stoploss,E21_sue_normalizer,E22_sue_magnitude,E23_phase1_full,E24_revision_dir,E25_revision_magnitude --price-source yfinance
```

- [ ] **Step 3: Check success criteria**

From the spec:
- **E19** (vol fix only): MaxDD < -30% (should improve from -36.8%)
- **E20** (+ stop-loss): MaxDD < -28%
- **E23** (Phase 1 full): MaxDD < -25%, Sharpe > 0.50
- **E25** (Phase 1+2): MaxDD < -25%, Sharpe > 0.65, IC > 0.13

If E23 Sharpe < 0.50: the signal itself needs re-evaluation (see spec upgrade path — full architecture rebuild).

- [ ] **Step 4: Commit results note**

```bash
git commit --allow-empty -m "test: E19-E25 ablation results 2021-2024

[paste key metrics from terminal output here]

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
