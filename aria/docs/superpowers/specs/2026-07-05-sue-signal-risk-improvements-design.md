# ARIA SUE Signal & Risk Management Improvements

**Date:** 2026-07-05
**Status:** Approved
**Goal:** Improve live-forward performance of the SUE+BSQ strategy (E17 baseline: Sharpe=+0.34, AnnRet=+12.3%, MaxDD=-36.8% over 2021-2024). Target: MaxDD < -25%, Sharpe > 0.6.

---

## Context

The full 2018-2024 ablation confirmed SUE is the only signal with genuine alpha (IC=+0.092–0.127). All other signals (ESQS, PEAD, IFR, IVRS) have near-zero or negative IC over this period. The strategy is being optimized for live trading from 2024 onward, not for extending the historical backtest.

Two diagnosed failure modes:

1. **Vol targeting underestimates L/S portfolio vol by ~47%** — positions are oversized, causing MaxDD=-36.8% against a 15% vol target.
2. **SUE normalizer is imprecise** — falls back to `abs(consensus_eps)` when `forecast_std` is missing, ignoring per-ticker analyst accuracy history.

---

## Phase 1 — Risk Management Fixes

### 1A. Vol Targeting Formula (phase3_runner.py)

**Current (wrong):** Treats the L/S book as a long-only portfolio.
```python
port_vol_est = avg_vol * np.sqrt(rho + (1.0 - rho) / n)  # n = total positions
```

**Fixed:** Separate long and short side vol, then compute L/S portfolio vol correctly.
```python
vol_L = avg_vol_longs  * sqrt(rho_within + (1 - rho_within) / n_long)
vol_S = avg_vol_shorts * sqrt(rho_within + (1 - rho_within) / n_short)
port_vol_est = sqrt(vol_L**2 + vol_S**2 - 2 * rho_ls * vol_L * vol_S)
scale = min(target_vol / max(port_vol_est, 0.01), 2.0)
```

Parameters: `rho_within = 0.30` (avg pairwise correlation within each side), `rho_ls = 0.30` (avg cross-correlation between long and short sides).

**Expected impact:** With n_long=n_short=1 and avg_vol=0.35, estimated port_vol rises from 0.28 to 0.41. Scale drops from 0.53 to 0.36. Position sizes shrink ~32%, bringing MaxDD from ~-37% to ~-25%.

Extract into a private `_vol_target_scale(longs, shorts, ticker_vols, target_vol)` method for testability.

### 1B. Per-Trade Stop-Loss (backtest/engine.py, research/ablation.py)

Add `stop_loss_pct: float = 0.0` to `BacktestConfig`. When nonzero, `BacktestEngine` tracks cumulative position PnL daily and exits on the first close where `cumulative_return < -stop_loss_pct`. The exit date replaces the scheduled hold-period exit.

Default for E17/E18: `stop_loss_pct = 0.10` (10% per position).

Implementation note: stop-loss applies at the individual position level (not portfolio). A long that drops 10% exits; the offsetting short remains open.

---

## Phase 2 — SUE Signal Quality

### 2A. Rolling Historical Error Std as Normalizer (sue.py, sue_loader.py)

Replace the `abs(consensus_eps)` fallback with a rolling std of the past 4 quarters of analyst errors for that ticker.

```python
# sue_loader.py
def compute_historical_errors(ticker, as_of_date, consensus_df, n_quarters=4):
    """Return list of (actual_eps - consensus_eps) for the n_quarters prior to as_of_date."""
    rows = (consensus_df
            .filter((pl.col("ticker") == ticker) & (pl.col("report_date") < as_of_date))
            .sort("report_date")
            .tail(n_quarters))
    return (rows["actual_eps"] - rows["consensus_eps"]).to_list()

# sue.py — normalizer priority
if len(hist_errors) >= 2:
    normalizer = max(float(np.std(hist_errors, ddof=1)), fallback_scale)
elif forecast_std is not None and forecast_std > fallback_scale:
    normalizer = forecast_std
else:
    normalizer = max(abs(consensus_eps), fallback_scale)
```

Requires ≥2 quarters of history to activate; falls back gracefully to existing logic.

### 2B. SUE Magnitude Weighting (phase3_runner.py)

Within the vol-targeting position sizing block, tilt weights by `|SUE_z|` clipped to [0.5, 3.0]:

```python
# sue_z_map: dict[ticker -> SUE_z] built from base DataFrame before sizing
sue_z_map = dict(zip(base["ticker"].to_list(), base["SUE_z"].to_list()))
for t in longs:
    lw[t] *= float(np.clip(abs(sue_z_map.get(t, 1.0)), 0.5, 3.0))
for t in shorts:
    sw[t] *= float(np.clip(abs(sue_z_map.get(t, 1.0)), 0.5, 3.0))
# renormalise each side to sum=1 before applying vol scale
lt = sum(lw.values()); st = sum(sw.values())
long_w  = {t: v / lt for t, v in lw.items()}
short_w = {t: v / st for t, v in sw.items()}
```

Renormalise long and short books to sum to 1 after applying magnitude weights. Gross exposure stays controlled by vol targeting. A 2.5σ surprise gets ~5× the weight of a 0.5σ surprise within the same cohort.

---

## Phase 3 — EPS Revision Direction (Approach 2)

### 3A. Revision Direction Proxy (sue_loader.py)

Compute a revision direction signal from existing consensus data by comparing the current quarter's consensus against the same fiscal quarter one year prior:

```python
def compute_revision_dir(ticker, report_date, consensus_df):
    """
    Returns float in [-1, 1]: positive = analysts raised expectations YoY,
    negative = analysts cut. Returns 0.0 if insufficient history.
    """
    rows = consensus_df.filter(pl.col("ticker") == ticker).sort("report_date")
    # Current quarter row (the row being evaluated)
    current = rows.filter(pl.col("report_date") == report_date)
    if current.is_empty():
        return 0.0
    current_consensus = float(current["consensus_eps"][0])
    # Prior same-quarter row: report_date between 320 and 410 days ago
    prior_window = rows.filter(
        (pl.col("report_date") >= report_date - timedelta(days=410)) &
        (pl.col("report_date") <= report_date - timedelta(days=320))
    )
    if prior_window.is_empty() or abs(float(prior_window["consensus_eps"][-1])) < 0.01:
        return 0.0
    prior_consensus = float(prior_window["consensus_eps"][-1])
    raw = (current_consensus - prior_consensus) / abs(prior_consensus)
    return float(np.clip(raw, -1.0, 1.0))
```

### 3B. Revision-Weighted Position Sizing (phase3_runner.py)

Combine revision direction with SUE magnitude as a soft multiplier on position weight:

```python
revision_multiplier = 1.0 + 0.5 * revision_dir   # range [0.5, 1.5]
final_weight ∝ |SUE_z| * revision_multiplier
```

Long candidates with positive revision direction get up to 1.5× weight; negative revision cuts to 0.5×. Zero-out longs with `revision_dir < -0.5` (analysts aggressively cutting consensus before print).

### 3C. Upgrade Path

If the proxy proves too noisy (IC improvement < 0.01), replace `compute_revision_dir()` with Tiingo EPS estimate history (~$10/mo). The weight formula and integration point stay identical; only the data source changes. No other code changes needed.

---

## Ablation Experiments to Add

| Exp | Changes vs E17 baseline | Purpose |
|---|---|---|
| E19_vol_fix | Vol targeting fix only | Isolate MaxDD impact |
| E20_vol_fix_stoploss | Vol fix + 10% stop-loss | Isolate stop-loss impact |
| E21_sue_normalizer | Vol fix + rolling error std normalizer | Isolate normalizer impact |
| E22_sue_magnitude | Vol fix + SUE magnitude weighting | Isolate magnitude sizing impact |
| E23_phase1_full | All Phase 1 changes combined | Phase 1 baseline |
| E24_revision_dir | Phase 1 + revision direction proxy | Phase 2 proxy |
| E25_revision_magnitude | Phase 1 + revision + magnitude combined | Full Phase 1+2 stack |

Run all on 2021-2024 (clean SUE data). Compare to E17 baseline (Sharpe=+0.34, MaxDD=-36.8%).

---

## File Map

| File | Change |
|---|---|
| `aria/research/phase3_runner.py` | Fix vol formula → `_vol_target_scale()`; add magnitude + revision weighting |
| `aria/backtest/engine.py` | Add `stop_loss_pct` to `BacktestConfig`; implement daily stop check |
| `aria/research/ablation.py` | Add E19–E25; add `stop_loss_pct: float = 0.0` field to `ExperimentSpec` dataclass |
| `aria/data/ingestion/sue_loader.py` | Add `compute_historical_errors()`, `compute_revision_dir()` |
| `aria/signals/sue.py` | Update `compute_sue_raw()` to accept `hist_errors` |
| `aria/tests/test_sue.py` | Cover new normalizer priority logic |
| `aria/tests/test_backtest.py` | Cover stop-loss exit timing |

---

## Success Criteria

- E23 (Phase 1 full): MaxDD < -25%, Sharpe > 0.50
- E25 (Phase 1+2 full): MaxDD < -25%, Sharpe > 0.65, IC > 0.13
- All existing tests pass; new tests cover each changed module
- Vol targeting achieves realized AnnVol within ±5pp of 15% target on E19

---

## Upgrade Path if Phase 1+2 Insufficient

If E25 Sharpe < 0.50 after clean 2021-2024 validation:
- Replace SUE-only alpha with a 3-factor earnings stack: SUE + estimate revision breadth (paid data) + post-earnings 1-5d price confirmation gate
- Remove BSQ filter; replace with sector-neutral constraint
- This is a full architecture rebuild (separate spec)
