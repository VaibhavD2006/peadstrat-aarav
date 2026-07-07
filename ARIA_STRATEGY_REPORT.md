# ARIA Strategy Report
**Date:** July 4, 2026  
**Status:** Phase 3 Complete — Results Below Target

---

## 1. Strategy Summary

ARIA is a long/short equity strategy that trades around corporate earnings announcements. The core hypothesis is that earnings quality signals — revenue surprise, margin expansion, EPS growth, and filing timeliness — can identify stocks likely to outperform (longs) or underperform (shorts) in the days following their earnings release.

### Universe
- ~800 US-listed tickers with average daily volume ≥ $50M (SimFin free tier)
- Filtered to tickers with available quarterly income statement history

### Entry and Exit
- **Entry:** The earnings publish date for each ticker (per-event, not quarterly batches)
- **Exit:** 5 trading days after entry (1-week hold)
- **Position sizing:** Equal-weight within longs and shorts; top 10% = longs, bottom 10% = shorts
- **Transaction costs:** ~32 basis points round-trip (bid/ask spread + market impact + borrow cost)

### Signals

| Signal | Description | Data Source |
|--------|-------------|-------------|
| **ESQS** | Earnings Surprise Quality Score — composite of revenue surprise (30%), gross margin expansion (25%), SGA efficiency (20%), EPS YoY change (25%) | SimFin quarterly income statements |
| **FTS** | Filing Timeliness Signal — days early/late to SEC 10-Q deadline vs. same quarter prior year | SEC EDGAR Submissions API (free) |
| **IFR** | Institutional Flow Residual — rolling 60-day OLS of ticker log-volume vs. sector ETF log-volume; positive residual = unusual buy pressure | SimFin prices + yfinance sector ETFs |
| **IVRS** | Implied Volatility Regime Signal — short/long realized-vol ratio cross-sectionally z-scored; used as a position-size multiplier, not a direction signal | SimFin daily prices |
| **HMM** | 4-state Hidden Markov Model regime classifier (Bull / Bear / Crisis / Recovery) trained on rolling 2-year windows; gates trades to Bull and Recovery regimes only | SimFin daily prices (SPY proxy) |

### ESQS Component Weights
```
rev_surp   30%  — revenue vs. same quarter prior year
gm_expand  25%  — gross margin vs. trailing average
sga_eff    20%  — SGA efficiency vs. trailing average
eps_yoy    25%  — EPS vs. same quarter prior year
```

---

## 2. Backtest Results

**Period:** January 1, 2021 – December 31, 2024  
**Benchmark:** SPY (~14% CAGR, MaxDD ~-25% over same period)  
**Target:** ≥12% CAGR, MaxDD < -20%

### Ablation Matrix (E01–E10)

| Experiment | Signals | Regime Filter | IVRS Mult | Notes |
|------------|---------|:---:|:---:|-------|
| E01 | ESQS | — | — | Baseline |
| E02 | FTS | — | — | Filing timeliness only |
| E03 | ESQS + FTS | — | — | Core composite |
| E04 | ESQS + FTS | — | ✓ | E03 + position sizing |
| E05 | IFR | — | — | Institutional flow standalone |
| E06 | ESQS + IFR | — | — | Earnings quality + flow |
| E07 | ESQS + FTS + IFR | — | ✓ | Full stack, no regime gate |
| E08 | ESQS + FTS + IFR | ✓ | ✓ | Full stack + HMM gate |
| E09 | ESQS + FTS + IFR (equal-weight) | ✓ | — | **Best result** |
| E10 | ESQS (YoY-fixed) | — | — | Isolation test |

### Best Result: E09

| Metric | E09 Result | Target |
|--------|-----------|--------|
| CAGR | **+8.5%** | ≥ 12% |
| Sharpe Ratio | **+0.36** | > 1.0 |
| Max Drawdown | **-58.9%** | < -20% |
| Win Rate | ~53% | — |

The best experiment fell short of both targets. CAGR is 3.5 percentage points below the 12% target, and the maximum drawdown of -58.9% is nearly three times worse than the -20% limit.

---

## 3. Problems with the Strategy

### Problem 1: The Primary Signal Has Near-Zero Predictive Power

This is the root cause. The ESQS revenue surprise component — which carries 30% of the composite weight — has an Information Coefficient (IC, Spearman rank correlation vs. forward returns) that is statistically indistinguishable from zero:

| Horizon | IC | p-value |
|---------|----|---------|
| 1-day forward return | +0.012 | 0.856 |
| 5-day forward return | **-0.086** | 0.185 |
| 20-day forward return | -0.044 | 0.502 |

All p-values are far above 0.05. The signal is random noise. A long/short strategy built on a zero-IC signal produces returns ≈ 0% minus transaction costs — exactly what we observed.

### Problem 2: No Analyst Consensus Data

The revenue "surprise" is computed as `(actual_revenue − prior_year_same_quarter_revenue) / prior_year`. This measures year-over-year growth, not surprise relative to analyst expectations.

The stock market already knows last year's revenue. YoY growth is priced in. A stock that grew revenue 15% YoY when analysts expected 20% will still fall on earnings day — but our signal would score it positively.

Without analyst consensus estimates (which require paid data: Bloomberg, FactSet, Refinitiv), we cannot compute a genuine earnings surprise. The entire ESQS "surprise" component is built on information the market has already incorporated.

### Problem 3: Max Drawdown of -58.9%

The strategy is constructed as equal-dollar-neutral (equal capital on longs and shorts), but not **beta-neutral**. During the 2022 bear market (SPY -20%), tech-heavy longs with market beta of 1.3–1.5 lost significantly more than defensive shorts with beta 0.6–0.8. The short book did not hedge the long book.

The -58.9% drawdown occurred primarily in 2022. A dollar-neutral strategy with mismatched beta bleeds like a directional long book during market selloffs.

### Problem 4: FTS Signal Is Effectively Zero

The Filing Timeliness Signal (FTS) requires SEC EDGAR filing data to be downloaded and cached per ticker. In the Phase 3 run, the EDGAR data was not pre-populated, so `get_filing_history()` returned empty results for most tickers. The FTS column was set to 0.0 for all positions. E09's "equal-weight ESQS + FTS + IFR" was in practice just "50% ESQS + 50% IFR" with FTS contributing nothing.

### Problem 5: Hold Period Too Short for the Effect We're Targeting

Post-Earnings Announcement Drift (PEAD) — the well-documented tendency for stocks to continue moving in the direction of their earnings reaction — plays out over 20–60 days, not 5. With a 5-day hold, the strategy captures initial noise around the announcement rather than the actual drift.

Empirical evidence shows:
- 5-day IC for earnings reaction vs. future return: ~0.05–0.08 (modest)
- 20-day IC for earnings reaction vs. future return: ~0.10–0.15 (meaningful)

Holding for 5 days exits before the signal has time to work.

### Problem 6: Transaction Costs Overwhelm the Alpha

At ~32 bps round-trip per trade, with the current signal generating IC ≈ 0.025 on the composite, the theoretical gross alpha is approximately:
```
Gross alpha ≈ IC × volatility × √(2/π) ≈ 0.025 × 7.4% × 0.8 ≈ 0.15% per 5-day period
Annualized ≈ 0.15% × 52 ≈ 7.8%
Less costs  ≈ 0.32% × 52 ≈ 16.6% (two 32bps trades per week, entry + exit)
Net         ≈ 7.8% − 16.6% = −8.8%
```

With the actual per-event frequency (not every week), cost drag is lower — but the signal is so weak that even a fraction of this cost structure wipes out the alpha.

---

## 4. What Would Actually Fix This

### Fix 1: Switch to PEAD (Post-Earnings Announcement Drift)

Instead of trying to predict whether a company will beat expectations before the announcement, enter the trade **after** the earnings release using the market's own reaction as the signal.

- **Signal:** The stock's return on announcement day (e.g., +5% gap = strong buy signal; -5% gap = strong short signal)
- **Entry:** Day after the announcement (not on the announcement date itself)
- **Hold:** 20 trading days
- **Why it works:** Institutions are slow to reposition; individual investors anchored to pre-announcement prices; analyst estimate revisions propagate over weeks

This requires zero paid data — it's purely price-based from SimFin. IC for this signal in the literature is typically +0.08 to +0.15 at the 20-day horizon.

### Fix 2: Beta-Neutral Position Sizing

Compute a rolling 252-day beta for each stock relative to SPY. Size each position proportional to `1/beta` so that the portfolio's net beta exposure is close to zero on both the long and short books.

Example:
- NVDA (beta ≈ 1.8): position weight = 1/1.8 = 55% of equal-weight size
- JNJ (beta ≈ 0.6): position weight = 1/0.6 = 167% of equal-weight size

This directly fixes the MaxDD problem by preventing the strategy from bleeding during market-wide selloffs.

### Fix 3: Longer Hold Period (20 Days)

Changing `hold_days` from 5 to 20 aligns the holding period with when PEAD effects actually materialize. This also reduces turnover by 4×, cutting transaction costs from ~16% annualized drag to ~4%.

### Fix 4: Pre-Populate EDGAR Cache for FTS Signal

The FTS code is complete and correct. The fix is operational: run a one-time script to download SEC EDGAR filing histories for all ~800 universe tickers before running the backtest. Once cached (30-day TTL), FTS will contribute real signal in subsequent runs.

---

## 5. Summary

The strategy as built has a valid architecture but an invalid core signal. Revenue YoY growth is not earnings surprise — it's already priced. The result is a long/short book with IC ≈ 0 that underperforms after costs, with severe drawdowns from unhedged market beta exposure during 2022.

The two highest-leverage fixes are: (1) replace the pre-announcement prediction approach with PEAD, and (2) make the book beta-neutral. Both are implementable with free data and changes to the existing codebase.
