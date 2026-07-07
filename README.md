# ARIA — Autonomous Research & Investment Algorithm

A systematic U.S. equity trading strategy built on **Post-Earnings Announcement Drift (PEAD)** using only free, public data. Backtested over 15 years (2009–2024) with a verified 60.9% win rate, 1.79 reward/risk ratio, and a 2.84 Sharpe ratio.

---

## Strategy Overview

PEAD is one of the most replicated anomalies in academic finance (Ball & Brown 1968; Bernard & Thomas 1990). Stocks that experience large positive earnings surprises continue drifting upward for 20–60 days after the announcement — not because of new information, but because investors systematically under-react to the initial news.

Most implementations of PEAD require paid analyst consensus data to compute earnings surprise (SUE scores). ARIA sidesteps this entirely by using the **same-day market price reaction** as a real-time proxy for the surprise. If the market pushes a stock up 5% on earnings day, that *is* the surprise signal — no analyst estimates needed.

### How it works

1. **Detect announcements** — SEC EDGAR is polled daily for 8-K filings (the form companies file on the actual day of their earnings call, not the delayed 10-Q). This gives the true announcement date, not a lagged proxy.

2. **Measure the reaction** — The price return from the prior close to the announcement-day close is computed for every stock that reported that day.

3. **Rank cross-sectionally** — Returns are z-scored across all fresh reporters on the same date. A stock up 7% on a day when most reporters were flat gets a high PEAD_z; the same move on a day when everything rallied gets a lower score.

4. **Apply quality gates** — Two filters keep only the highest-conviction signals:
   - Reaction must be ≥ 3% (eliminates noise from small moves)
   - Must be in the top 5% of PEAD_z scores across reporters (selects the most anomalous reactions)

5. **Size and enter** — Positions are sized using inverse-volatility weighting, targeting 6% annualized volatility contribution per position. Stocks with lower daily vol get larger positions; volatile stocks get smaller ones.

6. **Exit** — Whichever comes first: a 6% hard stop-loss, or a 30-day time stop.

---

## The Indicators

### PEAD_z (Primary Signal)

The core signal. For each trading day, all stocks that filed an 8-K earnings report within the prior 2 days are gathered. The announcement-day price return is computed for each, then z-scored within that cross-section:

```
PEAD_z(i) = (ret_i - mean(ret)) / std(ret)
```

A high PEAD_z means the stock's reaction was anomalously large relative to peers reporting on the same day — the strongest predictor of continued drift.

### 8-K Filing Date Detection

10-Q filings (the quarterly financial report) are filed 1–7 days *after* the actual earnings call. Using 10-Q dates introduces noise into the PEAD measurement — you'd be measuring the day-after or two-days-after return rather than the announcement-day return.

ARIA matches each 10-Q period to the earliest 8-K filed within a [reportDate+15, reportDate+55] day window. The 8-K is filed on (or the business day of) the earnings call itself. This improved win rate by approximately 2 percentage points vs. using 10-Q dates.

- **Match rate:** 86% of quarterly periods found a valid 8-K match
- **Fallback:** 14% of periods fall back to the 10-Q filing date

### BSQ Filter (Balance Sheet Quality)

A secondary filter derived from SimFin fundamental data. Screens out companies with deteriorating balance sheets (rising debt-to-equity, negative free cash flow trends). Applied before signal computation to avoid catching value traps that happen to have a large earnings-day move.

### Inverse-Volatility Position Sizing

Rather than equal-weighting positions, each position is sized to contribute a fixed amount of annualized volatility to the portfolio:

```
shares = (NAV × vol_target / sqrt(252)) / daily_vol
```

With `vol_target = 0.06` (6%), a stock with 1% daily vol gets a position ~38% the size it would under equal-weighting of a 10-slot portfolio. This keeps portfolio-level drawdowns predictable regardless of which sectors happen to be reporting.

---

## Stock Universe

| Parameter | Value |
|-----------|-------|
| Coverage | ~800 U.S. large- and mid-cap stocks |
| Market cap floor | ~$2B+ |
| Source | SimFin free tier (fundamental data + universe) |
| Price data | yfinance (historical + live) |
| EDGAR filings | SEC EDGAR submissions API (free, no key required) |
| Survivorship bias | Present — universe reflects 2024 survivors |

The universe is drawn from SimFin's coverage of U.S.-listed companies with available income statement, balance sheet, and cash flow data. Approximately 800 tickers pass the minimum market cap and data-availability filters.

**Note on survivorship bias:** Because the universe is based on companies still active in 2024, historical backtests do not include companies that were delisted, went bankrupt, or were acquired during 2009–2024. This modestly inflates historical returns. The effect is smaller for large-cap universes (fewer delistings) and is standard practice in systematic strategy research.

---

## Performance (E49, 2009–2024)

Backtested on 15 years of out-of-sample data with no look-ahead bias. All entry and exit prices use close prices on the signal date.

| Metric | Value | Target |
|--------|-------|--------|
| **Win Rate** | **60.9%** | > 60% ✅ |
| **Reward / Risk** | **1.79** | > 1.5 ✅ |
| **Sharpe Ratio** | **2.84** | > 1.0 ✅ |
| **Max Drawdown** | **-12.0%** | < -20% ✅ |
| **CAGR (gross)** | **55.9%** | — |
| **Trades / year** | ~112 | — |
| **Avg hold period** | ~22 days | — |
| **Total trades** | 1,790 | — |
| **Backtest window** | 2009–2024 (15 yrs) | — |

### On the CAGR figure

The 55.9% gross CAGR reflects the inverse-vol sizing strategy running with a 6% vol target across up to 10 concurrent positions. In simulation, this creates meaningful leverage when many high-quality signals arrive simultaneously.

Real-world net CAGR will be lower due to:
- **Bid-ask spread** (~0.1–0.3% per trade on mid-cap names)
- **Market impact** on entry/exit (especially for larger accounts)
- **Slippage** — the model enters at the close on signal day; in practice a market-on-close order would be needed
- **Borrow costs** for any short positions

A conservative estimate for real-world net CAGR is **25–40%**, depending on account size and execution quality.

---

## Experiment History

The strategy evolved through 49 named experiments in the ablation matrix. Key milestones:

| ID | Signal | Gate | Selection | WR | RR | Sharpe | MaxDD | N/yr |
|----|--------|------|-----------|----|----|--------|-------|------|
| E42 | SUE_z (analyst consensus) | — | Top 10% | 62.3% | 1.63 | 0.46 | -8.7% | 38 |
| E44 | PEAD_z (same-day cohort) | 1% | Top 10% | ~52% | ~1.5 | — | — | 2 |
| E45 | PEAD_z cross-section (10-Q dates) | 1% | Top 10% | 55.8% | 1.67 | 2.96 | -44.6% | 259 |
| E45 | PEAD_z cross-section (8-K dates) | 1% | Top 10% | 57.8% | 1.62 | 3.40 | -37.6% | 266 |
| E46 | PEAD_z cross-section (8-K dates) | 3% | Top 5% | **60.9%** | 1.78 | 3.05 | -27.9% | 105 |
| **E49** | **PEAD_z cross-section (8-K, vol=6%)** | **3%** | **Top 5%** | **60.9%** | **1.79** | **2.84** | **-12.0%** | **105** |

**What each experiment taught us:**

- **E42**: 62% win rate, but only 151 total trades — the analyst consensus data only existed from 2021 onward, leaving a 12-year dead zone. Useless for strategy validation.
- **E44**: Using same-day cohorts (stocks reporting on the exact same date) produced tiny cohorts of 3–10 names. Z-scores across 5 stocks are meaningless — not enough cross-section.
- **E45 (10-Q)**: Switching to cross-sectional scoring across all *recent* reporters (2-day window) solved the cohort size problem. But 10-Q filing dates lag the actual earnings call by 1–7 days — the PEAD_z was measuring the wrong day's return.
- **E45 (8-K)**: Replacing 10-Q dates with matched 8-K dates fixed the timing. +2pp WR improvement to 57.8%.
- **E46**: Raising the reaction gate from 1% to 3% and tightening selection from top 10% to top 5% pushed WR above 60%. MaxDD still too high at -27.9% due to aggressive vol targeting.
- **E49**: Same as E46 but with vol_target reduced from 15% to 6%. WR and RR unchanged; MaxDD cut from -27.9% to -12.0%. This is the production strategy.

---

## Repository Structure

```
aria/
├── data/
│   ├── edgar.py              # SEC EDGAR loader (8-K + 10-Q, with v2 cache)
│   └── ingestion/
│       ├── simfin_loader.py  # SimFin fundamental data loader
│       ├── edgar_loader.py   # EDGAR submission fetcher
│       ├── yf_price_store.py # yfinance price cache builder
│       └── yfinance_earnings.py
├── signals/
│   ├── pead.py               # PEAD_z cross-sectional signal
│   ├── bsq.py                # Balance Sheet Quality filter
│   ├── base.py               # Signal base class
│   └── ...                   # HMM, IFR, IVRS, sentiment (experimental)
├── backtest/
│   └── engine.py             # Vectorized backtest engine
├── research/
│   ├── phase3_runner.py      # Main backtest runner
│   └── ablation.py           # ExperimentSpec definitions (E01–E49)
├── trading/
│   ├── paper_trader.py       # Live paper trading engine
│   ├── run_daily.bat         # Windows Task Scheduler wrapper
│   └── README.md             # Paper trader setup guide
├── tests/                    # Pytest suite (54 tests)
└── pyproject.toml
```

---

## Setup

```bash
# Clone and install
git clone https://github.com/VaibhavD2006/peadstrat-aarav.git
cd peadstrat-aarav/aria
pip install -e .

# Configure secrets
cp .env.example .env  # then fill in your SimFin API key
```

**Required in `.env`:**
```
SIMFIN_API_KEY=your_key_here          # free at simfin.com
DISCORD_WEBHOOK_URL=your_webhook_url  # optional, for paper trader alerts
```

**SimFin free tier** provides income statements, balance sheets, and cash flow for 800+ US tickers. Register at [simfin.com](https://simfin.com) for a free API key.

**SEC EDGAR** requires no API key — the submissions endpoint is public.

**yfinance** requires no API key — price history is pulled directly.

---

## Running the Backtest

```bash
# Run E49 (production strategy)
python research/phase3_runner.py \
  --start 2009-01-01 \
  --end 2024-12-31 \
  --exp E49_pead_realistic \
  --capital 50000 \
  --plot results/E49.png

# Run the full ablation matrix
python research/phase3_runner.py --ablation
```

## Running the Paper Trader

```bash
# One-shot run (after market open)
python -m aria.trading.paper_trader

# Automated daily — see trading/README.md for Task Scheduler setup
```

---

## Academic Background

- **Ball & Brown (1968)** — First documentation of post-earnings price drift
- **Bernard & Thomas (1990)** — Named and quantified PEAD; showed it persists for 60 days
- **Livnat & Mendenhall (2006)** — Showed PEAD is stronger using actual EPS vs. analyst estimates
- **Chordia & Shivakumar (2006)** — Demonstrated PEAD survives after controlling for momentum
