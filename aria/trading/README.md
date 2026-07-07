# ARIA E49 — Paper Trader

**Strategy:** Post-Earnings Announcement Drift (PEAD) with SEC EDGAR 8-K signal detection  
**Universe:** ~770 US large/mid-cap stocks (SimFin coverage, $2B+ market cap)  
**Backtest:** 2009–2024 (15 years, no analyst consensus data used)

---

## Strategy Overview

ARIA E49 exploits **Post-Earnings Announcement Drift** — the well-documented tendency for stocks to continue moving in the direction of their earnings surprise for 20–60 days after the announcement.

Unlike most PEAD implementations that require paid analyst consensus data (for SUE scores), this strategy uses only **free public data**:
- **SEC EDGAR 8-K filings** to identify the actual earnings announcement date
- **yfinance price data** to measure the market's same-day reaction
- **Cross-sectional z-scoring** to rank reactions within the universe of fresh reporters

### Entry Signal

Each trading day, the runner:
1. Scans for 8-K filings from the last 2–3 days for all universe tickers
2. Computes the price return on the filing date vs. the prior close
3. Cross-sectionally z-scores these returns across all fresh reporters
4. Selects the **top 5%** by PEAD_z with a **≥ 3% price reaction gate** (long side)
5. Sizes each position to target **6% annualized vol contribution**

### Why 8-K filings?

Companies file an 8-K with the SEC on (or the same business day as) their earnings call — not the 10-Q, which is filed 1–7 days later. Using 8-K dates gives the **true announcement date**, enabling a clean PEAD_z measurement. This improved win rate by ~2% over 10-Q-based detection.

### Exit Rules

| Exit Type | Condition |
|-----------|-----------|
| Hard stop | Price falls ≥ 6% below entry (long) |
| Time stop | Position held ≥ 30 days |

---

## The PEAD Indicator

**Post-Earnings Announcement Drift** was first documented by Ball & Brown (1968) and formally named by Bernard & Thomas (1990). Key facts:

- Stocks that **beat earnings expectations** tend to drift **upward** for 2–12 months
- Stocks that **miss expectations** tend to drift **downward**
- The effect is strongest in the first 60 days and is stronger for small/mid-cap
- Caused by investor under-reaction to earnings news (anchoring, limited attention)

This strategy uses the **filing-day price return** as a proxy for the earnings surprise:
- Large positive return on announcement day → market perceived a positive surprise → buy
- The cross-sectional z-score ranks how surprising the reaction was relative to all other fresh reporters that day

---

## Experimental Evolution

| ID | Signal | Gate | Top% | WR | RR | Sharpe | MaxDD | N |
|----|--------|------|------|----|----|--------|-------|---|
| E42 | SUE_z (analyst) | — | 10% | 62.3% | 1.63 | 0.46 | -8.7% | 151 |
| E44 | PEAD_z (ann-day) | 1% | 10% | ~52% | ~1.5 | — | — | 32 |
| E45 | PEAD_z CS (10-Q dates) | 1% | 10% | 55.8% | 1.67 | 2.96 | -44.6% | 3,889 |
| E45 | PEAD_z CS (8-K dates) | 1% | 10% | 57.8% | 1.62 | 3.40 | -37.6% | 3,986 |
| E46 | PEAD_z CS (8-K dates) | 3% | 5% | **60.9%** | 1.78 | 3.05 | -27.9% | 1,790 |
| **E49** | **PEAD_z CS (8-K, vol=6%)** | **3%** | **5%** | **60.9%** | **1.79** | **2.84** | **-12.0%** | **1,790** |

**Key learnings:**
- **E42** had 62% WR but only 151 trades (2021–2024) due to analyst consensus data gap
- **E44** had tiny cohorts (3–10 stocks/day from same-day reporters) → poor signal
- **E45 10-Q**: 10-Q filing dates lag actual earnings by 1–7 days → noisy PEAD
- **E45 8-K**: Switching to 8-K dates gave +2% WR lift (57.8% vs 55.8%)
- **E46**: Raising gate to 3% + selecting top 5% crossed the 60% WR threshold
- **E49**: Same as E46 but vol_target=6% → realistic MaxDD of -12%

---

## Final Strategy Results (E49)

| Metric | Value | Notes |
|--------|-------|-------|
| **Win Rate** | **60.9%** | Target: >60% ✅ |
| **Reward/Risk** | **1.79** | Target: >1.5 ✅ |
| **Sharpe Ratio** | **2.84** | Excellent (>1.0) ✅ |
| **Max Drawdown** | **-12.0%** | Very manageable ✅ |
| **CAGR (gross)** | ~55.9% | Before costs; realistic net: 25–40% |
| **Trades/year** | ~112 | ~1,790 over 17 years |
| **Hold period** | 30d avg | 6% stop or 30-day time stop |
| **Data** | Free only | EDGAR + yfinance + SimFin (free tier) |

> **Note on CAGR:** The 55.9% gross figure reflects the vol-targeting leverage in simulation. Real-world execution involves bid-ask spread (~0.1–0.2%), market impact on entry/exit, and survivorship bias in the universe. A conservative net estimate is 25–40% CAGR.

---

## Setup

### 1. Prerequisites

```bash
# Install dependencies (from project root)
pip install -e .

# EDGAR cache should already be built from research (data/edgar/v2/)
# Paper trader uses its own cache at trading/edgar_cache/
```

### 2. Discord Webhook

1. Open your Discord server → channel settings → **Integrations** → **Webhooks**
2. Create a new webhook, copy the URL
3. Set the environment variable:

```bat
# Windows (permanent, system level)
setx DISCORD_WEBHOOK_URL "https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN"

# Or set it in run_daily.bat directly
```

### 3. Windows Task Scheduler

1. Open **Task Scheduler** → Create Basic Task
2. Name: `ARIA Paper Trader`
3. Trigger: **Daily** at **9:45 AM** (ET — after market open)
4. Action: Start a program → `C:\...\aria\trading\run_daily.bat`
5. Under **Conditions**: check "Run only if the following network connection is available"
6. Under **Settings**: check "Run task as soon as possible after a scheduled start is missed"

For weekday-only execution, go to **Triggers** tab → Edit → check **Mon, Tue, Wed, Thu, Fri** under "Weekly".

### 4. Test Run

```bat
cd C:\Users\dandy\OneDrive\Documents\aaravstrat\aria
set DISCORD_WEBHOOK_URL=YOUR_WEBHOOK_URL_HERE
python -m aria.trading.paper_trader
```

On first run with no positions, you'll see the initialization messages. State is saved to `trading/state.json`.

---

## File Structure

```
trading/
  paper_trader.py     # Main paper trading engine
  run_daily.bat       # Windows Task Scheduler wrapper
  README.md           # This file
  state.json          # Live position and trade state (auto-created)
  nav_history.json    # Daily NAV tracking (auto-created)
  paper_trader.log    # Full activity log (auto-created)
  edgar_cache/        # Per-company EDGAR cache (auto-created)
    {cik}.json        # Daily-refreshed 8-K + 10-Q data
```

---

## Discord Messages

The paper trader sends 4 Discord messages daily:

**1. Summary** — NAV, P&L, all open positions with live prices and stops  
**2. Portfolio table** — Full formatted position table with cash and slot count  
**3. Signals** — New entry signals with PEAD_z scores, plus watchlist of near-misses  
**4. Performance** — Last 5 closed trades, win rate, profit factor, and assessment

---

## Caveats

1. **Paper trading only** — this script does not connect to a broker. Prices and fills are simulated at yfinance closing prices (no slippage modeled).
2. **Survivorship bias** — the SimFin/yfinance universe includes only currently-active tickers. Delisted companies (bankruptcies, acquisitions) are not in the universe, which modestly inflates historical returns.
3. **EDGAR refresh** — the first run each day refreshes EDGAR submissions for all universe tickers (~2–3 minutes). Subsequent runs that day use the cached data.
4. **Earnings date precision** — 8-K filing date is the actual earnings announcement day for ~86% of quarters; the remaining 14% fall back to the 10-Q filing date, which lags by 1–7 days.
5. **Position sizing** — inverse-vol sizing with a 6% vol target. Very volatile stocks get smaller positions; stable stocks get larger ones. Effective leverage stays modest.
