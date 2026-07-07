# ARIA — Full Strategy Reference
*Transcribed from HTML strategy document. Gaps vs. current implementation noted throughout.*

---

## 01 — Strategy Overview

### Core Thesis

ARIA exploits a **three-part structural inefficiency** in U.S. large- and mid-cap equities:

1. **Analyst under-reaction** — Sell-side analysts systematically under-react in the immediate post-earnings window, creating a persistent, predictable drift in expectation revision paths.
2. **Options MM gamma unwinding** — Options market makers hedge their post-earnings implied volatility ("IV") exposure in a directionally predictable way as the event-risk premium collapses — a mechanical gamma-unwinding that creates transient price pressure decoupled from fundamentals.
3. **Institutional positioning constraints** — Index-tracking mandates, risk-budget depletion post-event, and sector rotation rules delay the absorption of new information into prices by **5 to 25 business days**.

ARIA sits at the intersection of these three dynamics: it is a **post-earnings drift strategy conditioned on a multi-dimensional regime classifier**, integrated with a real-time IV-surface signal and a residual institutional flow detector. The regime conditioning layer governs not just signal weights but position sizing, gross exposure, and sector concentration simultaneously.

> **Inefficiency Source:** The edge lives in the intersection of behavioral under-reaction (analysts), structural flow constraints (institutions), and mechanical volatility dynamics (options market makers). Each source is individually well-documented. The novelty lies in combining them into a single, regime-conditioned signal composite that dynamically re-weights based on market state.

---

### Why the Edge Persists

Post-earnings drift (PEAD) has been documented since 1968 (Ball & Brown). Despite this, it remains partially unexploited due to execution complexity in the window immediately following earnings. ARIA's version is harder to arbitrage than vanilla PEAD because:

| Differentiator | Description |
|---|---|
| **Regime Conditioning** | Signal weights and position sizes adapt to 4 distinct market regimes. A simple long earnings-surprise portfolio does not do this. The alpha is regime-conditional, not unconditional. |
| **IV Compression Signal** | Options IV collapse timing creates predictable price pressure. Most equity-only shops lack the cross-asset infrastructure to observe, clean, and integrate this signal at scale. |
| **Residual Flow Detection** | Detecting institutional selling/buying pressure through residual volume signatures requires tick-level data infrastructure beyond most systematic managers. |
| **Interaction Effects** | The alpha of the composite significantly exceeds the sum of individual signal alphas. Non-linear signal interactions (modeled via conditional entropy) are the key structural moat. |

---

## 02 — Economic & Behavioral Foundation

### Who Creates the Inefficiency

| Participant | Behavior | Inefficiency Created |
|---|---|---|
| **Sell-Side Analysts** | Anchoring to prior-period estimates; herding; updating incrementally over 2–6 weeks | Predictable upward/downward revision paths post-earnings |
| **Options Market Makers** | Unwind gamma hedges post-event as IV collapses from 70th to 30th percentile within 3–5 days | Mechanical, non-fundamental price pressure in 72-hour window |
| **Index/Factor Funds** | Rebalance on monthly/quarterly schedules; sector weights lag information by 30–90 days | Persistent price under-reaction to sector-relative earnings revisions |
| **Risk-Parity Funds** | Reduce equity exposure during earnings-driven volatility spikes; forced de-risking creates temporary supply | Indiscriminate selling in high-quality names during vol spikes |
| **Fundamental L/S Pods** | Capacity constraints force concentration; post-earnings exits create momentum continuation windows | Predictable flow patterns 5–15 days post-event in large positions |

### Primary Behavioral Biases

**Anchoring and Insufficient Adjustment (Tversky & Kahneman)**
Analysts anchor to prior-period estimates rather than the structural inflection implied by the earnings surprise. ARIA models this path using revision momentum velocity — the rate of change of consensus EPS estimates — not the level of the surprise.

**Herding Under Career Risk**
Sell-side analysts face asymmetric career risk from deviation from consensus. This creates clustering of revisions in the 3–6 week post-earnings window as analysts at a bridge bracket before revising. ARIA signals model this as a cross-sectional dispersion contraction in the 2–6 week window following a material surprise.

**Attention Allocation (Hirshleifer & Teoh)**
Investors have limited information bandwidth. During earnings seasons, 250–400 companies report within a 3-week window. Stocks reporting mid-cycle receive systematically less analyst and investor attention than first- or last-movers. ARIA overweights mid-cycle reporters in names with 20+ analyst coverage.

### Institutional Positioning Constraints

**Mandate Constraints**
Index and quasi-mutual funds with sector-neutral mandates cannot take concentrated bets on individual earnings outcomes. Post-earnings, they absorb information slowly through sector-level rebalancing. This creates a predictable 10–30 day window of price continuation in sector leaders following upside surprises.

**Liquidity-Adjusted Risk Budgets**
Most institutional risk budgets assess market impact risk based on 20-day trailing ADV. Post-earnings, a stock's ADV may temporarily spike 3–8× (event-driven activity), causing risk systems to underestimate true trading capacity. ARIA capitalizes on the post-spike ADV normalization by entering positions as risk systems re-engage with the name.

### Volatility Clustering and IV Dynamics

Implied volatility around earnings exhibits a **predictable term structure**: front-month IV inflates 3–10 days pre-earnings (event premium building), spikes at maximum 1–2 days prior, then collapses 40–70% within 24–72 hours post-announcement (the "vol crush"). Options market makers who were short gamma (long theta) during the event unwind this position, creating temporary directional pressure.

```
IV_crush(t) = IV_pre × exp(-λ × t)    where t ∈ [0, 3] days post-earnings

// λ ≈ 0.4–0.8 per day empirically; mean-reversion speed depends on sector vol regime

Net MM delta-hedge flow ≈ -Γ_agg × ΔS    where Γ_agg is aggregate gamma from all open strikes

// Positive Γ_agg → MMs buy on dips and sell on rallies (dampening)
// Negative Γ_agg → MMs amplify directional moves
```

### Crowding Risk and Reflexivity

ARIA faces non-trivial crowding risk in high-surprise-magnitude names where multiple systematic strategies may converge on identical long signals. The strategy monitors a **crowding coefficient** derived from short interest velocity, days-to-cover trends, and institutional ownership concentration. When crowding exceeds the 75th cross-sectional percentile, position size is capped at 40% of the unconstrained allocation.

> **Reflexivity Warning:** At sufficient AUM, ARIA itself becomes a market participant that reinforces the drift it exploits. Capacity is estimated at **$800M–$2B** before self-impact degrades the signal-to-noise ratio below acceptable thresholds. Above $1.5B AUM, an explicit market-impact model must be incorporated into the optimization objective function.

---

## 03 — Coverage Universe Construction

| Criterion | Threshold | Rationale |
|---|---|---|
| Market Capitalization | $2B (mid) to $50B+ (mega-cap) | Sufficient analyst coverage (≥4 estimates) for meaningful consensus; adequate liquidity for execution |
| Avg Daily Volume (ADV) | >$30M/day (30-day avg) | Position must be executable/re-adjustable within 3 trading days at ≤5% participation rate without material slippage |
| Analyst Coverage | 1–4 sell-side estimates | Minimum coverage for statistically valid consensus; below this, signals are noise-dominated |
| Options Market | Listed options with <$3k bid-ask IV spread | Required for IV compression signal; illiquid options markets produce unreliable IV term structure |
| Price | >$5/share | Exclude microcap/penny stocks; increasing availability and lot-size spread to degrade below this level |
| Earnings History | 6 consecutive quarterly reports | Required for proper z-score normalization; firms without earnings history below 20th percentile |
| Exclusions (Financials) | SIC 6000–6999 | Financial leverage-based earnings create structural model distortions |
| Exclusions (Utilities) | SIC 4900–4999 | Regulated earnings remove the predictive content of surprise (no genuine information revelation) |
| ADR Handling | Excluded (primary listing only) | ADRs have asymmetric and non-contemporaneous information flows that contaminate the FX-adjusted signal |
| Corporate Actions | Exclude ±30 days around M&A, spinoffs, secondary offerings | Event-driven price action dominates ARIA signals during corporate restructuring windows |

**Resulting universe: approximately 650–800 names** at any given time (Russell 1000 filtered). This provides sufficient cross-sectional depth for a 40–80 name portfolio with meaningful diversification.

### Capacity Analysis

At $1B AUM, 130/30 gross exposure ($1.3B long / $300M short), the average position size of $25–30M must be executable within 2 days against a universe with median ADV of ~$80M. This implies participation rates of 3–5% — well within institutional norms. The binding constraint is concentrated positions in mid-cap names (>$5B market cap), where ADV from $35M to $50M requires more careful execution scheduling.

---

## 04 — Signal Architecture

### Signal 1 — Earnings Surprise Quality Score (ESQS)

Standard SUE (Standardized Unexpected Earnings) normalizes the raw earnings surprise by a historical standard deviation. ARIA replaces this with a **quality-adjusted surprise** that decomposes the total EPS surprise into its components: revenue surprise, gross margin surprise, and SG&A efficiency surprise.

```
SUE_raw(i,t) = (EPS_actual - EPS_consensus) / σ_hist(EPS_surprise)

ESQS(i,t) = w1 × SUE_rev + w2 × SUE_gm + w3 × SUE_sgna - w4 × |SUE_below-line|

// w = [0.4, 0.35, 0.15, 0.10] empirically; below-line items penalized (tax/non-recurring)

Adjustment: ESQS_adj(i,t) = ESQS(i,t) × (1 - Guidance_Disappointment_Flag)

// Hard override: firms guiding below consensus have ESQS clipped to max 0
// regardless of beat magnitude
```

**Economic Intuition:** Revenue-driven beats reflect genuine demand acceleration; they have longer drift persistence (15–35 days) than below-the-line margin beats. A cost-cut-driven EPS beat with flat-to-declining revenue is a signal of deteriorating business quality masked by accounting, not a positive catalyst.

**Decay:** ESQS signal power decays approximately exponentially with a half-life of **18 trading days**. The dominant mechanism is analyst revision absorption — as consensus incorporates the new information, the price-to-fundamentals gap closes.

| Property | Value |
|---|---|
| Update Frequency | Daily |
| Half-Life | 18 trading days |
| Type | Fundamental |
| Window | Earnings Window |

> **Gap vs. ARIA current:** ARIA's implementation uses YoY same-quarter revenue as the consensus proxy (free data). The specification requires `EPS_consensus` time-series from a paid provider (Bloomberg, FactSet, Refinitiv). The `Guidance_Disappointment_Flag` is also not implemented. ARIA uses weights [0.30, 0.25, 0.20, 0.25] vs. specification weights [0.40, 0.35, 0.15, 0.10].

---

### Signal 2 — Implied Volatility Compression Asymmetry (IVCA)

Post-earnings IV collapse is not symmetric across the smile. The **skew** of the post-event IV surface encodes information about where options market makers hold residual exposure — specifically, whether the aggregate gamma position is concentrated in calls or puts.

```
IVCA(i,t) = [IV_25Δ call(t+1) - IV_25Δ put(t+1)] / IV_ATM(t+1)

// Positive IVCA → call-side IV remained elevated → MM net short calls → mechanical buy pressure

ΔIV_crush(i) = [IV_pre(i,-1d) - IV_post(i,+1d)] / IV_pre(i,-1d)

// Crush magnitude calibrates the size of the residual MM hedging flow

IVCA_adj = IVCA × ΔIV_crush × sign(ESQS)

// Only trade when IVCA reinforces fundamental surprise direction
```

**Economic Intuition:** When earnings surprise is strongly positive and the post-event put-call IV spread widens (puts remain expensive relative to calls), this reflects residual hedging demand from investors still holding downside protection — a sign the market has not yet fully re-rated upward. The MM must continue selling delta to maintain the hedge, creating sustained buying pressure.

**Decay:** IVCA is the fastest-decaying of all signals — 60–70% of the signal value evaporates within 3–5 trading days as gamma unwinds complete. Update frequency must be **daily intraday** with execution timed to the last 30 minutes of trading (when MM hedging is most active).

| Property | Value |
|---|---|
| Update Frequency | Intraday |
| Half-Life | 3 trading days |
| Type | Cross-Asset |
| Data | Options Surface |

> **Gap vs. ARIA current:** IVCA is entirely absent from ARIA. ARIA uses IVRS (realized vol ratio) as a position-size multiplier, which is a different concept. IVCA requires the full options IV surface — 25-delta call IV, 25-delta put IV, and ATM IV — which is paid data. This is the most data-intensive signal in the stack.

---

### Signal 3 — Revision Momentum Velocity (RMV)

Analyst revision momentum is well-known. RMV improves on it by measuring not just the direction of revisions but the **acceleration of the revision path** — analogous to measuring the second derivative of a price series rather than the first.

```
Rev_Δ(i,t) = [EPS_consensus(i,t) - EPS_consensus(i,t-5d)] / |EPS_consensus(i,t-5d)|

// 5-day revision delta, normalized by prior consensus level

RMV(i,t) = Rev_Δ(i,t) - Rev_Δ(i,t-5d)

// Second difference: is the revision rate accelerating or decelerating?

RMV_norm = z-score(RMV) within sector × time-period cohort

// Normalize within earnings season cohort to remove macro revision waves
```

**Economic Intuition:** An accelerating revision path (positive RMV) signals analyst herding onset — the critical mass necessary to drive institutional re-allocation is accumulating. A decelerating path (negative RMV) signals approaching consensus equilibrium, after which the drift window closes. The strategy enters on acceleration, exits on deceleration.

**Multicollinearity Risk:** RMV is moderately correlated with ESQS (ρ ≈ 0.35). **Residualization of RMV on ESQS is required** in the signal aggregation step to isolate the marginal information content of the revision path velocity beyond the initial surprise level.

| Property | Value |
|---|---|
| Update Frequency | Daily |
| Half-Life | 25 trading days |
| Type | Fundamental |
| Data | Revision-Based (paid) |

> **Gap vs. ARIA current:** ARIA's RMV module exists in code but is zeroed out entirely — no analyst consensus time-series is available from free data. The specification also requires residualizing RMV on ESQS before including it in the composite, which is not implemented. This is the second major paid-data dependency.

---

### Signal 4 — Institutional Flow Residual (IFR)

IFR estimates the **unexplained component of volume** after controlling for market-wide volume, sector volume, and stock-specific earnings-window volume seasonality. The residual is a proxy for institutional flow pressure — systematic buying or selling by large-allocation managers.

```
IFR(i,t) = Volume(i,t) - E[Volume(i,t) | Market, Sector, EarningsFlag, HistSeasonality]

// OLS or Gradient Boosting residual; trained rolling on 252-day window

IFR_sign(i,t) = IFR(i,t) × sign[VWAP(i,t) - Open(i,t)]

// Sign via intraday price direction: positive residual volume on up-day = buy pressure

IFR_agg(i) = Σ_{t=1..5} IFR_sign(i,t) × decay(t)    decay(t) = 0.85^(5-t)

// 5-day exponentially weighted accumulation
```

**Economic Intuition:** Institutional managers with multi-day implementation requirements leave predictable volume signatures. A large mutual fund rotating into a post-earnings winner must buy over 3–7 days, creating a detectable residual volume pattern that predicts continued buying pressure. This is a microstructure-derived version of "order flow imbalance."

**Implementation Complexity:** Requires tick-level intraday VWAP data. The model for expected volume must be re-estimated rolling daily and recalibrated at the start of each earnings season. The signal has meaningful implementation alpha — under-resourced managers cannot replicate it without substantial data engineering infrastructure.

| Property | Value |
|---|---|
| Update Frequency | Daily |
| Half-Life | 7 trading days |
| Type | Microstructure |
| Data | Flow-Based (tick-level VWAP) |

> **Gap vs. ARIA current:** ARIA's IFR uses daily close volume vs. sector ETF daily volume via OLS. The specification requires: (1) tick-level intraday VWAP to sign residuals (ARIA uses no signing), (2) additional controls for EarningsFlag and HistSeasonality in the expected-volume model (ARIA only controls for sector), (3) 5-day exponential accumulation of signed residuals (ARIA takes the latest single-day value). The signed and accumulated version is substantially more powerful.

---

### Signal 5 — Idiosyncratic Volatility Regime Signal (IVRS)

Stocks with *declining* idiosyncratic volatility in the 10–30 days post-earnings tend to exhibit stronger drift continuation. Volatility compression post-earnings reflects information uncertainty resolving — the market reaching consensus. Stocks with *elevated or re-expanding* idiosyncratic vol post-earnings signal continued disagreement, suppressing drift persistence.

```
ε(i,t) = r(i,t) - α - β×r_mkt(t) - Σγ_k×f_k(t)

// Barra 4-factor residual: strip market, sector, style, industry

IVOL(i,t) = σ[ε(i,t-21d:t)]    (21-day realized idiosyncratic vol)

IVRS(i,t) = -[IVOL(i,t) - IVOL(i,t-10d)] / IVOL(i,t-10d)

// Negative sign: declining IVOL → positive IVRS → reinforce drift long signals
```

**Risk Signal, Not Alpha Signal:** IVRS functions primarily as a **conviction multiplier** rather than a standalone alpha signal. It amplifies position sizes for names where uncertainty is resolving (IVRS > 0) and shrinks positions in names where post-earnings volatility is re-expanding. This is consistent with Ang, Hodrick, Xing & Zhang (2006) finding that high-IVOL stocks are systematically mispriced — but ARIA uses it conditionally rather than as a cross-sectional sort.

| Property | Value |
|---|---|
| Update Frequency | Daily |
| Type | Risk Modifier |
| Data | Volatility-Based |

> **Gap vs. ARIA current:** ARIA's IVRS computes a short/long realized vol ratio z-score from raw returns. The specification computes the idiosyncratic component using a Barra 4-factor model (market, sector, style, industry factors) before computing realized vol — this removes systematic vol from the signal so it measures only stock-specific uncertainty resolution, not market-wide vol shifts. The Barra factor loadings require a risk model (paid) or careful construction with free factor proxies.

---

### Signal 6 — Balance Sheet Quality Filter (BSQ)

BSQ is a **hard filter**, not an alpha signal. It eliminates firms with deteriorating financial quality from the long book and flags them as short candidates if they also exhibit negative ESQS.

```
BSQ(i,t) = f(Accruals, Altman-Z, Net Debt/EBITDA, FCF yield, Piotroski F-Score)

// Equal-weight composite of five standardized financial quality metrics

Accruals(i,t) = [NI(t) - CFO(t)] / AvgAssets(t)

// Sloan (1996): high accruals predict earnings disappointments 1–4 quarters ahead

Long eligibility:  BSQ > 30th percentile (cross-sectional)
Short eligibility: BSQ < 40th percentile AND ESQS < 0
```

| Property | Value |
|---|---|
| Update Frequency | Quarterly (on earnings release) |
| Type | Hard Filter |
| Data | Balance Sheet |

> **Gap vs. ARIA current:** BSQ is completely absent from ARIA. There is no balance sheet quality gate on the long book at all. This means ARIA's long book may include financially deteriorating firms that happen to beat on one quarter — a known risk that BSQ is designed to eliminate. All five components (Accruals, Altman-Z, Net Debt/EBITDA, FCF yield, Piotroski F-Score) are computable from SimFin quarterly income statement and balance sheet data.

---

## 05 — Gaps Between Friend's Specification and ARIA Implementation

### Signal-Level Gaps

| Component | Specification | ARIA Status | Severity |
|---|---|---|---|
| **ESQS consensus** | `EPS_consensus` from paid data provider | YoY same-quarter revenue (free proxy, low IC) | Critical |
| **ESQS weights** | [0.40, 0.35, 0.15, 0.10] | [0.30, 0.25, 0.20, 0.25] | Medium |
| **ESQS guidance flag** | Hard clip to 0 when guiding below consensus | Not implemented | High |
| **IVCA** | Full IV surface (25Δ call, put, ATM) + crush magnitude | Entirely absent (not built) | Critical |
| **RMV** | Second derivative of consensus EPS revisions, residualized on ESQS | Code exists but zeroed out (no paid data) | Critical |
| **IFR signing** | Signed by intraday VWAP direction per day | Not implemented (uses close volume only) | High |
| **IFR accumulation** | 5-day exponential decay accumulation | Not implemented (takes latest single value) | High |
| **IFR controls** | Market + Sector + EarningsFlag + HistSeasonality | Sector only | Medium |
| **IVRS factor model** | Barra 4-factor idiosyncratic residual | Raw return vol ratio (includes systematic vol) | Medium |
| **BSQ filter** | Hard eligibility gate on long and short book | Not implemented | High |
| **FTS signal** | Not in specification | Implemented (ARIA addition) | N/A |
| **HMM regime** | Not explicitly in specification | Implemented (ARIA addition) | N/A |

### Universe / Construction Gaps

| Component | Specification | ARIA Status |
|---|---|---|
| Min market cap | $2B | Not enforced (uses ADV only) |
| Financials exclusion | SIC 6000–6999 excluded | Not implemented |
| Utilities exclusion | SIC 4900–4999 excluded | Not implemented |
| ADR exclusion | Excluded (primary listing only) | Not implemented |
| Corporate actions window | ±30 days excluded | Not implemented |
| Options listing requirement | Listed options with <$3k IV spread required | Not implemented |
| Crowding coefficient | Cap at 40% when crowding > 75th percentile | Not implemented |
| Mid-cycle reporter overweight | Overweight when 20+ analyst coverage | Not implemented |

### Free-Data vs. Paid-Data Dependencies

Three of the six signals depend on paid data that ARIA does not have access to:

| Signal | Paid Data Required | Free Alternative Available? |
|---|---|---|
| ESQS | Analyst consensus EPS estimates | No — YoY comparison is a weak proxy with IC ≈ 0 |
| IVCA | Options IV surface (25Δ call/put, ATM) | Partial — some options data via yfinance but unreliable for 25Δ strikes |
| RMV | Time-series of consensus EPS (daily updates) | No — same as ESQS dependency |

The two signals that do not require paid data — IFR and IVRS — are partially implemented in ARIA but with gaps in signing, accumulation, and factor-model idiosyncratic decomposition.

### Root Cause of Underperformance

The strategy specification is built around analyst consensus EPS data as its primary input. ARIA's inability to access this data means the primary signal (ESQS) is measuring year-over-year growth (already priced) instead of earnings surprise relative to expectations (actionable edge). The measured IC of 0.012 at the 1-day horizon and -0.086 at the 5-day horizon confirms the proxy does not carry the intended signal content. Without this, the portfolio is effectively a random long/short book minus transaction costs.
