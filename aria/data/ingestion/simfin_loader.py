"""SimFin point-in-time loaders for income, balance sheet, and cash flow data."""
import os
import simfin as sf
import pandas as pd
import numpy as np
from datetime import date
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def _init_simfin():
    api_key = os.environ.get("SIMFIN_API_KEY", "")
    sf.set_api_key(api_key)
    sf.set_data_dir("data/simfin")


def load_income_statements(market: str = "us") -> pd.DataFrame:
    """Load cached SimFin quarterly income statements.

    Returns pandas DataFrame indexed by (Ticker, Report Date).
    Uses refresh_days=3650 so it always reads from disk cache.
    """
    _init_simfin()
    df = sf.load_income(variant="quarterly", market=market, refresh_days=3650)
    return df


def _safe_eps(row: pd.Series) -> float:
    """Compute EPS proxy: Net Income / Shares Basic."""
    ni = row.get("Net Income")
    shares = row.get("Shares (Basic)")
    if pd.notna(ni) and pd.notna(shares) and shares > 0:
        return float(ni) / float(shares)
    return np.nan


def get_esqs_inputs(
    ticker: str,
    publish_date_cutoff: date,
    income_df: pd.DataFrame,
    trailing_quarters: int = 4,
) -> Optional[dict]:
    """Return ESQS input dict for a ticker at a point-in-time date, or None.

    Logic:
    - Get all rows for ticker where Publish Date <= publish_date_cutoff
    - Sort by Report Date ascending
    - The most recent row is the "current" quarter (the one we are scoring)
    - Use the same-quarter prior year as revenue_consensus (YoY comparison)
      which correctly removes seasonality vs. the old trailing-mean approach
    - Add eps_yoy: YoY EPS change (replaces unused guidance_score)

    Returns dict with keys matching ESQSSignal.compute() expectations:
      ticker, revenue_actual, revenue_consensus (YoY),
      gross_margin_actual, gross_margin_trailing,
      sga_pct_actual, sga_pct_trailing,
      eps_yoy (float or 0.0)
    """
    try:
        ticker_df = income_df.xs(ticker, level="Ticker").reset_index()
    except KeyError:
        return None

    ticker_df = ticker_df.copy()
    ticker_df["Publish Date"] = pd.to_datetime(ticker_df["Publish Date"]).dt.date
    ticker_df["Report Date"]  = pd.to_datetime(ticker_df["Report Date"]).dt.date

    # Only rows published by the cutoff date (point-in-time safe)
    available = ticker_df[ticker_df["Publish Date"] <= publish_date_cutoff].copy()
    available = available.sort_values("Report Date").reset_index(drop=True)

    if len(available) < trailing_quarters + 1:
        return None

    current = available.iloc[-1]
    trailing = available.iloc[-(trailing_quarters + 1):-1]

    # Revenue
    revenue_actual = float(current["Revenue"]) if pd.notna(current["Revenue"]) else None
    if revenue_actual is None or revenue_actual == 0:
        return None

    # YoY revenue as consensus: same fiscal quarter ~1 year ago
    # Find the row whose Report Date is closest to (current_report_date - 365 days)
    current_report_date = current["Report Date"]
    one_year_ago = pd.Timestamp(current_report_date) - pd.DateOffset(days=365)
    available["report_ts"] = pd.to_datetime(available["Report Date"])
    date_diffs = (available["report_ts"] - one_year_ago).abs()

    # Only consider rows earlier than current
    prior_rows = available.iloc[:-1].copy()
    prior_rows["report_ts"] = pd.to_datetime(prior_rows["Report Date"])
    prior_date_diffs = (prior_rows["report_ts"] - one_year_ago).abs()

    if len(prior_rows) > 0 and prior_date_diffs.min().days <= 45:
        yoy_row = prior_rows.loc[prior_date_diffs.idxmin()]
        yoy_revenue = float(yoy_row["Revenue"]) if pd.notna(yoy_row["Revenue"]) else None
        if yoy_revenue and yoy_revenue != 0:
            revenue_consensus = yoy_revenue
        else:
            # Fallback to trailing mean if YoY row is missing
            trailing_rev = trailing["Revenue"].dropna()
            revenue_consensus = float(trailing_rev.mean()) if len(trailing_rev) > 0 else revenue_actual
    else:
        trailing_rev = trailing["Revenue"].dropna()
        if len(trailing_rev) == 0:
            return None
        revenue_consensus = float(trailing_rev.mean())

    # Gross margin: Gross Profit / Revenue
    gp = current["Gross Profit"]
    gross_margin_actual = float(gp / revenue_actual) if pd.notna(gp) and revenue_actual != 0 else None
    if gross_margin_actual is None:
        return None

    def safe_gm(row):
        rev = row["Revenue"]
        gp_ = row["Gross Profit"]
        if pd.notna(rev) and pd.notna(gp_) and rev != 0:
            return gp_ / rev
        return np.nan

    trailing_gm = trailing.apply(safe_gm, axis=1).dropna()
    gross_margin_trailing = float(trailing_gm.mean()) if len(trailing_gm) > 0 else gross_margin_actual

    # SG&A as % of revenue (SimFin stores SG&A as negative -> take abs)
    sga_raw = current["Selling, General & Administrative"]
    sga_actual_abs = abs(float(sga_raw)) if pd.notna(sga_raw) else None
    sga_pct_actual = (sga_actual_abs / revenue_actual) if sga_actual_abs is not None else 0.10

    def safe_sga_pct(row):
        sga_ = row["Selling, General & Administrative"]
        rev = row["Revenue"]
        if pd.notna(sga_) and pd.notna(rev) and rev != 0:
            return abs(sga_) / rev
        return np.nan

    trailing_sga = trailing.apply(safe_sga_pct, axis=1).dropna()
    sga_pct_trailing = float(trailing_sga.mean()) if len(trailing_sga) > 0 else sga_pct_actual

    # EPS YoY: (current EPS - prior year EPS) / |prior year EPS|
    eps_current = _safe_eps(current)
    eps_yoy = 0.0
    if len(prior_rows) > 0 and prior_date_diffs.min().days <= 45:
        yoy_row = prior_rows.loc[prior_date_diffs.idxmin()]
        eps_prior = _safe_eps(yoy_row)
        if not np.isnan(eps_current) and not np.isnan(eps_prior) and eps_prior != 0:
            eps_yoy = float((eps_current - eps_prior) / abs(eps_prior))
            eps_yoy = float(np.clip(eps_yoy, -2.0, 2.0))  # winsorise

    return {
        "ticker": ticker,
        "revenue_actual": revenue_actual,
        "revenue_consensus": revenue_consensus,
        "gross_margin_actual": gross_margin_actual,
        "gross_margin_trailing": gross_margin_trailing,
        "sga_pct_actual": sga_pct_actual,
        "sga_pct_trailing": sga_pct_trailing,
        "eps_yoy": eps_yoy,
    }


def load_balance_sheets(market: str = "us") -> pd.DataFrame:
    """Load cached SimFin quarterly balance sheets.

    Returns pandas DataFrame indexed by (Ticker, Report Date).
    Uses refresh_days=3650 so it always reads from disk cache.
    """
    _init_simfin()
    return sf.load_balance(variant="quarterly", market=market, refresh_days=3650)


def load_cash_flows(market: str = "us") -> pd.DataFrame:
    """Load cached SimFin quarterly cash flow statements.

    Returns pandas DataFrame indexed by (Ticker, Report Date).
    Uses refresh_days=3650 so it always reads from disk cache.
    """
    _init_simfin()
    return sf.load_cashflow(variant="quarterly", market=market, refresh_days=3650)


def _get_pit_rows(
    df: pd.DataFrame,
    ticker: str,
    publish_date_cutoff: date,
) -> Optional[pd.DataFrame]:
    """Extract point-in-time available rows for ticker from a SimFin DataFrame."""
    try:
        ticker_df = df.xs(ticker, level="Ticker").reset_index().copy()
    except KeyError:
        return None
    ticker_df["Publish Date"] = pd.to_datetime(ticker_df["Publish Date"]).dt.date
    ticker_df["Report Date"] = pd.to_datetime(ticker_df["Report Date"]).dt.date
    available = ticker_df[ticker_df["Publish Date"] <= publish_date_cutoff].copy()
    available = available.sort_values("Report Date").reset_index(drop=True)
    return available if len(available) > 0 else None


def get_bsq_inputs(
    ticker: str,
    publish_date_cutoff: date,
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
) -> Optional[dict]:
    """Return BSQ component dict for a ticker at a point-in-time date, or None.

    Components:
      accruals     = -(NI - CFO) / avg_total_assets  (negated: lower accruals = better)
      cfo_margin   = CFO / Revenue                   (higher = better)
      debt_burden  = -Net Debt / EBITDA              (negated: lower leverage = better)
      cash_quality = 1 if CFO > NI else 0            (binary: cash backing of earnings)
    """
    inc = _get_pit_rows(income_df, ticker, publish_date_cutoff)
    bal = _get_pit_rows(balance_df, ticker, publish_date_cutoff)
    cf = _get_pit_rows(cashflow_df, ticker, publish_date_cutoff)

    if inc is None or len(inc) < 1:
        return None

    current_inc = inc.iloc[-1]

    def _safe(row, col):
        v = row.get(col)
        return float(v) if pd.notna(v) else None

    # Net Income from income statement
    ni = _safe(current_inc, "Net Income")
    revenue = _safe(current_inc, "Revenue")
    op_income = _safe(current_inc, "Operating Income (Loss)")
    da = _safe(current_inc, "Depreciation & Amortization")

    components = {}

    # CFO from cash flow statement (most recent row aligned to same quarter)
    if cf is not None and len(cf) > 0:
        current_cf = cf.iloc[-1]
        cfo = _safe(current_cf, "Net Cash from Operating Activities")

        if ni is not None and cfo is not None:
            # Accruals: lower = better earnings quality (Sloan 1996)
            if bal is not None and len(bal) > 0:
                cur_assets = _safe(bal.iloc[-1], "Total Assets")
                prior_assets = _safe(bal.iloc[-2], "Total Assets") if len(bal) > 1 else cur_assets
                if cur_assets is not None and prior_assets is not None:
                    avg_assets = (cur_assets + prior_assets) / 2
                    if avg_assets > 0:
                        components["accruals"] = -(ni - cfo) / avg_assets

            # CFO margin: higher = more cash-generative
            if revenue is not None and revenue > 0:
                components["cfo_margin"] = cfo / revenue

            # Cash quality flag: binary
            components["cash_quality"] = 1.0 if cfo > ni else 0.0

    # Debt burden from balance sheet
    if bal is not None and len(bal) > 0:
        cur_bal = bal.iloc[-1]
        lt_debt = _safe(cur_bal, "Long Term Debt") or 0.0
        st_debt = _safe(cur_bal, "Short Term Debt") or 0.0
        cash = _safe(cur_bal, "Cash & Equivalents") or 0.0
        total_debt = lt_debt + st_debt
        net_debt = total_debt - cash

        ebitda = None
        if op_income is not None and da is not None:
            ebitda = op_income + da
        if ebitda is not None and ebitda > 0 and net_debt is not None:
            components["debt_burden"] = -net_debt / ebitda

    if not components:
        return None

    return {"ticker": ticker, **components}


def build_esqs_batch(
    tickers: list[str],
    publish_date_cutoff: date,
    income_df: Optional[pd.DataFrame] = None,
) -> list[dict]:
    """Build ESQS input rows for a batch of tickers at a given point-in-time date."""
    if income_df is None:
        income_df = load_income_statements()
    rows = []
    for ticker in tickers:
        result = get_esqs_inputs(ticker, publish_date_cutoff, income_df)
        if result is not None:
            rows.append(result)
    return rows
