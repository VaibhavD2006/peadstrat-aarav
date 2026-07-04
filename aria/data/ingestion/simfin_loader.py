"""SimFin point-in-time income statement loader for ESQS signal inputs."""
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
    - The prior `trailing_quarters` rows are the trailing baseline for margins

    Returns dict with keys matching ESQSSignal.compute() expectations:
      ticker, revenue_actual, revenue_consensus,
      gross_margin_actual, gross_margin_trailing,
      sga_pct_actual, sga_pct_trailing,
      guidance_score (always None — caller can override)
    """
    try:
        ticker_df = income_df.xs(ticker, level="Ticker").reset_index()
    except KeyError:
        return None

    # Convert Publish Date to date objects
    ticker_df = ticker_df.copy()
    ticker_df["Publish Date"] = pd.to_datetime(ticker_df["Publish Date"]).dt.date

    # Only rows published by the cutoff date (point-in-time safe)
    available = ticker_df[ticker_df["Publish Date"] <= publish_date_cutoff].copy()
    available = available.sort_values("Report Date")

    if len(available) < trailing_quarters + 1:
        return None

    current = available.iloc[-1]
    trailing = available.iloc[-(trailing_quarters + 1):-1]

    # Revenue
    revenue_actual = float(current["Revenue"]) if pd.notna(current["Revenue"]) else None
    if revenue_actual is None or revenue_actual == 0:
        return None

    # Trailing revenue mean as pseudo-consensus
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

    # SG&A as % of revenue (SimFin stores SG&A as negative → take abs)
    sga_raw = current["Selling, General & Administrative"]
    sga_actual_abs = abs(float(sga_raw)) if pd.notna(sga_raw) else None
    sga_pct_actual = (sga_actual_abs / revenue_actual) if sga_actual_abs is not None else None
    if sga_pct_actual is None:
        sga_pct_actual = 0.10  # fallback

    def safe_sga_pct(row):
        sga_ = row["Selling, General & Administrative"]
        rev = row["Revenue"]
        if pd.notna(sga_) and pd.notna(rev) and rev != 0:
            return abs(sga_) / rev
        return np.nan

    trailing_sga = trailing.apply(safe_sga_pct, axis=1).dropna()
    sga_pct_trailing = float(trailing_sga.mean()) if len(trailing_sga) > 0 else sga_pct_actual

    return {
        "ticker": ticker,
        "revenue_actual": revenue_actual,
        "revenue_consensus": revenue_consensus,
        "gross_margin_actual": gross_margin_actual,
        "gross_margin_trailing": gross_margin_trailing,
        "sga_pct_actual": sga_pct_actual,
        "sga_pct_trailing": sga_pct_trailing,
        "guidance_score": None,
    }


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
