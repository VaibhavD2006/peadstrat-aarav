"""Load analyst consensus EPS data for SUE signal computation.

Expected CSV format (any paid provider — see docs for column mapping):
    ticker, report_date, fiscal_quarter_end, consensus_eps, actual_eps
    [optional: n_analysts, std_eps]

Where:
    report_date         — date the earnings were announced (announcement date)
    fiscal_quarter_end  — last day of the fiscal quarter being reported
    consensus_eps       — mean analyst EPS estimate prior to announcement
    actual_eps          — reported EPS
    n_analysts          — (optional) number of analyst estimates
    std_eps             — (optional) standard deviation of estimates; used as
                          normalizer instead of abs(consensus) when available
"""
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from aria.signals.sue import compute_sue_raw

_DEFAULT_CSV = Path("data/consensus/eps_consensus.csv")

# Column aliases from common providers so callers can pass raw provider exports.
# Each entry: (canonical_name, [provider_aliases...])
_COL_ALIASES: list[tuple[str, list[str]]] = [
    ("ticker",              ["symbol", "Ticker", "Symbol"]),
    ("report_date",         ["announcement_date", "reportDate", "date"]),
    ("fiscal_quarter_end",  ["period_end", "fiscalDateEnding", "periodEnd"]),
    ("consensus_eps",       ["estimated_eps", "epsEstimated", "estimatedEPS",
                             "consensus", "mean_eps"]),
    ("actual_eps",          ["actual", "epsActual", "actualEPS", "reportedEPS"]),
    ("n_analysts",          ["num_analysts", "numberOfAnalysts", "analyst_count"]),
    ("std_eps",             ["eps_std", "stdev", "stdDevEPS"]),
]


def _normalise_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename provider-specific column names to canonical names."""
    rename_map: dict[str, str] = {}
    existing = set(df.columns)
    for canonical, aliases in _COL_ALIASES:
        if canonical in existing:
            continue
        for alias in aliases:
            if alias in existing:
                rename_map[alias] = canonical
                break
    return df.rename(rename_map) if rename_map else df


def load_consensus(csv_path: Path = _DEFAULT_CSV) -> Optional[pl.DataFrame]:
    """Load consensus EPS data from CSV.

    Returns None if the file doesn't exist (graceful fallback — the runner
    will skip SUE computation rather than failing).

    Returns a Polars DataFrame with at minimum:
        [ticker, report_date, consensus_eps, actual_eps]
    and optionally:
        [fiscal_quarter_end, n_analysts, std_eps]
    """
    if not csv_path.exists():
        return None

    df = pl.read_csv(str(csv_path), try_parse_dates=True)
    df = _normalise_columns(df)

    required = {"ticker", "report_date", "consensus_eps", "actual_eps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Consensus CSV is missing required columns: {missing}. "
            f"Present columns: {list(df.columns)}"
        )

    # Cast report_date to pl.Date if it came in as a string
    if df["report_date"].dtype != pl.Date:
        df = df.with_columns(pl.col("report_date").cast(pl.Utf8).str.to_date(strict=False))

    df = df.drop_nulls(subset=["ticker", "report_date", "consensus_eps", "actual_eps"])
    return df.sort(["ticker", "report_date"])


def compute_historical_errors(
    ticker: str,
    as_of_date: date,
    consensus_df: pl.DataFrame,
    n_quarters: int = 4,
) -> list[float]:
    """Return list of (actual_eps - consensus_eps) for the n_quarters prior to as_of_date.

    Point-in-time safe: excludes rows with report_date >= as_of_date.
    Returns [] if no prior rows exist.
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


def compute_revision_dir(
    ticker: str,
    report_date: date,
    consensus_df: pl.DataFrame,
) -> float:
    """EPS revision direction proxy: (current_consensus - prior_year_consensus) / |prior|.

    Compares this quarter's analyst consensus against the same fiscal quarter
    one year ago (report_date ± 320-410 days). Returns float in [-1, 1]:
      positive = analysts raised expectations YoY (bullish signal)
      negative = analysts cut expectations YoY (bearish signal)
      0.0      = insufficient history
    """
    rows = consensus_df.filter(pl.col("ticker") == ticker).sort("report_date")

    current = rows.filter(pl.col("report_date") == report_date)
    if current.is_empty():
        return 0.0
    current_consensus = float(current["consensus_eps"][0])

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


def get_sue_inputs(
    tickers: list[str],
    as_of_date: date,
    consensus_df: pl.DataFrame,
) -> list[dict]:
    """For each ticker, find the most recent earnings announcement on or before
    as_of_date and compute the raw SUE value.

    Args:
        tickers:        Tickers to score (the event cohort).
        as_of_date:     Look-back cutoff; only use announcements ≤ this date.
        consensus_df:   Full consensus DataFrame from load_consensus().

    Returns:
        List of dicts [{ticker, sue_raw}] — one entry per ticker that has data.
        Tickers with no consensus history are silently omitted (SUE_z will be
        left at its default 0.0 in the base frame).
    """
    relevant = consensus_df.filter(
        pl.col("ticker").is_in(tickers) &
        (pl.col("report_date") <= as_of_date)
    )
    if relevant.is_empty():
        return []

    # Most recent announcement per ticker
    latest = (
        relevant
        .sort("report_date")
        .group_by("ticker")
        .tail(1)
    )

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

    return rows
