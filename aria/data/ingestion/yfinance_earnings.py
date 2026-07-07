"""Actual earnings announcement date loader via yfinance.

SimFin's 'Publish Date' is when SimFin ingested the report, typically 2-5 days
after the actual announcement. This module fetches real announcement dates from
yfinance and caches them locally, allowing PEAD to enter on day+1 instead of
day+4, recovering the IC that was being left on the table.
"""
import json
import time
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

CACHE_DIR = Path("data/earnings_dates")
_FETCH_DELAY = 0.3  # seconds between yfinance requests to avoid throttling


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.json"


def _load_from_cache(ticker: str) -> Optional[list[date]]:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return [date.fromisoformat(d) for d in data]
    except Exception:
        return None


def _save_to_cache(ticker: str, dates: list[date]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(ticker), "w", encoding="utf-8") as f:
        json.dump([str(d) for d in dates], f)


def _session_with_timeout(timeout: float = 5.0) -> requests.Session:
    """Return a requests Session whose every call honours the given timeout."""
    session = requests.Session()
    _orig = session.request
    def _request(*args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return _orig(*args, **kwargs)
    session.request = _request  # type: ignore[method-assign]
    return session


def fetch_earnings_dates(ticker: str, limit: int = 40, request_timeout: float = 5.0) -> list[date]:
    """Fetch actual earnings announcement dates for one ticker from yfinance.

    Filters to only CONFIRMED historical dates (rows where Reported EPS is
    not NaN). Future/estimated dates are excluded.  Uses request_timeout so
    delisted/slow tickers fail fast instead of waiting 30 s.

    Returns list of dates sorted ascending, or [] on error.
    """
    try:
        session = _session_with_timeout(request_timeout)
        t = yf.Ticker(ticker, session=session)
        df = t.get_earnings_dates(limit=limit)
        if df is None or df.empty:
            return []

        # Keep only confirmed reports (Reported EPS is not NaN)
        if "Reported EPS" in df.columns:
            df = df[df["Reported EPS"].notna()]

        if df.empty:
            return []

        # Index is DatetimeIndex (possibly timezone-aware)
        dates = sorted(set(
            idx.date() if hasattr(idx, "date") else idx.to_pydatetime().date()
            for idx in df.index
            if idx is not None and not pd.isna(idx)
        ))
        return dates
    except Exception:
        return []


def load_earnings_dates(
    tickers: list[str],
    limit: int = 40,
    delay: float = _FETCH_DELAY,
    request_timeout: float = 5.0,
    verbose: bool = True,
) -> dict[str, list[date]]:
    """Return {ticker: [announce_date, ...]} for each ticker.

    Reads from disk cache; fetches from yfinance only for tickers not cached.
    Dates are sorted ascending. Confirmed historical dates only.
    """
    result: dict[str, list[date]] = {}
    to_fetch = []

    for ticker in tickers:
        cached = _load_from_cache(ticker)
        if cached is not None:
            result[ticker] = cached
        else:
            to_fetch.append(ticker)

    if to_fetch and verbose:
        print(f"[yf_earnings] Fetching {len(to_fetch)} tickers from yfinance "
              f"({len(result)} already cached)...")

    for i, ticker in enumerate(to_fetch):
        dates = fetch_earnings_dates(ticker, limit=limit, request_timeout=request_timeout)
        _save_to_cache(ticker, dates)  # save even if empty so we don't re-fetch delisted tickers
        if dates:
            result[ticker] = dates
        if delay > 0 and i < len(to_fetch) - 1:
            time.sleep(delay)
        if verbose and (i + 1) % 50 == 0:
            pct = (i + 1) / len(to_fetch) * 100
            print(f"[yf_earnings]   {pct:.0f}% ({i+1}/{len(to_fetch)})")

    if to_fetch and verbose:
        n_ok = sum(1 for t in to_fetch if t in result)
        print(f"[yf_earnings] Done: {n_ok}/{len(to_fetch)} fetched, "
              f"{len(result)} total with dates")

    return result


def build_announce_map(
    ticker_pubdate_pairs: list[tuple[str, date]],
    yf_dates: dict[str, list[date]],
    window_days: int = 20,
) -> dict[tuple[str, date], date]:
    """Map each (ticker, simfin_publish_date) to the closest actual announcement date.

    Searches within ±window_days of the SimFin publish_date. Falls back to
    publish_date when no yfinance match is found.

    Returns {(ticker, publish_date): actual_announce_date}.
    """
    result: dict[tuple[str, date], date] = {}

    for ticker, pub_date in ticker_pubdate_pairs:
        best = pub_date  # fallback
        ticker_dates = yf_dates.get(ticker, [])
        if ticker_dates:
            best_delta = window_days + 1
            for d in ticker_dates:
                # yfinance date is usually a few days BEFORE publish_date
                delta = (pub_date - d).days  # positive = d is before pub_date
                if 0 <= delta <= window_days and delta < best_delta:
                    best_delta = delta
                    best = d
        result[(ticker, pub_date)] = best

    return result
