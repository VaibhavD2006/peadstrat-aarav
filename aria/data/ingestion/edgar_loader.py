"""SEC EDGAR data loader — CIK map and 10-Q/10-K filing metadata.

Uses the free EDGAR XBRL/submissions API (no API key required).
Rate limit: 10 requests/second per SEC fair-use policy.
"""
import json
import time
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

CACHE_DIR = Path("data/edgar")
CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
HEADERS = {"User-Agent": "ARIA-Research aria@aaravstrat.com"}
_REQUEST_INTERVAL = 0.11  # ~9 req/s, safely under 10/s


def _fetch(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
            else:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def get_cik_map(cache_days: int = 7) -> dict[str, int]:
    """Return {ticker: cik_int} from SEC company_tickers.json, cached locally."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "cik_map.json"

    if cache_path.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days
        if age_days < cache_days:
            with open(cache_path) as f:
                return json.load(f)

    raw = _fetch(CIK_MAP_URL)
    # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": ...}, ...}
    cik_map = {}
    for entry in raw.values():
        ticker = entry.get("ticker", "").upper().strip()
        cik = entry.get("cik_str")
        if ticker and cik:
            cik_map[ticker] = int(cik)

    with open(cache_path, "w") as f:
        json.dump(cik_map, f)
    return cik_map


def get_filing_history(
    ticker: str,
    cik_map: Optional[dict[str, int]] = None,
    form_types: tuple[str, ...] = ("10-Q", "10-K"),
    cache_days: int = 30,
) -> list[dict]:
    """
    Return list of filing dicts for a ticker, sorted by filing date ascending.

    Each dict has:
      filing_date: date
      form_type:   str  (10-Q or 10-K)
      period_end:  date  (period of report)
      accession:   str
    """
    if cik_map is None:
        cik_map = get_cik_map()

    cik = cik_map.get(ticker.upper())
    if cik is None:
        return []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker.upper()}_{cik}.json"

    if cache_path.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days
        if age_days < cache_days:
            with open(cache_path) as f:
                raw = json.load(f)
                return _parse_filings(raw, form_types)

    time.sleep(_REQUEST_INTERVAL)
    try:
        raw = _fetch(SUBMISSIONS_URL.format(cik=cik))
    except Exception:
        return []

    with open(cache_path, "w") as f:
        json.dump(raw, f)

    return _parse_filings(raw, form_types)


def _parse_filings(raw: dict, form_types: tuple[str, ...]) -> list[dict]:
    filings_block = raw.get("filings", {}).get("recent", {})
    forms       = filings_block.get("form", [])
    filed_dates = filings_block.get("filingDate", [])
    periods     = filings_block.get("reportDate", [])
    accessions  = filings_block.get("accessionNumber", [])

    results = []
    for form, filed, period, acc in zip(forms, filed_dates, periods, accessions):
        if form not in form_types:
            continue
        try:
            fd = date.fromisoformat(filed)
            pd_ = date.fromisoformat(period) if period else fd
        except ValueError:
            continue
        results.append({
            "filing_date": fd,
            "form_type":   form,
            "period_end":  pd_,
            "accession":   acc,
        })

    return sorted(results, key=lambda x: x["filing_date"])


def get_sic_map() -> dict[str, int]:
    """Return {ticker: sic_code} from cached EDGAR submissions files.

    Reads SIC codes from submissions JSONs already cached by get_filing_history().
    Tickers not yet cached are omitted — caller should treat missing as unknown (no filter).
    """
    sic_map: dict[str, int] = {}
    if not CACHE_DIR.exists():
        return sic_map
    for fpath in CACHE_DIR.glob("*.json"):
        if fpath.name == "cik_map.json":
            continue
        # Files are named "{TICKER}_{CIK}.json"
        ticker = fpath.stem.split("_")[0].upper()
        try:
            with open(fpath) as f:
                raw = json.load(f)
            sic = raw.get("sic")
            if sic is not None:
                sic_map[ticker] = int(sic)
        except Exception:
            continue
    return sic_map


def get_filing_lateness_days(
    ticker: str,
    period_end: date,
    cik_map: Optional[dict[str, int]] = None,
) -> Optional[int]:
    """
    Return how many days after period_end the 10-Q/10-K was filed.

    SEC deadline: 10-Q due 40 days after quarter-end (large accelerated filer).
    Negative = early, positive = late (past deadline), None = not found.
    """
    filings = get_filing_history(ticker, cik_map)
    # Find the filing whose period_end matches (within ±15 days)
    for f in filings:
        delta = abs((f["period_end"] - period_end).days)
        if delta <= 15 and f["form_type"] in ("10-Q", "10-K"):
            lateness = (f["filing_date"] - period_end).days
            return lateness
    return None


def batch_filing_metadata(
    tickers: list[str],
    start_date: date,
    end_date: date,
    cik_map: Optional[dict[str, int]] = None,
) -> dict[str, list[dict]]:
    """
    Return {ticker: [filing_dicts]} for all tickers, filings within [start_date, end_date].
    Respects EDGAR rate limit between requests.
    """
    if cik_map is None:
        cik_map = get_cik_map()

    result = {}
    for ticker in tickers:
        filings = get_filing_history(ticker, cik_map)
        in_range = [
            f for f in filings
            if start_date <= f["filing_date"] <= end_date
        ]
        if in_range:
            result[ticker] = in_range
    return result
