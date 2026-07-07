"""
SEC EDGAR filing date loader.

Fetches 10-Q / 10-K filing dates from the SEC EDGAR submissions API and
builds an event_by_date dict compatible with the phase3_runner.

v1 (build_event_by_date): uses 10-Q filing dates as event dates.
v2 (build_event_by_date_8k): uses the earliest earnings-release 8-K filed
  within 15-55 days after quarter-end.  8-K filings are filed on the actual
  earnings announcement day, giving a clean PEAD signal.  Falls back to
  the 10-Q date when no matching 8-K is found.

All results are cached under data/edgar/ so network round-trips only
happen once (~15-20 min for the full 3,787-ticker universe).
"""

import json
import time
import pathlib
import urllib.request
import urllib.error
from datetime import date, timedelta
from typing import Optional

# SEC requires a descriptive User-Agent with contact email
_HEADERS = {"User-Agent": "ARIA Research dandybee06@gmail.com"}
_SEC_BASE = "https://www.sec.gov"
_DATA_BASE = "https://data.sec.gov"
_RATE_DELAY = 0.12          # 8.3 req/sec — safely under the 10/sec limit
_TIMEOUT = 30
_RETRIES = 3


def _get(url: str) -> dict:
    """Rate-limited JSON fetch with retry."""
    last_exc: Exception = RuntimeError("no attempt")
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {}          # company not found — not a retry-able error
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        time.sleep(_RATE_DELAY * (attempt + 1))
    raise last_exc


def _load_json(path: pathlib.Path) -> Optional[list]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_json(path: pathlib.Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class EdgarLoader:
    """
    Load SEC filing dates for a set of tickers.

    v1 — 10-Q/10-K dates only::

        loader = EdgarLoader()
        event_by_date = loader.build_event_by_date(tickers, date(2009,1,1), date(2024,12,31))

    v2 — 8-K announcement dates (preferred for PEAD)::

        event_by_date = loader.build_event_by_date_8k(tickers, date(2009,1,1), date(2024,12,31))
    """

    def __init__(self, cache_dir: str = "data/edgar"):
        self._cache = pathlib.Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._cik_map: Optional[dict[str, str]] = None

    # ------------------------------------------------------------------
    # CIK lookup
    # ------------------------------------------------------------------

    def _ensure_cik_map(self) -> dict[str, str]:
        if self._cik_map is not None:
            return self._cik_map
        cache_file = self._cache / "ticker_cik_map.json"
        cached = _load_json(cache_file)
        if cached is not None:
            self._cik_map = cached
            return self._cik_map

        print("[Edgar] Downloading ticker->CIK map from SEC...")
        data = _get(f"{_SEC_BASE}/files/company_tickers.json")
        cik_map: dict[str, str] = {}
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper().strip()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if ticker:
                cik_map[ticker] = cik
        _save_json(cache_file, cik_map)
        self._cik_map = cik_map
        print(f"[Edgar] CIK map: {len(cik_map):,} tickers")
        return cik_map

    def get_cik(self, ticker: str) -> Optional[str]:
        return self._ensure_cik_map().get(ticker.upper())

    # ------------------------------------------------------------------
    # v1: 10-Q / 10-K submissions (original)
    # ------------------------------------------------------------------

    def _fetch_submissions(self, cik: str) -> list[dict]:
        """
        Return list of {form, filingDate, reportDate} for 10-Q/10-K filings.
        Cached in data/edgar/filings/{cik}.json.
        """
        cache_file = self._cache / "filings" / f"{cik}.json"
        cached = _load_json(cache_file)
        if cached is not None:
            return cached

        all_rows: list[dict] = []

        def _parse_batch(batch: dict) -> None:
            forms = batch.get("form", [])
            dates = batch.get("filingDate", [])
            reports = batch.get("reportDate", [])
            for form, fd, rd in zip(forms, dates, reports):
                if form in ("10-Q", "10-K"):
                    all_rows.append({"form": form, "filingDate": fd, "reportDate": rd})

        primary = _get(f"{_DATA_BASE}/submissions/CIK{cik}.json")
        if not primary:
            _save_json(cache_file, [])
            return []

        recent = primary.get("filings", {}).get("recent", {})
        _parse_batch(recent)

        for extra_meta in primary.get("filings", {}).get("files", []):
            fname = extra_meta.get("name", "")
            if not fname:
                continue
            time.sleep(_RATE_DELAY)
            extra = _get(f"{_DATA_BASE}/submissions/{fname}")
            _parse_batch(extra)

        _save_json(cache_file, all_rows)
        return all_rows

    def build_event_by_date(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[date, list[str]]:
        """
        Build {filing_date: [ticker, ...]} using 10-Q/10-K filing dates (v1).

        Per-company data cached in data/edgar/filings/{cik}.json.
        """
        cik_map = self._ensure_cik_map()
        rows: list[dict] = []
        n = len(tickers)
        fetched = 0

        for i, ticker in enumerate(tickers):
            cik = cik_map.get(ticker.upper())
            if not cik:
                continue

            cache_file = self._cache / "filings" / f"{cik}.json"
            already_cached = cache_file.exists()

            if not already_cached:
                fetched += 1
                if fetched % 100 == 1:
                    print(f"[Edgar] Fetching submissions {i}/{n} (net)...")
            try:
                submissions = self._fetch_submissions(cik)
                if not already_cached:
                    time.sleep(_RATE_DELAY)
            except Exception as exc:
                print(f"[Edgar] {ticker} ({cik}): {exc}")
                continue

            for sub in submissions:
                fd_str = sub.get("filingDate", "")
                if not fd_str:
                    continue
                try:
                    fd = date.fromisoformat(fd_str)
                except ValueError:
                    continue
                if start_date <= fd <= end_date:
                    rows.append({"ticker": ticker, "filing_date": fd})

        if not rows:
            return {}

        seen: dict[date, set[str]] = {}
        result: dict[date, list[str]] = {}
        for r in rows:
            d, t = r["filing_date"], r["ticker"]
            if t not in seen.setdefault(d, set()):
                seen[d].add(t)
                result.setdefault(d, []).append(t)

        print(f"[Edgar] {len(result)} event dates, {len(rows)} ticker-date rows")
        return result

    # ------------------------------------------------------------------
    # v2: 8-K announcement dates (preferred for PEAD signal)
    # ------------------------------------------------------------------

    def _fetch_submissions_v2(self, cik: str) -> list[dict]:
        """
        Return list of {form, filingDate, reportDate} for 10-Q/10-K/8-K.
        Cached in data/edgar/v2/{cik}.json — separate from v1 cache.
        """
        cache_file = self._cache / "v2" / f"{cik}.json"
        cached = _load_json(cache_file)
        if cached is not None:
            return cached

        all_rows: list[dict] = []

        def _parse_batch(batch: dict) -> None:
            forms = batch.get("form", [])
            dates = batch.get("filingDate", [])
            reports = batch.get("reportDate", [])
            for form, fd, rd in zip(forms, dates, reports):
                if form in ("10-Q", "10-K", "8-K"):
                    all_rows.append({"form": form, "filingDate": fd, "reportDate": rd or ""})

        primary = _get(f"{_DATA_BASE}/submissions/CIK{cik}.json")
        if not primary:
            _save_json(cache_file, [])
            return []

        recent = primary.get("filings", {}).get("recent", {})
        _parse_batch(recent)

        for extra_meta in primary.get("filings", {}).get("files", []):
            fname = extra_meta.get("name", "")
            if not fname:
                continue
            time.sleep(_RATE_DELAY)
            extra = _get(f"{_DATA_BASE}/submissions/{fname}")
            _parse_batch(extra)

        _save_json(cache_file, all_rows)
        return all_rows

    def build_event_by_date_8k(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
        min_days_after_qtr: int = 15,
        max_days_after_qtr: int = 55,
    ) -> dict[date, list[str]]:
        """
        Build {announcement_date: [ticker, ...]} using earnings 8-K dates (v2).

        For each 10-Q filing, finds the earliest 8-K filed within
        [reportDate + min_days_after_qtr, reportDate + max_days_after_qtr].
        This 8-K is the earnings release filed on the actual announcement day.
        Falls back to the 10-Q filing date when no matching 8-K is found.

        Per-company data cached in data/edgar/v2/{cik}.json.
        """
        cik_map = self._ensure_cik_map()
        rows: list[dict] = []
        n_8k_used = 0
        n_fallback = 0
        n = len(tickers)
        fetched = 0

        for i, ticker in enumerate(tickers):
            cik = cik_map.get(ticker.upper())
            if not cik:
                continue

            cache_file = self._cache / "v2" / f"{cik}.json"
            already_cached = cache_file.exists()

            if not already_cached:
                fetched += 1
                if fetched % 100 == 1:
                    print(f"[Edgar/8K] Fetching submissions {i}/{n} (net)...")
            try:
                submissions = self._fetch_submissions_v2(cik)
                if not already_cached:
                    time.sleep(_RATE_DELAY)
            except Exception as exc:
                print(f"[Edgar/8K] {ticker} ({cik}): {exc}")
                continue

            # Parse into quarterly filings and 8-K dates
            quarterly: list[tuple[date, date]] = []  # (filingDate, reportDate)
            eightk_dates: list[date] = []

            for sub in submissions:
                form = sub.get("form", "")
                fd_str = sub.get("filingDate", "")
                rd_str = sub.get("reportDate", "")
                try:
                    fd = date.fromisoformat(fd_str)
                except (ValueError, TypeError):
                    continue

                if form == "8-K":
                    eightk_dates.append(fd)
                elif form in ("10-Q", "10-K") and rd_str:
                    try:
                        rd = date.fromisoformat(rd_str)
                        quarterly.append((fd, rd))
                    except (ValueError, TypeError):
                        pass

            # For each quarterly filing, find the announcement date
            for q_filing_date, q_report_date in quarterly:
                win_start = q_report_date + timedelta(days=min_days_after_qtr)
                win_end = q_report_date + timedelta(days=max_days_after_qtr)

                matching = sorted(d for d in eightk_dates if win_start <= d <= win_end)
                if matching:
                    announcement_date = matching[0]
                    n_8k_used += 1
                else:
                    announcement_date = q_filing_date
                    n_fallback += 1

                if start_date <= announcement_date <= end_date:
                    rows.append({"ticker": ticker, "filing_date": announcement_date})

        if not rows:
            return {}

        seen: dict[date, set[str]] = {}
        result: dict[date, list[str]] = {}
        for r in rows:
            d, t = r["filing_date"], r["ticker"]
            if t not in seen.setdefault(d, set()):
                seen[d].add(t)
                result.setdefault(d, []).append(t)

        total = n_8k_used + n_fallback
        pct = 100 * n_8k_used / total if total else 0
        print(
            f"[Edgar/8K] {len(result)} event dates, {len(rows)} ticker-date rows "
            f"({pct:.0f}% used 8-K dates, {n_fallback} fell back to 10-Q dates)"
        )
        return result
