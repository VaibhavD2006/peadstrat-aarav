"""FTS - Filing Timeliness Signal.

A free-data proxy for analyst revision momentum (RMV).

Logic:
  1. Filing lateness: days between fiscal quarter-end and 10-Q filing date
     - On-time / early filer → positive (company confident in results)
     - Late filer → negative (often hiding bad news)
  2. YoY filing lateness trend: if the company filed earlier than same quarter
     prior year, that's an improvement signal
  3. Combined FTS score is cross-sectionally z-scored per publish date

Uses SEC EDGAR submissions API (free, no key required).
"""
import polars as pl
import numpy as np
from datetime import date, timedelta
from typing import Optional

from aria.signals.base import cross_sectional_zscore
from aria.data.ingestion.edgar_loader import get_cik_map, get_filing_history


# SEC 10-Q filing deadlines (calendar days after period end)
# Large accelerated filer: 40 days; accelerated: 40 days; non-accelerated: 45 days
_10Q_DEADLINE_DAYS = 40
_10K_DEADLINE_DAYS = 60  # Large accelerated: 60 days


class FTSSignal:
    """
    Filing Timeliness Signal — free proxy replacing RMV.

    Returns FTS_z: cross-sectional z-score of filing timeliness.
    Higher FTS_z = filed earlier relative to peers = bullish proxy.
    Lower FTS_z = filed later / past deadline = bearish proxy.
    """

    def __init__(self, lookback_quarters: int = 4):
        self.lookback_quarters = lookback_quarters
        self._cik_map: Optional[dict[str, int]] = None

    def _get_cik_map(self) -> dict[str, int]:
        if self._cik_map is None:
            self._cik_map = get_cik_map()
        return self._cik_map

    def _deadline_days(self, form_type: str) -> int:
        return _10K_DEADLINE_DAYS if form_type == "10-K" else _10Q_DEADLINE_DAYS

    def compute_ticker(
        self,
        ticker: str,
        as_of_date: date,
    ) -> Optional[float]:
        """
        Compute raw FTS score for one ticker as of as_of_date.

        Returns float (raw lateness score, lower = more bullish) or None.
        """
        cik_map = self._get_cik_map()
        filings = get_filing_history(ticker, cik_map)

        # Only use filings published before as_of_date
        available = [f for f in filings if f["filing_date"] <= as_of_date]
        if not available:
            return None

        # Most recent filing
        current = available[-1]
        deadline_days = self._deadline_days(current["form_type"])

        # Days-to-deadline: negative = filed early, positive = filed late
        lateness = (current["filing_date"] - current["period_end"]).days - deadline_days

        # YoY comparison: same quarter ~1 year ago
        one_year_ago = current["period_end"] - timedelta(days=365)
        prior_year = [
            f for f in available
            if abs((f["period_end"] - one_year_ago).days) <= 45
        ]
        yoy_improvement = 0.0
        if prior_year:
            prior = prior_year[-1]
            prior_lateness = (prior["filing_date"] - prior["period_end"]).days - deadline_days
            yoy_improvement = prior_lateness - lateness  # positive = filed earlier this year

        # Combined score: -lateness (earlier=better) + yoy_improvement
        raw_score = -lateness + 0.5 * yoy_improvement
        return float(raw_score)

    def compute_batch(
        self,
        tickers: list[str],
        as_of_date: date,
    ) -> pl.DataFrame:
        """
        Compute FTS for multiple tickers, returning cross-sectional z-scores.

        Returns DataFrame with [ticker, FTS_z].
        """
        rows = []
        for ticker in tickers:
            score = self.compute_ticker(ticker, as_of_date)
            if score is not None:
                rows.append({"ticker": ticker, "FTS_raw": score})

        if len(rows) < 3:
            return pl.DataFrame({
                "ticker": pl.Series([], dtype=pl.Utf8),
                "FTS_z":  pl.Series([], dtype=pl.Float64),
            })

        df = pl.DataFrame(rows)
        z = cross_sectional_zscore(df["FTS_raw"])
        return df.with_columns(z.alias("FTS_z")).select(["ticker", "FTS_z"])

    def compute_panel(
        self,
        tickers: list[str],
        dates: list[date],
    ) -> pl.DataFrame:
        """
        Compute FTS for a panel of (ticker, date) combinations.
        Groups by date for cross-sectional scoring.

        Returns DataFrame with [ticker, date, FTS_z].
        """
        all_rows = []
        for d in sorted(set(dates)):
            batch = self.compute_batch(tickers, d)
            if not batch.is_empty():
                batch = batch.with_columns(pl.lit(d).alias("date"))
                all_rows.append(batch)

        if not all_rows:
            return pl.DataFrame({
                "ticker": pl.Series([], dtype=pl.Utf8),
                "date":   pl.Series([], dtype=pl.Date),
                "FTS_z":  pl.Series([], dtype=pl.Float64),
            })

        return pl.concat(all_rows).with_columns(pl.col("date").cast(pl.Date))
