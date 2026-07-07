"""PEAD — Post-Earnings Announcement Drift signal.

Empirical edge: stocks continue drifting in the direction of their 1-day
earnings reaction for 20–40 trading days (Ball & Brown 1968; Bernard & Thomas 1989).
IC vs. 20-day forward return is typically 0.08–0.15 in large-cap universes.

Signal construction:
  earnings_reaction(i) = (close_day0 - close_day_minus1) / close_day_minus1
  PEAD_z = cross-sectional z-score of earnings_reaction within each announce-date cohort

Entry: announce_date + 1 trading day (capture drift, not the gap itself).
Hold:  20 trading days recommended.
"""
import numpy as np
import polars as pl
from datetime import date, timedelta
from typing import Optional

from aria.signals.base import cross_sectional_zscore


class PEADSignal:
    """Post-Earnings Announcement Drift signal.

    Computes the 1-day earnings reaction for each ticker and cross-sectionally
    z-scores it within each announce-date cohort. Positive PEAD_z = stock
    gapped up on earnings → go long on day+1. Negative → go short.
    """

    def _find_price(
        self,
        ticker_prices: pl.DataFrame,
        target_date: date,
        search_forward: bool = False,
    ) -> Optional[float]:
        """Return close price on target_date, or None if not found.

        If search_forward=True, looks for the next available trading day
        at or after target_date (up to 5 calendar days ahead).
        """
        exact = ticker_prices.filter(pl.col("date") == target_date)
        if not exact.is_empty():
            return float(exact["close"][0])
        if not search_forward:
            return None
        for offset in range(1, 6):
            shifted = ticker_prices.filter(pl.col("date") == target_date + timedelta(days=offset))
            if not shifted.is_empty():
                return float(shifted["close"][0])
        return None

    def _prior_trading_close(
        self, ticker_prices: pl.DataFrame, before_date: date
    ) -> Optional[float]:
        """Return the close price on the last trading day strictly before before_date."""
        prior = ticker_prices.filter(pl.col("date") < before_date).sort("date")
        if prior.is_empty():
            return None
        return float(prior["close"][-1])

    def compute_batch(
        self,
        prices: pl.DataFrame,
        event_by_date: dict[date, list[str]],
    ) -> pl.DataFrame:
        """Compute PEAD_z for all (ticker, announce_date) events.

        Args:
            prices:         DataFrame with columns [ticker, date, close]
            event_by_date:  {announce_date: [tickers that reported that day]}

        Returns:
            DataFrame with columns [ticker, announce_date, entry_date, earnings_reaction, PEAD_z]
            Only rows where a valid earnings reaction was computable are returned.
        """
        rows = []

        for announce_date, tickers in sorted(event_by_date.items()):
            cohort_reactions: list[tuple[str, float]] = []

            for ticker in tickers:
                tp = prices.filter(pl.col("ticker") == ticker).sort("date")
                if tp.is_empty():
                    continue

                # Price on announce day (may be next trading day if announce is weekend/holiday)
                close_day0 = self._find_price(tp, announce_date, search_forward=True)
                if close_day0 is None:
                    continue

                # Actual announce date used (may differ from scheduled publish date)
                actual_day0 = announce_date
                for offset in range(6):
                    candidate = announce_date + timedelta(days=offset)
                    if not tp.filter(pl.col("date") == candidate).is_empty():
                        actual_day0 = candidate
                        break

                prior_close = self._prior_trading_close(tp, actual_day0)
                if prior_close is None or prior_close == 0:
                    continue

                reaction = (close_day0 - prior_close) / prior_close
                cohort_reactions.append((ticker, reaction))

            if len(cohort_reactions) < 3:
                continue

            # Cross-sectional z-score within cohort
            vals = np.array([r for _, r in cohort_reactions])
            mu, sd = vals.mean(), vals.std()
            if sd < 1e-10:
                continue
            z_scores = np.clip((vals - mu) / sd, -3.0, 3.0)

            # Entry date = next trading day after announce_date
            entry_date = announce_date + timedelta(days=1)
            while entry_date.weekday() >= 5:
                entry_date += timedelta(days=1)

            for (ticker, reaction), z in zip(cohort_reactions, z_scores):
                rows.append({
                    "ticker": ticker,
                    "announce_date": announce_date,
                    "entry_date": entry_date,
                    "earnings_reaction": float(reaction),
                    "PEAD_z": float(z),
                })

        if not rows:
            return pl.DataFrame({
                "ticker": pl.Series([], dtype=pl.Utf8),
                "announce_date": pl.Series([], dtype=pl.Date),
                "entry_date": pl.Series([], dtype=pl.Date),
                "earnings_reaction": pl.Series([], dtype=pl.Float64),
                "PEAD_z": pl.Series([], dtype=pl.Float64),
            })

        return pl.DataFrame(rows).with_columns([
            pl.col("announce_date").cast(pl.Date),
            pl.col("entry_date").cast(pl.Date),
        ])
