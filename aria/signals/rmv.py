import polars as pl
import numpy as np
from datetime import date, timedelta
from aria.signals.base import cross_sectional_zscore

class RMVSignal:
    def __init__(self, lookback_days: int = 90, min_analysts: int = 3, alpha: float = 0.6):
        self.lookback_days = lookback_days
        self.min_analysts = min_analysts
        self.alpha = alpha

    def compute(self, revisions: pl.DataFrame, earnings_date: date) -> dict:
        """
        revisions: DataFrame with columns [ticker, analyst_id, revision_date, revision_pct, earnings_date]
        Returns dict with key "RMV" (float or nan).
        """
        t0 = earnings_date
        window_start = t0 - timedelta(days=self.lookback_days)

        df = revisions.filter(
            (pl.col("revision_date") >= window_start) &
            (pl.col("revision_date") < t0)
        )

        if df.is_empty() or df["analyst_id"].n_unique() < self.min_analysts:
            return {"RMV": float("nan")}

        def bucket_mean(days_from_end_start: int, days_from_end_end: int) -> float:
            b_start = t0 - timedelta(days=days_from_end_start)
            b_end   = t0 - timedelta(days=days_from_end_end)
            sub = df.filter(
                (pl.col("revision_date") >= b_start) &
                (pl.col("revision_date") < b_end)
            )
            return float(sub["revision_pct"].mean()) if not sub.is_empty() else 0.0

        b1 = bucket_mean(90, 60)
        b2 = bucket_mean(60, 30)
        b3 = bucket_mean(30, 0)

        momentum     = b3 - b1
        acceleration = b3 - 2 * b2 + b1

        rmv = self.alpha * momentum + (1 - self.alpha) * acceleration
        return {"RMV": float(rmv)}

    def compute_batch(self, revisions: pl.DataFrame,
                      events: list[tuple[str, date]]) -> pl.DataFrame:
        rows = []
        for ticker, edate in events:
            sub = revisions.filter(pl.col("ticker") == ticker)
            result = self.compute(sub, edate)
            rows.append({"ticker": ticker, "earnings_date": edate, "RMV": result["RMV"]})
        return pl.DataFrame(rows)
