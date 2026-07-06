"""SUE — Standardized Unexpected Earnings.

SUE = (actual_EPS - consensus_EPS) / normalizer

where normalizer is the standard deviation of analyst forecast errors over the
prior 8 quarters (if available) or abs(consensus_EPS) as a fallback.

References: Jones & Litzenberger (1970); Bernard & Thomas (1989); Livnat & Mendenhall (2006)

Expected CSV schema from data providers (see sue_loader.py):
    ticker, report_date, fiscal_quarter_end, consensus_eps, actual_eps[, n_analysts, std_eps]
"""
import numpy as np
import polars as pl
from datetime import date
from typing import Optional


class SUESignal:
    """Standardized Unexpected Earnings — cross-sectional z-score within each earnings cohort."""

    def compute_normalized(self, rows: list[dict]) -> pl.DataFrame:
        """Compute SUE_z from a list of per-ticker input dicts.

        Each dict must have:
            ticker          str
            sue_raw         float   (actual_eps - consensus_eps) / normalizer

        Returns DataFrame [ticker, SUE_z] with cross-sectional z-score clipped to ±3.
        """
        if not rows:
            return pl.DataFrame({"ticker": pl.Series([], dtype=pl.Utf8),
                                 "SUE_z": pl.Series([], dtype=pl.Float64)})

        tickers = [r["ticker"] for r in rows]
        raws = np.array([r["sue_raw"] for r in rows], dtype=float)

        valid = ~np.isnan(raws)
        z = np.full(len(raws), 0.0)
        if valid.sum() >= 3:
            mu = raws[valid].mean()
            sd = raws[valid].std()
            if sd > 1e-10:
                z[valid] = np.clip((raws[valid] - mu) / sd, -3.0, 3.0)

        return pl.DataFrame({"ticker": tickers, "SUE_z": z})


def compute_sue_raw(
    actual_eps: float,
    consensus_eps: float,
    forecast_std: Optional[float] = None,
    hist_errors: Optional[list[float]] = None,
    fallback_scale: float = 0.01,
) -> float:
    """Compute a single ticker's raw SUE value.

    Normalizer priority:
      1. Rolling std of hist_errors (past N quarters of actual-minus-consensus)
         — requires >= 2 values; captures per-ticker analyst accuracy history.
      2. forecast_std (cross-sectional std of analyst estimates).
      3. max(abs(consensus_eps), fallback_scale) — always available.
    """
    surprise = actual_eps - consensus_eps

    if hist_errors is not None and len(hist_errors) >= 2:
        normalizer = max(float(np.std(hist_errors, ddof=1)), fallback_scale)
        return surprise / normalizer

    if forecast_std is not None and forecast_std >= fallback_scale:
        return surprise / forecast_std

    denom = max(abs(consensus_eps), fallback_scale)
    return surprise / denom
