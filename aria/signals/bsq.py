"""BSQ — Balance Sheet Quality Filter.

A hard eligibility filter (not an alpha signal) that gates the long and short books
based on financial health. Based on the strategy specification:

Long eligibility:  BSQ_score > 30th percentile cross-sectional
Short eligibility: BSQ_score < 40th percentile AND PEAD_z < 0

Components (equal-weight composite of available metrics):
  accruals     = -(NI - CFO) / avg_total_assets   [Sloan 1996; negated: lower = better]
  cfo_margin   = CFO / Revenue                    [cash generation quality]
  debt_burden  = -Net Debt / EBITDA               [negated: lower leverage = better]
  cash_quality = 1 if CFO > NI else 0             [binary earnings quality check]

References: Sloan (1996); Piotroski (2000); Altman (1968)
"""
import numpy as np
import polars as pl
from datetime import date
from typing import Optional

from aria.signals.base import cross_sectional_zscore

_BSQ_COMPONENTS = ["accruals", "cfo_margin", "debt_burden", "cash_quality"]


class BSQSignal:
    """Balance Sheet Quality hard filter for long/short book eligibility.

    BSQ_score is a cross-sectional z-score of the equal-weight average of
    available financial quality components. Missing components are excluded
    from that ticker's average rather than zeroed.
    """

    LONG_PERCENTILE = 0.30   # must be above 30th pct to go long
    SHORT_PERCENTILE = 0.40  # must be below 40th pct to go short

    def compute_batch(
        self,
        component_rows: list[dict],
        pead_z_map: Optional[dict[str, float]] = None,
    ) -> pl.DataFrame:
        """Compute BSQ scores and eligibility flags for a batch of tickers.

        Args:
            component_rows:  List of dicts from simfin_loader.get_bsq_inputs().
                             Each dict has 'ticker' plus any subset of BSQ component keys.
            pead_z_map:      {ticker: PEAD_z} — optional; used for short eligibility gate.
                             If None, short eligibility only uses BSQ threshold.

        Returns:
            DataFrame with columns [ticker, BSQ_score, long_eligible, short_eligible].
        """
        if not component_rows:
            return pl.DataFrame({
                "ticker": pl.Series([], dtype=pl.Utf8),
                "BSQ_score": pl.Series([], dtype=pl.Float64),
                "long_eligible": pl.Series([], dtype=pl.Boolean),
                "short_eligible": pl.Series([], dtype=pl.Boolean),
            })

        # Build composite score per ticker (mean of available components)
        scored = []
        for row in component_rows:
            ticker = row["ticker"]
            vals = [row[c] for c in _BSQ_COMPONENTS if c in row and row[c] is not None]
            if not vals:
                continue
            scored.append({"ticker": ticker, "raw_bsq": float(np.mean(vals))})

        if len(scored) < 3:
            return pl.DataFrame({
                "ticker": pl.Series([], dtype=pl.Utf8),
                "BSQ_score": pl.Series([], dtype=pl.Float64),
                "long_eligible": pl.Series([], dtype=pl.Boolean),
                "short_eligible": pl.Series([], dtype=pl.Boolean),
            })

        df = pl.DataFrame(scored)

        # Cross-sectional z-score
        raw = df["raw_bsq"].to_numpy()
        mu, sd = raw.mean(), raw.std()
        if sd < 1e-10:
            z = np.zeros(len(raw))
        else:
            z = np.clip((raw - mu) / sd, -3.0, 3.0)
        df = df.with_columns(pl.Series("BSQ_score", z))

        # Eligibility thresholds
        long_thresh = float(np.percentile(z, self.LONG_PERCENTILE * 100))
        short_thresh = float(np.percentile(z, self.SHORT_PERCENTILE * 100))

        tickers = df["ticker"].to_list()
        bsq_scores = df["BSQ_score"].to_list()

        long_eligible = []
        short_eligible = []

        for ticker, bsq in zip(tickers, bsq_scores):
            is_long = bsq > long_thresh
            pead_neg = True
            if pead_z_map is not None:
                pz = pead_z_map.get(ticker, 0.0)
                pead_neg = pz < 0
            is_short = bsq < short_thresh and pead_neg
            long_eligible.append(is_long)
            short_eligible.append(is_short)

        return df.with_columns([
            pl.Series("long_eligible", long_eligible),
            pl.Series("short_eligible", short_eligible),
        ]).select(["ticker", "BSQ_score", "long_eligible", "short_eligible"])

    def apply_filter(
        self,
        longs: list[str],
        shorts: list[str],
        eligibility_df: pl.DataFrame,
    ) -> tuple[list[str], list[str]]:
        """Filter long and short lists using BSQ eligibility flags.

        Tickers not present in eligibility_df are passed through unchanged
        (conservative: don't block trades for which we have no data).
        """
        if eligibility_df.is_empty():
            return longs, shorts

        long_ok = set(
            eligibility_df.filter(pl.col("long_eligible"))["ticker"].to_list()
        )
        short_ok = set(
            eligibility_df.filter(pl.col("short_eligible"))["ticker"].to_list()
        )
        all_known = set(eligibility_df["ticker"].to_list())

        filtered_longs = [t for t in longs if t not in all_known or t in long_ok]
        filtered_shorts = [t for t in shorts if t not in all_known or t in short_ok]
        return filtered_longs, filtered_shorts
