import polars as pl
import numpy as np
from aria.signals.base import cross_sectional_zscore

class ESQSSignal:
    """
    Earnings Surprise Quality Score.
    Decomposes earnings quality into: revenue surprise, GM expansion,
    SGA efficiency, and guidance direction.
    """
    WEIGHTS = {"rev_surp": 0.30, "gm_expand": 0.25, "sga_eff": 0.20, "guidance": 0.25}

    def _raw_components(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            ((pl.col("revenue_actual") - pl.col("revenue_consensus"))
             / pl.col("revenue_consensus").abs()).alias("rev_surp"),
            (pl.col("gross_margin_actual") - pl.col("gross_margin_trailing")).alias("gm_expand"),
            (-(pl.col("sga_pct_actual") - pl.col("sga_pct_trailing"))).alias("sga_eff"),
            pl.col("guidance_score").fill_null(0.0).alias("guidance"),
        ])

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self._raw_components(df)
        w = self.WEIGHTS
        scores = (
            w["rev_surp"]  * df["rev_surp"].to_numpy(allow_copy=True).astype(float) +
            w["gm_expand"] * df["gm_expand"].to_numpy(allow_copy=True).astype(float) +
            w["sga_eff"]   * df["sga_eff"].to_numpy(allow_copy=True).astype(float) +
            w["guidance"]  * df["guidance"].to_numpy(allow_copy=True).astype(float)
        )
        return df.with_columns(pl.Series(name="ESQS", values=scores, dtype=pl.Float64))

    def compute_normalized(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self.compute(df)
        return df.with_columns(cross_sectional_zscore(df["ESQS"]).alias("ESQS_z"))
