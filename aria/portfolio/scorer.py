import polars as pl
import numpy as np
from aria.signals.base import cross_sectional_zscore

class CompositeScorer:
    def __init__(self, weights: dict[str, float], winsorize_sigma: float = 3.0):
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.winsorize_sigma = winsorize_sigma

    def score(self, df: pl.DataFrame) -> pl.DataFrame:
        composite = np.zeros(len(df))
        for col, w in self.weights.items():
            if col in df.columns:
                arr = df[col].to_numpy(allow_copy=True).astype(float)
                arr = np.where(np.isnan(arr), 0.0, arr)
                composite += w * arr
        composite = np.clip(composite, -self.winsorize_sigma, self.winsorize_sigma)
        return df.with_columns(pl.Series(name="composite", values=composite, dtype=pl.Float64))

    def select_long_short(self, scored: pl.DataFrame,
                          top_pct: float = 0.20,
                          bottom_pct: float = 0.20) -> tuple[list[str], list[str]]:
        n = len(scored)
        k_long  = max(1, int(n * top_pct))
        k_short = max(1, int(n * bottom_pct))
        sorted_df = scored.sort("composite", descending=True)
        longs  = sorted_df["ticker"][:k_long].to_list()
        shorts = sorted_df["ticker"][-k_short:].to_list()
        return longs, shorts
