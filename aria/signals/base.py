import polars as pl
import numpy as np


def cross_sectional_zscore(series: pl.Series, winsorize_sigma: float = 3.0) -> pl.Series:
    """Z-score a series cross-sectionally, then winsorize at +/- winsorize_sigma."""
    arr = series.to_numpy(allow_copy=True).astype(float)
    mask = ~np.isnan(arr)
    if mask.sum() < 3:
        return pl.Series(values=np.full(len(arr), np.nan), dtype=pl.Float64)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if sd < 1e-10:
        return pl.Series(values=np.zeros(len(arr)), dtype=pl.Float64)
    z = (arr - mu) / sd
    z = np.clip(z, -winsorize_sigma, winsorize_sigma)
    return pl.Series(values=z, dtype=pl.Float64)


def cross_sectional_zscore_expr(winsorize_sigma: float = 3.0) -> pl.Expr:
    """
    Return a Polars expression for cross-sectional z-score with winsorization.
    Use like: df.with_columns(cross_sectional_zscore_expr().over("date").alias("zscore"))
    """
    return (
        (pl.col("") - pl.col("").mean()) / pl.col("").std()
    ).clip(-winsorize_sigma, winsorize_sigma)


def cs_zscore(col: str, winsorize_sigma: float = 3.0) -> pl.Expr:
    """
    Cross-sectional z-score expression for a specific column.
    Use like: df.with_columns(cs_zscore("value").over("date").alias("value_z"))
    """
    return (
        (pl.col(col) - pl.col(col).mean()) / pl.col(col).std()
    ).clip(-winsorize_sigma, winsorize_sigma)