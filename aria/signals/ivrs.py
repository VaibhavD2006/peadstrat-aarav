"""IVRS - Idiosyncratic Volatility Regime Signal (Conviction Multiplier).

Measures the rate of change of a stock's idiosyncratic volatility after earnings.
Declining idiosyncratic vol = information uncertainty resolving = stronger drift conviction.
Expanding idiosyncratic vol = continued disagreement = weaker drift persistence.

Per the strategy specification (Signal 5):
  ε(i,t)    = r(i,t) - β_mkt × r_mkt(t) - β_sector × r_sector(t)  [2-factor residual]
  IVOL(i,t) = σ[ε(i,t-21d:t)]          [21-day realized idiosyncratic vol]
  IVRS(i,t) = -[IVOL(i,t) - IVOL(i,t-10d)] / IVOL(i,t-10d)
  // Negative sign: declining IVOL → positive IVRS → amplify long signals

If factor returns are not supplied, falls back to the raw vol-ratio approach
(backward compatible for existing tests).

Reference: Ang, Hodrick, Xing & Zhang (2006) — high-IVOL stocks are mispriced,
but ARIA uses IVRS conditionally (post-earnings direction) not as a cross-sectional sort.
"""
import polars as pl
import numpy as np
from typing import Optional
from aria.signals.base import cs_zscore


class IVRSignal:
    """
    IVRS Conviction Multiplier — idiosyncratic volatility regime signal.

    Acts as a position-size modifier: positive IVRS amplifies positions (vol declining),
    negative IVRS shrinks positions (vol re-expanding post-earnings).
    """

    def __init__(self, ivol_window: int = 21, change_lag: int = 10,
                 short_window: int = 20, long_window: int = 60):
        self.ivol_window = ivol_window      # window for realized idiosyncratic vol
        self.change_lag = change_lag        # lag for computing IVOL change
        self.short_window = short_window    # fallback: short vol window
        self.long_window = long_window      # fallback: long vol window

    # ------------------------------------------------------------------
    # 2-factor idiosyncratic residual computation
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_beta(y: np.ndarray, x: np.ndarray, window: int = 63) -> np.ndarray:
        """Rolling OLS beta of y on x using a fixed window."""
        n = len(y)
        betas = np.full(n, np.nan)
        for i in range(window - 1, n):
            yi = y[i - window + 1 : i + 1]
            xi = x[i - window + 1 : i + 1]
            valid = ~(np.isnan(yi) | np.isnan(xi))
            if valid.sum() < window // 2:
                continue
            yv, xv = yi[valid], xi[valid]
            var_x = np.var(xv)
            if var_x > 1e-12:
                betas[i] = np.cov(xv, yv)[0, 1] / var_x
        return betas

    def _compute_idiosyncratic_returns(
        self,
        prices: pl.DataFrame,
        mkt_returns: Optional[pl.DataFrame],
        sector_etf_returns: Optional[dict[str, pl.DataFrame]],
        sector_etf_map: Optional[dict[str, str]],
        beta_window: int = 63,
    ) -> pl.DataFrame:
        """Strip systematic returns to get idiosyncratic residuals per ticker."""
        df = prices.sort(["ticker", "date"]).with_columns([
            pl.col("adj_close").log().diff().over("ticker").alias("r_ticker")
        ])

        if mkt_returns is None:
            # Fallback: use log returns directly
            return df.with_columns(pl.col("r_ticker").alias("r_idio"))

        # Merge market returns
        mkt = mkt_returns.rename({"r_mkt": "r_mkt"}) if "r_mkt" in mkt_returns.columns else (
            mkt_returns.with_columns(
                pl.col("adj_close").log().diff().alias("r_mkt")
            ).select(["date", "r_mkt"])
        )
        df = df.join(mkt, on="date", how="left")

        results = []
        for (ticker,), grp in df.group_by(["ticker"]):
            grp = grp.sort("date")
            r = grp["r_ticker"].to_numpy().astype(float)
            r_mkt = grp["r_mkt"].to_numpy().astype(float) if "r_mkt" in grp.columns else np.zeros(len(r))

            beta_mkt = self._rolling_beta(r, r_mkt, beta_window)

            # Sector residualization if available
            beta_sector = np.zeros(len(r))
            r_sector = np.zeros(len(r))
            if sector_etf_map and sector_etf_returns:
                etf = sector_etf_map.get(ticker)
                if etf and etf in sector_etf_returns:
                    s_df = sector_etf_returns[etf].sort("date")
                    s_df = s_df.with_columns(
                        pl.col("adj_close").log().diff().alias("r_sec")
                    ).select(["date", "r_sec"])
                    merged = grp.join(s_df, on="date", how="left")
                    r_sec = merged["r_sec"].to_numpy().astype(float)
                    beta_sector = self._rolling_beta(r, r_sec, beta_window)
                    r_sector = r_sec

            # Idiosyncratic residual
            r_idio = r - np.nan_to_num(beta_mkt) * r_mkt - np.nan_to_num(beta_sector) * r_sector
            results.append(grp.with_columns(pl.Series("r_idio", r_idio)))

        return pl.concat(results) if results else df.with_columns(pl.col("r_ticker").alias("r_idio"))

    # ------------------------------------------------------------------
    # IVRS computation
    # ------------------------------------------------------------------

    def _compute_ivrs(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute IVRS from idiosyncratic returns."""
        df = df.sort(["ticker", "date"])

        df = df.with_columns([
            pl.col("r_idio")
              .rolling_std(window_size=self.ivol_window, min_samples=self.ivol_window // 2)
              .over("ticker")
              .alias("IVOL"),
        ])
        df = df.with_columns([
            pl.col("IVOL").shift(self.change_lag).over("ticker").alias("IVOL_lag")
        ])
        # IVRS = -(IVOL_now - IVOL_lag) / IVOL_lag  (negative: declining vol = positive signal)
        df = df.with_columns([
            (-(pl.col("IVOL") - pl.col("IVOL_lag")) / pl.col("IVOL_lag")).alias("IVRS_raw")
        ])
        return df

    # ------------------------------------------------------------------
    # Fallback: raw vol-ratio (backward compatible)
    # ------------------------------------------------------------------

    def _compute_vol_ratio(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.sort(["ticker", "date"]).with_columns([
            pl.col("adj_close").log().diff().over("ticker").alias("log_ret")
        ])
        df = df.with_columns([
            pl.col("log_ret")
              .rolling_std(window_size=self.short_window, min_samples=self.short_window // 2)
              .over("ticker").alias("vol_short"),
            pl.col("log_ret")
              .rolling_std(window_size=self.long_window, min_samples=self.long_window // 2)
              .over("ticker").alias("vol_long"),
        ])
        return df.with_columns(
            (pl.col("vol_short") / pl.col("vol_long")).alias("IVRS_raw")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        prices: pl.DataFrame,
        mkt_returns: Optional[pl.DataFrame] = None,
        sector_etf_returns: Optional[dict[str, pl.DataFrame]] = None,
        sector_etf_map: Optional[dict[str, str]] = None,
    ) -> pl.DataFrame:
        """Compute IVRS conviction multiplier for each ticker/date.

        Args:
            prices:            DataFrame with [ticker, date, adj_close]
            mkt_returns:       Optional [date, adj_close] or [date, r_mkt] for market factor
            sector_etf_returns: Optional {etf_ticker: [date, adj_close]} for sector factor
            sector_etf_map:    Optional {ticker: etf_ticker} mapping

        Returns:
            DataFrame with [ticker, date, IVRS_z]
        """
        if mkt_returns is not None:
            df = self._compute_idiosyncratic_returns(
                prices, mkt_returns, sector_etf_returns, sector_etf_map
            )
            df = self._compute_ivrs(df)
        else:
            df = self._compute_vol_ratio(prices)

        # Cross-sectional z-score per date
        df = df.with_columns([
            cs_zscore("IVRS_raw").over("date").alias("IVRS_z")
        ])
        return df.select(["ticker", "date", "IVRS_z"]).drop_nulls()

    def compute_latest(self, prices: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """Get IVRS for the most recent date per ticker."""
        df = self.compute(prices, **kwargs)
        return df.sort("date").group_by("ticker").tail(1).select(["ticker", "IVRS_z"])