"""4-State HMM Regime Classifier.

Classifies market regimes using return, volatility, and trend features.
States: 0=Bull, 1=Bear, 2=Crisis, 3=Recovery (ordered by return/vol profile).
"""
import numpy as np
import polars as pl
from hmmlearn import hmm
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


class RegimeHMM:
    """
    4-State Hidden Markov Model for market regime classification.

    Features (per date): daily log return, rolling-20d realized vol, rolling-20d price slope.
    """

    def __init__(self, n_states: int = 4, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self._dates: Optional[list] = None  # dates aligned to feature rows after drop_nulls

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(self, prices: pl.DataFrame, window: int = 20) -> tuple[np.ndarray, list]:
        """Return (X, dates) where X has shape (n, 3) and dates is aligned list."""
        df = prices.sort("date").with_columns([
            pl.col("adj_close").log().diff().alias("log_ret")
        ])
        df = df.with_columns([
            pl.col("log_ret").rolling_std(window_size=window, min_samples=window // 2).alias("vol_20d")
        ])

        # Rolling linear-trend slope via numpy polyfit applied row-wise
        closes = df["adj_close"].to_numpy()
        n = len(closes)
        slopes = np.full(n, np.nan)
        for i in range(window - 1, n):
            seg = closes[i - window + 1 : i + 1]
            if not np.any(np.isnan(seg)):
                slopes[i] = np.polyfit(np.arange(window), seg, 1)[0]

        df = df.with_columns(pl.Series("trend_20d", slopes))

        # drop_nulls() only removes Polars nulls; use is_not_nan() for float NaN
        feature_df = (
            df.select(["date", "log_ret", "vol_20d", "trend_20d"])
            .filter(
                pl.col("log_ret").is_not_null() & pl.col("log_ret").is_not_nan() &
                pl.col("vol_20d").is_not_null() & pl.col("vol_20d").is_not_nan() &
                pl.col("trend_20d").is_not_null() & pl.col("trend_20d").is_not_nan()
            )
        )
        X = feature_df.select(["log_ret", "vol_20d", "trend_20d"]).to_numpy().astype(float)
        dates = feature_df["date"].to_list()
        return X, dates

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, prices: pl.DataFrame, window: int = 20) -> "RegimeHMM":
        X, self._dates = self._build_features(prices, window)
        if len(X) < 100:
            raise ValueError(f"Need at least 100 observations after feature prep, got {len(X)}")

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=100,
            tol=1e-4,
            random_state=self.random_state,
        )
        self.model.fit(X)
        self._reorder_states()
        return self

    def _reorder_states(self) -> None:
        """Reorder so state 0=Bull (best Sharpe), 2=Crisis (worst)."""
        if self.model is None:
            return
        means = self.model.means_[:, 0]  # return dimension
        vols  = self.model.means_[:, 1]  # vol dimension
        scores = means / (vols + 1e-8)
        sorted_idx = np.argsort(scores)           # ascending: worst first
        remapping = np.empty(self.n_states, dtype=int)
        desired = [2, 1, 3, 0]                    # sorted position -> new label
        for new_label, old_idx in zip(desired, sorted_idx):
            remapping[old_idx] = new_label
        order = np.argsort(remapping)
        self.model.startprob_ = self.model.startprob_[order]
        self.model.transmat_  = self.model.transmat_[order][:, order]
        self.model.means_     = self.model.means_[order]
        # Bypass the setter (which validates shape) and reindex the raw internal array
        self.model._covars_   = self.model._covars_[order]

    def predict(self, prices: pl.DataFrame, window: int = 20) -> pl.DataFrame:
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X, dates = self._build_features(prices, window)
        regimes = self.model.predict(X)
        return pl.DataFrame({"date": dates, "regime": regimes})

    def predict_proba(self, prices: pl.DataFrame, window: int = 20) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X, _ = self._build_features(prices, window)
        return self.model.predict_proba(X)

    def get_transition_matrix(self) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.transmat_.copy()

    def get_regime_stats(self, prices: pl.DataFrame, window: int = 20) -> dict:
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        regimes_df = self.predict(prices, window)
        df = prices.sort("date").join(regimes_df, on="date", how="inner")
        df = df.with_columns(pl.col("adj_close").log().diff().alias("log_ret"))

        stats = {}
        for regime in range(self.n_states):
            sub = df.filter(pl.col("regime") == regime)
            if sub.shape[0] > 0:
                rets = sub["log_ret"].drop_nulls().to_numpy()
                std = float(np.std(rets)) if len(rets) > 1 else 0.0
                stats[regime] = {
                    "name": self.regime_labels()[regime],
                    "count": sub.shape[0],
                    "mean_return": float(np.mean(rets)),
                    "std_return": std,
                    "ann_return": float(np.mean(rets) * 252),
                    "ann_vol": float(std * np.sqrt(252)),
                    "sharpe": float(np.mean(rets) / std * np.sqrt(252)) if std > 0 else 0.0,
                }
        return stats

    def regime_labels(self) -> list[str]:
        return ["Bull", "Bear", "Crisis", "Recovery"]


def create_regime_filter(
    prices: pl.DataFrame,
    allowed_regimes: list[int] = None,
    window: int = 20,
) -> pl.DataFrame:
    """
    Fit HMM and return [date, regime, regime_filter] where regime_filter=1 for allowed regimes.
    Defaults to Bull (0) and Recovery (3).
    """
    if allowed_regimes is None:
        allowed_regimes = [0, 3]
    model = RegimeHMM().fit(prices, window)
    regimes_df = model.predict(prices, window)
    return regimes_df.with_columns(
        pl.col("regime").is_in(allowed_regimes).cast(pl.Int64).alias("regime_filter")
    )


class RollingRegimeHMM:
    """
    Rolling HMM regime classifier — avoids look-ahead bias.

    Fits a new HMM every `refit_freq_days` on a trailing `lookback_days` window.
    Cached by quarter so we don't refit on every event.
    """

    def __init__(
        self,
        lookback_days: int = 504,   # ~2 years of trading days
        refit_freq_days: int = 63,  # re-fit every quarter
        n_states: int = 4,
        random_state: int = 42,
        allowed_regimes: list[int] = None,
    ):
        self.lookback_days = lookback_days
        self.refit_freq_days = refit_freq_days
        self.n_states = n_states
        self.random_state = random_state
        self.allowed_regimes = allowed_regimes if allowed_regimes is not None else [0, 3]
        self._cache: dict[str, RegimeHMM] = {}  # cache key: YYYY-Qn

    def _cache_key(self, d: "date") -> str:
        from datetime import date
        q = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{q}"

    def get_regime(self, prices: pl.DataFrame, as_of_date: "date", window: int = 20) -> int:
        """
        Return regime label (0-3) for as_of_date without look-ahead.
        Fits on data strictly before as_of_date.
        """
        key = self._cache_key(as_of_date)
        if key not in self._cache:
            from datetime import timedelta
            cutoff = as_of_date
            lookback_start = cutoff - timedelta(days=self.lookback_days + 30)
            p_slice = prices.filter(
                (pl.col("date") >= lookback_start) & (pl.col("date") < cutoff)
            )
            if p_slice.shape[0] < 120:
                return -1  # insufficient history
            try:
                model = RegimeHMM(n_states=self.n_states, random_state=self.random_state)
                model.fit(p_slice, window=window)
                self._cache[key] = model
            except Exception:
                return -1

        model = self._cache[key]
        # Predict on a small slice ending at as_of_date
        from datetime import timedelta
        p_recent = prices.filter(
            (pl.col("date") <= as_of_date) &
            (pl.col("date") >= as_of_date - timedelta(days=60))
        )
        if p_recent.shape[0] < 25:
            return -1
        try:
            regimes_df = model.predict(p_recent, window=window)
            if regimes_df.is_empty():
                return -1
            return int(regimes_df["regime"][-1])
        except Exception:
            return -1

    def is_allowed(self, prices: pl.DataFrame, as_of_date: "date", window: int = 20) -> bool:
        """Return True if current regime is in allowed_regimes (or if regime is unknown)."""
        r = self.get_regime(prices, as_of_date, window)
        if r == -1:
            return True  # unknown → don't block trade
        return r in self.allowed_regimes
