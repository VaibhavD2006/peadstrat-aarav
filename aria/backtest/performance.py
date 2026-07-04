import numpy as np
from scipy import stats

class PerformanceAnalytics:
    def sharpe(self, daily_returns: np.ndarray, rf_annual: float = 0.05) -> float:
        rf_daily = rf_annual / 252
        excess   = daily_returns - rf_daily
        std = excess.std()
        if std < 1e-10:
            mean = excess.mean()
            if mean > 1e-10:
                return float("inf")
            elif mean < -1e-10:
                return float("-inf")
            return 0.0
        return float(excess.mean() / std * np.sqrt(252))

    def max_drawdown(self, equity_curve: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity_curve)
        dd   = (equity_curve - peak) / np.where(peak == 0, 1.0, peak)
        return float(dd.min())

    def annualized_return(self, daily_returns: np.ndarray) -> float:
        return float((1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1)

    def annualized_vol(self, daily_returns: np.ndarray) -> float:
        return float(daily_returns.std() * np.sqrt(252))

    def information_coefficient(self, signals: np.ndarray,
                                 forward_returns: np.ndarray) -> float:
        mask = ~(np.isnan(signals) | np.isnan(forward_returns))
        if mask.sum() < 5:
            return float("nan")
        ic, _ = stats.spearmanr(signals[mask], forward_returns[mask])
        return float(ic)

    def summarize(self, daily_returns: np.ndarray, rf_annual: float = 0.05) -> dict:
        equity = np.cumprod(1 + daily_returns) * 100.0
        return {
            "sharpe":        self.sharpe(daily_returns, rf_annual),
            "annual_return": self.annualized_return(daily_returns),
            "annual_vol":    self.annualized_vol(daily_returns),
            "max_drawdown":  self.max_drawdown(equity),
            "n_periods":     len(daily_returns),
        }
