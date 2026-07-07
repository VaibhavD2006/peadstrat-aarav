"""IFR - Institutional Flow Residual.

Volume-based signal that regresses ticker volume against sector ETF volume
to detect anomalous institutional flow. Uses only yfinance data - zero additional cost.
"""
import polars as pl
import numpy as np
from datetime import date, timedelta
from typing import Optional
from aria.signals.base import cs_zscore, cross_sectional_zscore
from aria.data.ingestion.price import PriceLoader


SECTOR_ETF_MAP = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}


class IFRSignal:
    """
    IFR Signal: Volume residual from sector ETF regression.

    Logic:
    - For each ticker, get daily volume and sector ETF volume
    - Run rolling OLS: log(ticker_volume) ~ alpha + beta * log(sector_etf_volume)
    - Residual = actual log(volume) - predicted log(volume)
    - Cross-sectionally z-score residuals across tickers per date
    - High positive residual = unusual buying pressure (institutional accumulation)
    - High negative residual = unusual selling pressure (institutional distribution)
    """

    def __init__(self, lookback_days: int = 60, min_periods: int = 30):
        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.price_loader = PriceLoader()

    def _get_sector_etf(self, ticker: str) -> str:
        """Map ticker to sector ETF."""
        tech = {"AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD", "INTC"}
        fin  = {"JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF"}
        hc   = {"JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN"}
        if ticker in tech:
            return "XLK"
        if ticker in fin:
            return "XLF"
        if ticker in hc:
            return "XLV"
        return "XLK"

    def _load_sector_volumes(self, sector_etfs: list[str], start: str, end: str) -> pl.DataFrame:
        """Load volume data for sector ETFs."""
        frames = []
        for etf in sector_etfs:
            try:
                df = self.price_loader.load(etf, start, end)
                if df.shape[0] > 0:
                    df = df.select(["date", "volume"]).rename({"volume": f"vol_{etf}"})
                    frames.append(df)
            except Exception:
                pass
        if not frames:
            return pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})
        result = frames[0]
        for f in frames[1:]:
            result = result.join(f, on="date", how="full", coalesce=True)
        return result.sort("date")

    def _rolling_ols_residuals(self, y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
        """Vectorised rolling OLS residuals: y ~ alpha + beta*x."""
        n = len(y)
        residuals = np.full(n, np.nan)
        for i in range(window - 1, n):
            yi = y[i - window + 1 : i + 1]
            xi = x[i - window + 1 : i + 1]
            valid = ~(np.isnan(yi) | np.isnan(xi))
            if valid.sum() < self.min_periods:
                continue
            yi_v, xi_v = yi[valid], xi[valid]
            beta = np.cov(xi_v, yi_v)[0, 1] / np.var(xi_v) if np.var(xi_v) > 0 else 0.0
            alpha = yi_v.mean() - beta * xi_v.mean()
            residuals[i] = y[i] - (alpha + beta * x[i])
        return residuals

    @staticmethod
    def _sign_residuals(residuals: np.ndarray, close: np.ndarray, open_: np.ndarray) -> np.ndarray:
        """Sign OLS residuals by intraday price direction (close > open = buy day)."""
        direction = np.sign(close - open_)
        direction[direction == 0] = 1.0  # flat day treated as neutral positive
        return residuals * direction

    @staticmethod
    def _exponential_accumulate(arr: np.ndarray, decay: float = 0.85, window: int = 5) -> np.ndarray:
        """5-day exponentially-weighted accumulation: today × 1, yesterday × decay, ..."""
        n = len(arr)
        result = np.full(n, np.nan)
        weights = np.array([decay ** t for t in range(window)])  # [1, 0.85, 0.85^2, ...]
        for i in range(window - 1, n):
            window_vals = arr[i - window + 1 : i + 1][::-1]  # most recent first
            valid = ~np.isnan(window_vals)
            if valid.sum() >= max(1, window // 2):
                w = weights.copy()
                w[~valid] = 0.0
                result[i] = float(np.dot(window_vals * valid, w[:len(window_vals)]))
        return result

    def compute(
        self,
        ticker: str,
        prices: pl.DataFrame,
        sector_volumes: Optional[pl.DataFrame] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Compute IFR for a single ticker.

        Returns DataFrame with [date, IFR_raw] (not yet cross-sectionally scored).
        """
        sector_etf = self._get_sector_etf(ticker)

        if "date" not in prices.columns or "volume" not in prices.columns:
            raise ValueError("prices must have 'date' and 'volume' columns")

        prices = prices.sort("date").with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("volume").cast(pl.Float64),
        ])

        if sector_volumes is None:
            min_date = prices["date"].min()
            max_date = prices["date"].max()
            s = start or str(min_date - timedelta(days=self.lookback_days + 10))
            e = end or str(max_date)
            sector_volumes = self._load_sector_volumes([sector_etf], s, e)

        if sector_volumes.is_empty():
            return pl.DataFrame({"date": pl.Series([], dtype=pl.Date), "IFR_raw": pl.Series([], dtype=pl.Float64)})

        sector_col = f"vol_{sector_etf}"
        if sector_col not in sector_volumes.columns:
            return pl.DataFrame({"date": pl.Series([], dtype=pl.Date), "IFR_raw": pl.Series([], dtype=pl.Float64)})

        df = prices.join(sector_volumes.select(["date", sector_col]), on="date", how="inner")
        df = df.filter((pl.col("volume") > 0) & (pl.col(sector_col) > 0))

        if df.shape[0] < self.min_periods:
            return pl.DataFrame({"date": pl.Series([], dtype=pl.Date), "IFR_raw": pl.Series([], dtype=pl.Float64)})

        log_y = np.log(df["volume"].to_numpy())
        log_x = np.log(df[sector_col].to_numpy())
        residuals = self._rolling_ols_residuals(log_y, log_x, self.lookback_days)

        # Sign by intraday direction if open/close available
        if "open" in df.columns and "close" in df.columns:
            residuals = self._sign_residuals(
                residuals,
                df["close"].to_numpy().astype(float),
                df["open"].to_numpy().astype(float),
            )

        # 5-day exponential accumulation
        residuals = self._exponential_accumulate(residuals)

        # Time-series z-score for single-ticker output
        valid_mask = ~np.isnan(residuals)
        z = np.full(len(residuals), np.nan)
        if valid_mask.sum() >= 3:
            mu = residuals[valid_mask].mean()
            sd = residuals[valid_mask].std()
            if sd > 1e-10:
                z[valid_mask] = np.clip((residuals[valid_mask] - mu) / sd, -3, 3)

        result = pl.DataFrame({"date": df["date"], "IFR_z": z})
        return result.filter(pl.col("IFR_z").is_not_null() & pl.col("IFR_z").is_not_nan())

    def compute_batch(
        self,
        tickers: list[str],
        prices_dict: dict[str, pl.DataFrame],
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Compute IFR for multiple tickers with cross-sectional z-scores per date.

        Returns DataFrame with [ticker, date, IFR_z].
        """
        sector_etfs = list({self._get_sector_etf(t) for t in tickers})
        sector_volumes = self._load_sector_volumes(sector_etfs, start, end)

        all_results = []
        for ticker in tickers:
            if ticker not in prices_dict:
                continue
            result = self.compute(ticker, prices_dict[ticker], sector_volumes, start, end)
            if result.shape[0] > 0:
                result = result.with_columns(pl.lit(ticker).alias("ticker"))
                all_results.append(result)

        if not all_results:
            return pl.DataFrame({
                "ticker": pl.Series([], dtype=pl.Utf8),
                "date": pl.Series([], dtype=pl.Date),
                "IFR_z": pl.Series([], dtype=pl.Float64),
            })

        combined = pl.concat(all_results)

        # Re-score cross-sectionally per date (each ticker's IFR_z was time-series normalised;
        # now normalise across tickers within each date)
        def _cs_zscore_group(grp: pl.DataFrame) -> pl.DataFrame:
            vals = grp["IFR_z"].to_numpy().astype(float)
            valid = ~np.isnan(vals)
            z = vals.copy()
            if valid.sum() >= 3:
                mu, sd = vals[valid].mean(), vals[valid].std()
                if sd > 1e-10:
                    z[valid] = np.clip((vals[valid] - mu) / sd, -3, 3)
            return grp.with_columns(pl.Series("IFR_z", z))

        combined = (
            combined
            .group_by("date")
            .map_groups(_cs_zscore_group)
            .sort(["date", "ticker"])
        )

        return combined.select(["ticker", "date", "IFR_z"]).filter(
            pl.col("IFR_z").is_not_null() & pl.col("IFR_z").is_not_nan()
        )
