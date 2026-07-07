"""yfinance-backed price store for extended date ranges.

Supplements the SimFin CSV (which only covers 2020-08 to 2023-10) by downloading
full daily OHLCV from yfinance for any requested ticker universe and date range.
Results are cached as a single parquet file so subsequent runs are instant.
"""
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import yfinance as yf

_CACHE_DIR = Path("data/parquet/yf_prices")
_BATCH_SIZE = 200
_BATCH_DELAY = 1.0  # seconds between batches


def _cache_path(start: date, end: date) -> Path:
    return _CACHE_DIR / f"universe_{start}_{end}.parquet"


def _download_batch(tickers: list[str], start: date, end: date) -> pl.DataFrame:
    """Download one batch of tickers and return a normalized Polars DataFrame."""
    raw = yf.download(
        tickers,
        start=str(start),
        end=str(end + timedelta(days=1)),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pl.DataFrame()

    # yfinance 1.x returns MultiIndex (Price, Ticker) columns
    frames = []
    for ticker in tickers:
        try:
            t_df = raw.xs(ticker, axis=1, level="Ticker") if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        if t_df.empty or t_df["Close"].isna().all():
            continue
        t_df = t_df.reset_index()[["Date", "Open", "Close", "Volume"]].copy()
        t_df.columns = ["date", "open", "close", "volume"]
        t_df["adj_close"] = t_df["close"]  # auto_adjust=True means Close IS adj close
        t_df["ticker"] = ticker
        t_df = t_df.dropna(subset=["close"])
        t_df["volume"] = t_df["volume"].astype(float)  # normalise to Float64 across all batches
        if t_df.empty:
            continue
        frames.append(
            pl.from_pandas(t_df).with_columns(pl.col("date").cast(pl.Date))
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)


def build_price_cache(
    tickers: list[str],
    start: date,
    end: date,
    verbose: bool = True,
) -> pl.DataFrame:
    """Download prices for all tickers and save to parquet cache.

    Skips any tickers already in the cache (safe to call incrementally).
    Returns the full cached DataFrame.
    """
    path = _cache_path(start, end)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing cache if present
    if path.exists():
        existing = pl.read_parquet(path)
        cached_tickers = set(existing["ticker"].unique().to_list())
        to_fetch = [t for t in tickers if t not in cached_tickers]
        if not to_fetch:
            if verbose:
                print(f"[YFPrices] All {len(tickers)} tickers cached ({existing.shape[0]:,} rows).")
            return existing
        if verbose:
            print(f"[YFPrices] {len(cached_tickers)} cached; fetching {len(to_fetch)} new tickers...")
    else:
        existing = None
        to_fetch = tickers
        if verbose:
            print(f"[YFPrices] Downloading {len(to_fetch)} tickers ({start} to {end})...")

    n_batches = (len(to_fetch) - 1) // _BATCH_SIZE + 1
    new_frames = []
    for i in range(0, len(to_fetch), _BATCH_SIZE):
        batch = to_fetch[i : i + _BATCH_SIZE]
        b_num = i // _BATCH_SIZE + 1
        if verbose:
            print(f"[YFPrices]   Batch {b_num}/{n_batches} ({len(batch)} tickers)...")
        try:
            frame = _download_batch(batch, start, end)
            if not frame.is_empty():
                new_frames.append(frame)
        except Exception as e:
            if verbose:
                print(f"[YFPrices]   Batch {b_num} failed: {e}")
        if i + _BATCH_SIZE < len(to_fetch):
            time.sleep(_BATCH_DELAY)

    if not new_frames:
        if verbose:
            print("[YFPrices]   Warning: no new data downloaded.")
        return existing if existing is not None else pl.DataFrame()

    new_data = pl.concat(new_frames).sort(["ticker", "date"])
    combined = pl.concat([existing, new_data]) if existing is not None else new_data
    combined.write_parquet(path)
    if verbose:
        print(f"[YFPrices] Saved {combined.shape[0]:,} rows ({combined['ticker'].n_unique()} tickers) to {path}")
    return combined


class YFinancePriceStore:
    """Drop-in replacement for SimFinPriceStore backed by yfinance data.

    On first use for a given (start, end) range, downloads all tickers from
    yfinance and saves to parquet. Subsequent runs load from cache instantly.
    """

    def __init__(self, start: date, end: date):
        self._start = start
        self._end = end
        self._df: Optional[pl.DataFrame] = None

        path = _cache_path(start, end)
        if path.exists():
            print("[YFPrices] Loading price cache...")
            self._df = pl.read_parquet(path)
            print(f"[YFPrices] Loaded {self._df.shape[0]:,} rows, "
                  f"{self._df['ticker'].n_unique()} tickers.")

    def ensure_built(self, tickers: list[str]) -> None:
        """Download missing tickers if the cache is incomplete."""
        self._df = build_price_cache(tickers, self._start, self._end, verbose=True)

    def get(
        self,
        tickers: Optional[list[str]] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pl.DataFrame:
        df = self._df
        if df is None:
            return pl.DataFrame()
        if tickers:
            df = df.filter(pl.col("ticker").is_in(tickers))
        if start:
            df = df.filter(pl.col("date") >= start)
        if end:
            df = df.filter(pl.col("date") <= end)
        return df

    def liquid_tickers(self, min_adv_usd: float = 5e7) -> list[str]:
        if self._df is None:
            return []
        adv = (
            self._df
            .with_columns((pl.col("close") * pl.col("volume")).alias("dolvol"))
            .group_by("ticker")
            .agg(pl.col("dolvol").mean().alias("adv"))
            .filter(pl.col("adv") >= min_adv_usd)
        )
        return adv["ticker"].to_list()
