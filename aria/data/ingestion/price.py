import yfinance as yf
import polars as pl
from pathlib import Path

class PriceLoader:
    def __init__(self, cache_dir: str = "data/parquet/prices"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, start: str, end: str) -> Path:
        return self.cache_dir / f"{ticker}_{start}_{end}.parquet"

    def _fetch_from_yfinance(self, ticker: str, start: str, end: str) -> pl.DataFrame:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return pl.DataFrame({"date": [], "open": [], "high": [], "low": [],
                                  "adj_close": [], "volume": [], "ticker": []})
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]
        raw = raw.reset_index()
        raw.columns = [str(c).lower() for c in raw.columns]
        if "close" in raw.columns and "adj_close" not in raw.columns:
            raw = raw.rename(columns={"close": "adj_close"})
        df = pl.from_pandas(raw)
        df = df.with_columns(pl.lit(ticker).alias("ticker"))
        # ensure date column is Date type
        if "date" in df.columns and df["date"].dtype != pl.Date:
            df = df.with_columns(pl.col("date").cast(pl.Date))
        return df

    def load(self, ticker: str, start: str, end: str) -> pl.DataFrame:
        cache_path = self._cache_path(ticker, start, end)
        if cache_path.exists():
            return pl.read_parquet(cache_path)
        df = self._fetch_from_yfinance(ticker, start, end)
        if df.shape[0] > 0:
            df.write_parquet(cache_path)
        return df

    def load_many(self, tickers: list[str], start: str, end: str) -> pl.DataFrame:
        frames = [self.load(t, start, end) for t in tickers]
        return pl.concat([f for f in frames if f.shape[0] > 0])
