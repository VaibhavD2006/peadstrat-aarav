"""Download historical earnings consensus via yfinance (Yahoo Finance).

Free — uses the yfinance library already installed for price data.
Provides EPS Estimate (consensus), Reported EPS, and announcement date.
Coverage: ~24 quarters (~6 years) per ticker.

Usage:
    python scripts/download_eodhd_consensus.py

Output: data/consensus/eps_consensus.csv
    ticker, report_date, consensus_eps, actual_eps, fiscal_quarter_end

Safe to interrupt and resume — saves every 50 tickers.
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

_OUT        = Path("data/consensus/eps_consensus.csv")
_SAVE_EVERY = 50
_DELAY      = 0.3    # seconds between tickers (polite to Yahoo)


def fetch_earnings(ticker: str) -> list[dict]:
    """Pull earnings dates with EPS estimate and actual for one ticker."""
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return []
        rows = []
        for dt, row in ed.iterrows():
            actual = row.get("Reported EPS")
            est    = row.get("EPS Estimate")
            if pd.isna(actual) or pd.isna(est):
                continue
            rows.append({
                "ticker":             ticker,
                "report_date":        dt.date().isoformat(),
                "fiscal_quarter_end": "",   # Yahoo doesn't provide this directly
                "consensus_eps":      float(est),
                "actual_eps":         float(actual),
            })
        return rows
    except Exception:
        return []


def load_universe() -> list[str]:
    import polars as pl

    # yfinance price cache — already the filtered active universe
    yf_dir = Path("data/parquet/yf_prices")
    if yf_dir.exists():
        files = sorted(yf_dir.glob("*.parquet"))
        if files:
            df = pl.read_parquet(files[0])
            tickers = df["ticker"].unique().to_list()
            print(f"Using yfinance price cache: {len(tickers)} tickers")
            return tickers

    # SimFin parquet filtered to 2021+
    parquet = Path("data/parquet/income_flat.parquet")
    if parquet.exists():
        df = pl.read_parquet(parquet)
        if "Publish Date" in df.columns:
            df = df.filter(pl.col("Publish Date") >= "2021-01-01")
        tickers = df["Ticker"].unique().to_list()
        print(f"Using SimFin 2021+ universe: {len(tickers)} tickers")
        return tickers

    raise FileNotFoundError("No universe source found.")


def save(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "report_date"])
    df = df.sort_values(["ticker", "report_date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    print("Loading universe...")
    tickers = load_universe()

    # Incremental: skip already-downloaded tickers
    if _OUT.exists():
        existing = pd.read_csv(_OUT)
        done = set(existing["ticker"].unique())
        to_fetch = [t for t in tickers if t not in done]
        print(f"Already cached: {len(done)}. Fetching {len(to_fetch)} new tickers.")
        all_rows = existing.to_dict("records")
    else:
        to_fetch = tickers
        all_rows = []

    n = len(to_fetch)
    if n == 0:
        print("All tickers already cached.")
        return

    for i, ticker in enumerate(to_fetch):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i+1}/{n}  ({ticker})")
        rows = fetch_earnings(ticker)
        all_rows.extend(rows)
        time.sleep(_DELAY)

        if (i + 1) % _SAVE_EVERY == 0:
            save(all_rows, _OUT)

    save(all_rows, _OUT)

    if _OUT.exists():
        df = pd.read_csv(_OUT)
        print(f"\nSaved {len(df):,} rows ({df['ticker'].nunique()} tickers) -> {_OUT}")
        print(f"Date range: {df['report_date'].min()} to {df['report_date'].max()}")
    else:
        print("No data saved (all tickers returned empty).")


if __name__ == "__main__":
    main()
