"""One-time download of historical earnings surprises from Financial Modeling Prep.

Usage:
    python scripts/download_fmp_consensus.py --api-key YOUR_KEY_HERE

Output: data/consensus/eps_consensus.csv
    ticker, report_date, fiscal_quarter_end, consensus_eps, actual_eps, n_analysts

Run once to build the cache. Re-run anytime to pick up new quarters.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

_BASE = "https://financialmodelingprep.com/api/v3"
_OUT  = Path("data/consensus/eps_consensus.csv")
_DELAY = 0.25   # seconds between requests (Starter: 250 calls/day, no burst limit stated)


def fetch_earnings_history(ticker: str, api_key: str) -> list[dict]:
    """Pull historical EPS actuals + estimates for one ticker."""
    url = f"{_BASE}/historical/earning_calendar/{ticker}"
    try:
        r = requests.get(url, params={"apikey": api_key}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return []
        rows = []
        for item in data:
            actual  = item.get("eps")
            est     = item.get("epsEstimated")
            rdate   = item.get("date")            # announcement date
            fqend   = item.get("fiscalDateEnding")
            if actual is None or est is None or rdate is None:
                continue
            rows.append({
                "ticker":              ticker,
                "report_date":         rdate,
                "fiscal_quarter_end":  fqend or "",
                "consensus_eps":       est,
                "actual_eps":          actual,
                "n_analysts":          item.get("numberAnalysts") or "",
            })
        return rows
    except Exception as e:
        print(f"  [{ticker}] error: {e}")
        return []


def load_universe() -> list[str]:
    """Read the SimFin ticker universe from income statement parquet/CSV."""
    # Try parquet first, fall back to CSV
    parquet = Path("data/parquet/income_flat.parquet")
    if parquet.exists():
        import polars as pl
        return pl.read_parquet(parquet)["Ticker"].unique().to_list()

    csv = Path("data/simfin/us-income-quarterly.csv")
    if csv.exists():
        df = pd.read_csv(csv, sep=";", usecols=["Ticker"]).dropna()
        return df["Ticker"].unique().tolist()

    raise FileNotFoundError(
        "Could not find SimFin income data. "
        "Run the SimFin loader first to build the universe."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="FMP API key")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated subset (default: full SimFin universe)")
    args = parser.parse_args()

    _OUT.parent.mkdir(parents=True, exist_ok=True)

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        print("Loading SimFin universe...")
        tickers = load_universe()
        print(f"Universe: {len(tickers)} tickers")

    # Load existing cache so we can skip already-downloaded tickers
    if _OUT.exists():
        existing = pd.read_csv(_OUT)
        done = set(existing["ticker"].unique())
        tickers = [t for t in tickers if t not in done]
        print(f"Already cached: {len(done)} tickers. Fetching {len(tickers)} new.")
        all_rows = existing.to_dict("records")
    else:
        done = set()
        all_rows = []

    n = len(tickers)
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i+1}/{n}  ({ticker})")
        rows = fetch_earnings_history(ticker, args.api_key)
        all_rows.extend(rows)
        time.sleep(_DELAY)

    if not all_rows:
        print("No data downloaded.")
        return

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["ticker", "report_date"])
    df = df.sort_values(["ticker", "report_date"])
    df.to_csv(_OUT, index=False)
    print(f"\nSaved {len(df):,} rows ({df['ticker'].nunique()} tickers) → {_OUT}")
    print(f"Date range: {df['report_date'].min()} to {df['report_date'].max()}")


if __name__ == "__main__":
    main()
