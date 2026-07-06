"""
Phase 3/4 ablation runner on real SimFin data.

Phase 3 (E01-E10): ESQS / FTS / IFR / IVRS / HMM signals
Phase 4 (E11-E15): PEAD primary alpha + BSQ filter + beta-neutral sizing +
                   signed/accumulated IFR + idiosyncratic IVRS + SIC/mktcap universe fixes
"""
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Force UTF-8 output on Windows; also disable line buffering so progress prints
# appear immediately when stdout is redirected to a file.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
else:
    sys.stdout.reconfigure(line_buffering=True)
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as scipy_stats

from aria.backtest.engine import BacktestConfig, BacktestEngine
from aria.backtest.performance import PerformanceAnalytics
from aria.data.ingestion.price import PriceLoader
from aria.data.ingestion.yf_price_store import YFinancePriceStore
from aria.data.ingestion.simfin_loader import (
    load_income_statements, get_esqs_inputs,
    load_balance_sheets, load_cash_flows, get_bsq_inputs,
)
from aria.data.ingestion.edgar_loader import get_sic_map
from aria.data.ingestion.yfinance_earnings import load_earnings_dates, build_announce_map
from aria.data.ingestion.sue_loader import load_consensus, get_sue_inputs, compute_revision_dir
from aria.portfolio.scorer import CompositeScorer
from aria.research.ablation import ABLATION_MATRIX, AblationRunner, ExperimentSpec
from aria.signals.esqs import ESQSSignal
from aria.signals.ivrs import IVRSignal
from aria.signals.hmm import RollingRegimeHMM
from aria.signals.pead import PEADSignal
from aria.signals.bsq import BSQSignal
from aria.signals.sue import SUESignal
from aria.signals.base import cross_sectional_zscore

# SIC ranges excluded from universe
_SIC_FINANCIALS = set(range(6000, 7000))
_SIC_UTILITIES = set(range(4900, 5000))
_MIN_MARKET_CAP = 2e9  # $2B

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]

SECTOR_ETF_MAP: dict[str, str] = {
    # Tech
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOGL": "XLK", "META": "XLK",
    "AMZN": "XLK", "TSLA": "XLK", "AVGO": "XLK", "AMD": "XLK", "INTC": "XLK",
    "ADBE": "XLK", "CRM": "XLK", "ACN": "XLK", "CSCO": "XLK", "ORCL": "XLK",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "C": "XLF", "USB": "XLF", "PNC": "XLF", "TFC": "XLF", "COF": "XLF",
    "BLK": "XLF", "SCHW": "XLF", "AXP": "XLF",
    # Health Care
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "ABBV": "XLV", "MRK": "XLV",
    "TMO": "XLV", "ABT": "XLV", "DHR": "XLV", "BMY": "XLV", "AMGN": "XLV",
    "GILD": "XLV", "CVS": "XLV", "ISRG": "XLV",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",
    # Industrials
    "HON": "XLI", "UPS": "XLI", "CAT": "XLI", "DE": "XLI", "MMM": "XLI",
    "GE": "XLI", "LMT": "XLI", "RTX": "XLI",
    # Consumer
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP", "WMT": "XLP",
    "HD": "XLY", "NKE": "XLY", "MCD": "XLY", "SBUX": "XLY", "TGT": "XLY",
    # Communication
    "NFLX": "XLC", "DIS": "XLC", "T": "XLC",
    "VZ": "XLC", "CMCSA": "XLC",
}


def _next_weekday(d: date) -> date:
    """Return d if it's a weekday, else advance to the next Monday."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# SimFin price store
# ---------------------------------------------------------------------------

class SimFinPriceStore:
    def __init__(self, csv_path: str = "data/simfin/us-shareprices-daily.csv"):
        print(f"[Phase3] Loading SimFin prices ...")
        raw = pd.read_csv(csv_path, sep=";", usecols=[
            "Ticker", "Date", "Open", "Close", "Adj. Close", "Volume"
        ], parse_dates=["Date"])
        raw = raw.rename(columns={
            "Ticker": "ticker", "Date": "date", "Open": "open",
            "Close": "close", "Adj. Close": "adj_close", "Volume": "volume",
        })
        raw = raw.dropna(subset=["adj_close", "close"])
        raw["date"] = pd.to_datetime(raw["date"]).dt.date
        self._df = pl.from_pandas(raw).with_columns(pl.col("date").cast(pl.Date))
        print(f"[Phase3] Loaded {self._df.shape[0]:,} price rows, "
              f"{self._df['ticker'].n_unique()} tickers.")

    def get(self, tickers: Optional[list[str]] = None,
            start: Optional[date] = None, end: Optional[date] = None) -> pl.DataFrame:
        df = self._df
        if tickers:
            df = df.filter(pl.col("ticker").is_in(tickers))
        if start:
            df = df.filter(pl.col("date") >= start)
        if end:
            df = df.filter(pl.col("date") <= end)
        return df

    def liquid_tickers(self, min_adv_usd: float = 5e7) -> list[str]:
        adv = (
            self._df
            .with_columns((pl.col("close") * pl.col("volume")).alias("dolvol"))
            .group_by("ticker")
            .agg(pl.col("dolvol").mean().alias("adv"))
            .filter(pl.col("adv") >= min_adv_usd)
        )
        return adv["ticker"].to_list()


# ---------------------------------------------------------------------------
# Sector ETF volume cache
# ---------------------------------------------------------------------------

def load_sector_etf_volumes(start: str, end: str) -> pl.DataFrame:
    """Download and cache all sector ETF volumes. Returns [date, vol_XLK, vol_XLF, ...]."""
    cache_path = Path(f"data/parquet/sector_etfs_{start}_{end}.parquet")
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    loader = PriceLoader()
    frames = []
    for etf in SECTOR_ETFS:
        try:
            df = loader.load(etf, start, end)
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
    result = result.sort("date")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(cache_path)
    return result


# ---------------------------------------------------------------------------
# IFR with real sector ETF volumes
# ---------------------------------------------------------------------------

def build_ifr_with_sector_etfs(
    prices: pl.DataFrame,
    sector_volumes: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compute rolling-OLS volume residual using per-ticker sector ETF volume.
    Returns [ticker, date, IFR_z] (cross-sectional z-score per date).
    """
    df = prices.sort(["ticker", "date"])
    df = df.filter(pl.col("volume") > 0).with_columns([
        pl.col("volume").log().alias("log_vol")
    ])

    results = []
    for (ticker,), grp in df.group_by(["ticker"]):
        grp = grp.sort("date")
        etf = SECTOR_ETF_MAP.get(ticker, None)
        etf_col = f"vol_{etf}" if etf else None

        if etf_col and etf_col in sector_volumes.columns:
            sv = sector_volumes.select(["date", etf_col]).filter(pl.col(etf_col) > 0)
            sv = sv.with_columns(pl.col(etf_col).log().alias("log_etf"))
            merged = grp.join(sv.select(["date", "log_etf"]), on="date", how="inner")
            x_arr = merged["log_etf"].to_numpy()
        else:
            # Fallback: market-average log volume for that date
            mkt = (
                df.group_by("date")
                .agg(pl.col("log_vol").mean().alias("mkt_log_vol"))
            )
            merged = grp.join(mkt, on="date", how="left")
            x_arr = merged["mkt_log_vol"].to_numpy()

        y_arr = merged["log_vol"].to_numpy()
        dates = merged["date"].to_list()
        n = len(y_arr)
        resid = np.full(n, np.nan)
        window = 60
        for i in range(window - 1, n):
            yi = y_arr[i - window + 1: i + 1]
            xi = x_arr[i - window + 1: i + 1]
            valid = ~(np.isnan(yi) | np.isnan(xi))
            if valid.sum() < 30:
                continue
            yv, xv = yi[valid], xi[valid]
            var_x = np.var(xv)
            if var_x < 1e-12:
                continue
            beta = np.cov(xv, yv)[0, 1] / var_x
            alpha = yv.mean() - beta * xv.mean()
            resid[i] = y_arr[i] - (alpha + beta * x_arr[i])

        # Sign residuals by intraday direction (close vs open)
        if "close" in merged.columns and "open" in merged.columns:
            closes = merged["close"].to_numpy()
            opens = merged["open"].to_numpy()
            signs = np.where(closes != opens, np.sign(closes - opens).astype(float), 1.0)
            resid = resid * signs

        results.append(pl.DataFrame({"ticker": [ticker] * n, "date": dates, "IFR_raw": resid}))

    if not results:
        return pl.DataFrame({"ticker": [], "date": [], "IFR_z": []})

    combined = pl.concat(results)

    def _cs_z(grp: pl.DataFrame) -> pl.DataFrame:
        vals = grp["IFR_raw"].to_numpy().astype(float)
        valid = ~np.isnan(vals)
        z = vals.copy()
        if valid.sum() >= 3:
            mu, sd = vals[valid].mean(), vals[valid].std()
            if sd > 1e-10:
                z[valid] = np.clip((vals[valid] - mu) / sd, -3.0, 3.0)
        return grp.with_columns(pl.Series("IFR_z", z))

    combined = (
        combined
        .filter(pl.col("IFR_raw").is_not_null() & pl.col("IFR_raw").is_not_nan())
        .group_by("date")
        .map_groups(_cs_z)
        .sort(["date", "ticker"])
    )
    return combined.select(["ticker", "date", "IFR_z"]).filter(
        pl.col("IFR_z").is_not_null() & pl.col("IFR_z").is_not_nan()
    )


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def compute_ic(
    signals: pl.DataFrame,
    prices: pl.DataFrame,
    entry_date: date,
    forward_days: int = 5,
    signal_cols: list[str] = None,
) -> dict[str, float]:
    """
    Compute Information Coefficient for each signal column:
    Spearman rank correlation between signal z-score and forward return.
    """
    if signal_cols is None:
        signal_cols = ["ESQS_z", "IFR_z", "FTS_z"]

    fwd_end = entry_date + timedelta(days=forward_days * 2)
    ic = {}

    tickers = signals["ticker"].to_list()
    fwd_rets = {}
    for ticker in tickers:
        p = prices.filter(
            (pl.col("ticker") == ticker) & (pl.col("date") > entry_date)
        ).sort("date")
        if p.shape[0] >= forward_days:
            entry_price = None
            entry_row = prices.filter(
                (pl.col("ticker") == ticker) & (pl.col("date") == entry_date)
            )
            if not entry_row.is_empty():
                entry_price = float(entry_row["close"][0])
            if entry_price and entry_price > 0:
                exit_price = float(p["close"][forward_days - 1])
                fwd_rets[ticker] = (exit_price - entry_price) / entry_price

    if len(fwd_rets) < 5:
        return {col: float("nan") for col in signal_cols}

    ret_series = pd.Series(fwd_rets)
    for col in signal_cols:
        if col not in signals.columns:
            ic[col] = float("nan")
            continue
        sig_series = signals.select(["ticker", col]).to_pandas().set_index("ticker")[col]
        aligned = pd.concat([sig_series, ret_series], axis=1, join="inner").dropna()
        if len(aligned) < 5:
            ic[col] = float("nan")
        else:
            rho, _ = scipy_stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
            ic[col] = float(rho)

    return ic


# ---------------------------------------------------------------------------
# Main Phase 3 runner (v2)
# ---------------------------------------------------------------------------

class Phase3Runner:
    """
    Runs E01-E10 ablation matrix with per-event entries against SimFin data.
    """

    def __init__(
        self,
        price_csv: str = "data/simfin/us-shareprices-daily.csv",
        min_adv_usd: float = 5e7,
        min_event_pool: int = 5,
        hold_days: int = 5,
        top_pct: float = 0.10,
        bottom_pct: float = 0.10,
        initial_capital: float = 100_000_000,
        price_source: str = "simfin",  # "simfin" or "yfinance"
    ):
        self._price_source = price_source
        self._price_csv = price_csv
        # Always build SimFin store — used for liquid_tickers() universe determination.
        # In yfinance mode it is replaced by YFinancePriceStore after universe is known.
        self.price_store = SimFinPriceStore(price_csv)
        self.min_adv_usd = min_adv_usd
        self.min_event_pool = min_event_pool
        self.hold_days = hold_days
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.initial_capital = initial_capital
        self._pa = PerformanceAnalytics()
        self._rolling_hmm = RollingRegimeHMM()

    # ------------------------------------------------------------------
    # Phase 4 helpers
    # ------------------------------------------------------------------

    def _load_spy_returns(self, start: date, end: date) -> pl.DataFrame:
        """Load SPY daily log returns, cached locally."""
        cache_path = Path(f"data/parquet/spy_{start}_{end}.parquet")
        if cache_path.exists():
            return pl.read_parquet(cache_path)
        try:
            loader = PriceLoader()
            spy = loader.load("SPY", str(start), str(end))
            spy = (
                spy.sort("date")
                .with_columns(pl.col("adj_close").log().diff().alias("r_spy"))
                .select(["date", "r_spy"])
                .drop_nulls()
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            spy.write_parquet(cache_path)
            return spy
        except Exception as e:
            print(f"[Phase4] SPY load failed ({e}), beta-neutral sizing disabled")
            return pl.DataFrame()

    def _compute_betas(
        self,
        tickers: list[str],
        all_prices: pl.DataFrame,
        spy_returns: pl.DataFrame,
        as_of_date: date,
        window: int = 252,
    ) -> dict[str, float]:
        """Rolling beta vs SPY for a small set of tickers (called per event)."""
        if spy_returns.is_empty():
            return {t: 1.0 for t in tickers}
        spy_map = dict(zip(spy_returns["date"].to_list(), spy_returns["r_spy"].to_list()))
        betas: dict[str, float] = {}
        for ticker in tickers:
            p = (
                all_prices
                .filter((pl.col("ticker") == ticker) & (pl.col("date") <= as_of_date))
                .sort("date")
                .tail(window)
            )
            if p.shape[0] < 50:
                betas[ticker] = 1.0
                continue
            r = (
                p.with_columns(pl.col("adj_close").log().diff().alias("r"))
                ["r"].drop_nulls().to_numpy()
            )
            dates_arr = p["date"].to_list()[1:]  # diff shifts by 1
            r_spy = np.array([spy_map.get(d, float("nan")) for d in dates_arr])
            valid = ~(np.isnan(r) | np.isnan(r_spy))
            if valid.sum() < 30:
                betas[ticker] = 1.0
                continue
            rv, rs = r[valid], r_spy[valid]
            var_s = np.var(rs)
            if var_s < 1e-12:
                betas[ticker] = 1.0
            else:
                b = np.cov(rv, rs)[0, 1] / var_s
                betas[ticker] = max(0.2, min(3.0, abs(b)))
        return betas

    def _compute_ticker_vols(
        self,
        tickers: list[str],
        all_prices: pl.DataFrame,
        as_of_date: date,
        window: int = 21,
    ) -> dict[str, float]:
        """Trailing annualised realized vol per ticker for inverse-vol sizing."""
        vols: dict[str, float] = {}
        for ticker in tickers:
            p = (
                all_prices
                .filter((pl.col("ticker") == ticker) & (pl.col("date") <= as_of_date))
                .sort("date")
                .tail(window + 1)
            )
            if p.shape[0] < 10:
                vols[ticker] = 0.30
                continue
            r = (
                p.with_columns(pl.col("adj_close").log().diff().alias("_r"))
                ["_r"].drop_nulls().to_numpy()
            )
            vols[ticker] = float(np.std(r, ddof=1) * np.sqrt(252)) if len(r) >= 5 else 0.30
        return vols

    def _vol_target_scale(
        self,
        longs: list[str],
        shorts: list[str],
        ticker_vols: dict[str, float],
        target_vol: float,
        rho_within: float = 0.30,
        rho_ls: float = 0.30,
    ) -> float:
        n_long  = max(len(longs), 1)
        n_short = max(len(shorts), 1)
        avg_vol_long  = float(np.mean([ticker_vols.get(t, 0.30) for t in longs])) if longs else 0.30
        avg_vol_short = float(np.mean([ticker_vols.get(t, 0.30) for t in shorts])) if shorts else 0.30
        vol_L = avg_vol_long  * np.sqrt(rho_within + (1.0 - rho_within) / n_long)
        vol_S = avg_vol_short * np.sqrt(rho_within + (1.0 - rho_within) / n_short)
        port_vol_est = np.sqrt(vol_L**2 + vol_S**2 - 2.0 * rho_ls * vol_L * vol_S)
        return min(target_vol / max(port_vol_est, 0.01), 2.0)

    def _get_bsq_eligibility(
        self,
        event_date: date,
        tickers: list[str],
        income_df,
        balance_df,
        cashflow_df,
        bsq_signal: "BSQSignal",
        pead_z_map: dict[str, float],
    ) -> pl.DataFrame:
        """Compute BSQ eligibility for the current earnings cohort."""
        rows = []
        for ticker in tickers:
            inp = get_bsq_inputs(ticker, event_date, income_df, balance_df, cashflow_df)
            if inp:
                rows.append(inp)
        if not rows:
            return pl.DataFrame()
        return bsq_signal.compute_batch(rows, pead_z_map=pead_z_map)

    # ------------------------------------------------------------------

    def run(
        self,
        start_date: date = date(2021, 1, 1),
        end_date: date = date(2024, 12, 31),
        experiments: Optional[list[ExperimentSpec]] = None,
    ) -> AblationRunner:
        if experiments is None:
            experiments = ABLATION_MATRIX

        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[Phase3/4] {len(experiments)} experiments | {start_date} to {end_date}")
        print(f"{'='*60}\n")

        # --- income data ---
        print("[Phase3] Loading income statements...")
        income_df = load_income_statements()
        inc_flat = income_df.reset_index() if income_df.index.names[0] is not None else income_df
        if "Ticker" not in inc_flat.columns:
            inc_flat = income_df.reset_index()

        price_start = start_date - timedelta(days=600)

        # --- base universe (ADV + income coverage) ---
        liquid = set(self.price_store.liquid_tickers(self.min_adv_usd))
        inc_tickers = set(inc_flat["Ticker"].unique())
        universe = sorted(liquid & inc_tickers)
        print(f"[Phase3] Pre-filter universe: {len(universe)} tickers")

        # --- SIC exclusion (financials 6000-6999, utilities 4900-4999) ---
        try:
            sic_map = get_sic_map()
            excluded_sic = _SIC_FINANCIALS | _SIC_UTILITIES
            before = len(universe)
            universe = [t for t in universe if sic_map.get(t, 0) not in excluded_sic]
            print(f"[Phase4] SIC filter: {before} -> {len(universe)} tickers")
        except Exception as e:
            print(f"[Phase4] SIC map unavailable ({e}), skipping SIC filter")

        # --- market cap floor ($2B = shares_basic × close) ---
        try:
            if "Shares (Basic)" in inc_flat.columns:
                inc_flat_copy2 = inc_flat.copy()
                inc_flat_copy2["Publish Date"] = pd.to_datetime(
                    inc_flat_copy2["Publish Date"]
                ).dt.date
                recent_shares = (
                    inc_flat_copy2[inc_flat_copy2["Ticker"].isin(set(universe))]
                    .sort_values("Publish Date")
                    .groupby("Ticker")["Shares (Basic)"]
                    .last()
                )
                recent_close_df = (
                    self.price_store.get(tickers=universe, end=end_date)
                    .sort("date")
                    .group_by("ticker")
                    .tail(1)
                    .select(["ticker", "close"])
                )
                close_map = dict(
                    zip(recent_close_df["ticker"].to_list(),
                        recent_close_df["close"].to_list())
                )
                before = len(universe)
                universe = [
                    t for t in universe
                    if close_map.get(t) is not None
                    and not pd.isna(recent_shares.get(t, float("nan")))
                    and float(recent_shares.get(t, 0)) * float(close_map.get(t, 0))
                    >= _MIN_MARKET_CAP
                ]
                print(f"[Phase4] MktCap filter: {before} -> {len(universe)} tickers ($2B+ floor)")
            else:
                print("[Phase4] Shares (Basic) column not found, skipping mktcap filter")
        except Exception as e:
            print(f"[Phase4] MktCap filter failed ({e}), skipping")

        # --- swap to yfinance price store now that universe is finalised ---
        # SimFin CSV only covers ~2020-08 to ~2023-10; yfinance provides full 2018-2024.
        if self._price_source == "yfinance":
            yf_store = YFinancePriceStore(price_start, end_date)
            yf_store.ensure_built(universe)
            self.price_store = yf_store

        # --- price history (+ buffer for rolling signals) ---
        all_prices = self.price_store.get(tickers=universe, start=price_start, end=end_date)
        print(f"[Phase3] Price rows: {all_prices.shape[0]:,}")

        # --- sector ETF volumes ---
        print("[Phase3] Loading sector ETF volumes...")
        sector_volumes = load_sector_etf_volumes(
            str(price_start), str(end_date + timedelta(days=30))
        )
        print(f"[Phase3] Sector ETF rows: {sector_volumes.shape[0]}")

        # --- IFR (pre-compute once; now signed by intraday direction) ---
        print("[Phase3] Computing IFR (sector ETF regression, signed)...")
        ifr_df = build_ifr_with_sector_etfs(all_prices, sector_volumes)
        print(f"[Phase3] IFR rows: {ifr_df.shape[0]:,}")

        # --- IVRS (pre-compute once) ---
        print("[Phase3] Computing IVRS...")
        try:
            ivrs_df = IVRSignal().compute(all_prices)
            print(f"[Phase3] IVRS rows: {ivrs_df.shape[0]:,}")
        except Exception as e:
            print(f"[Phase3] IVRS failed ({e}), zeroing")
            ivrs_df = pl.DataFrame()

        # --- determine which Phase 4/5 features are needed ---
        has_pead = any("PEAD_z" in exp.signals for exp in experiments)
        needs_bsq = any(exp.bsq_filter or "BSQ_z" in exp.signals for exp in experiments)
        needs_beta_neutral = any(exp.beta_neutral for exp in experiments)
        has_sue = any("SUE_z" in exp.signals for exp in experiments)
        needs_vol_target = any(exp.vol_target > 0 for exp in experiments)

        # --- SPY returns (for beta-neutral sizing) ---
        spy_returns = pl.DataFrame()
        if needs_beta_neutral:
            print("[Phase4] Loading SPY returns...")
            spy_returns = self._load_spy_returns(price_start, end_date)
            print(f"[Phase4] SPY rows: {spy_returns.shape[0]}")

        # --- balance sheets + cash flows (for BSQ) ---
        balance_df = None
        cashflow_df = None
        bsq_signal = None
        if needs_bsq:
            print("[Phase4] Loading balance sheets and cash flows for BSQ...")
            try:
                balance_df = load_balance_sheets()
                cashflow_df = load_cash_flows()
                bsq_signal = BSQSignal()
                print("[Phase4] Balance/cashflow data loaded.")
            except Exception as e:
                print(f"[Phase4] BSQ data load failed ({e}), BSQ filter disabled")
                needs_bsq = False

        # --- analyst consensus data (for SUE) ---
        consensus_df = None
        sue_signal = None
        if has_sue:
            print("[Phase5] Loading analyst consensus data for SUE...")
            try:
                consensus_df = load_consensus()
                if consensus_df is None:
                    print("[Phase5] No consensus CSV found at data/consensus/eps_consensus.csv "
                          "— SUE_z will be zeroed. Add paid consensus data to enable this signal.")
                    has_sue = False
                else:
                    sue_signal = SUESignal()
                    print(f"[Phase5] Consensus loaded: {consensus_df.shape[0]:,} rows, "
                          f"{consensus_df['ticker'].n_unique()} tickers.")
            except Exception as e:
                print(f"[Phase5] Consensus load failed ({e}), SUE disabled")
                has_sue = False

        # --- per-event earnings calendar ---
        inc_flat_copy = inc_flat.copy()
        inc_flat_copy["Publish Date"] = pd.to_datetime(inc_flat_copy["Publish Date"]).dt.date
        events_all = inc_flat_copy[
            inc_flat_copy["Ticker"].isin(set(universe)) &
            (inc_flat_copy["Publish Date"] >= start_date) &
            (inc_flat_copy["Publish Date"] <= end_date)
        ][["Ticker", "Publish Date"]].drop_duplicates().sort_values("Publish Date")
        events_all = events_all.rename(
            columns={"Ticker": "ticker", "Publish Date": "publish_date"}
        )
        event_by_date: dict[date, list[str]] = {}
        for _, row in events_all.iterrows():
            d = row["publish_date"]
            event_by_date.setdefault(d, [])
            if row["ticker"] not in event_by_date[d]:
                event_by_date[d].append(row["ticker"])

        sorted_dates = sorted(event_by_date)
        print(f"[Phase3] Event dates: {len(sorted_dates)}, total events: {events_all.shape[0]}")

        # --- Actual announcement dates for PEAD (yfinance vs SimFin publish lag) ---
        # SimFin Publish Date lags actual announcement by 2-5 days; entering on
        # pub_date+1 misses most of the PEAD drift.  yfinance gives us the real
        # announcement date so we can enter at announce_date+1 instead.
        announce_event_by_date: dict[date, list[str]] = {}
        if has_pead:
            print("[Phase4] Loading actual earnings announcement dates (yfinance cache)...")
            try:
                yf_dates = load_earnings_dates(universe, verbose=False)
                ticker_pub_pairs = [
                    (row["ticker"], row["publish_date"])
                    for _, row in events_all.iterrows()
                ]
                ann_map = build_announce_map(ticker_pub_pairs, yf_dates)
                for (ticker, _pub), ann_date in ann_map.items():
                    announce_event_by_date.setdefault(ann_date, [])
                    if ticker not in announce_event_by_date[ann_date]:
                        announce_event_by_date[ann_date].append(ticker)
                n_improved = sum(1 for (_, p), a in ann_map.items() if a != p)
                print(f"[Phase4] Announce map: {len(ann_map)} pairs, "
                      f"{n_improved} corrected from SimFin publish date")
            except Exception as e:
                print(f"[Phase4] yfinance dates unavailable ({e}); "
                      f"falling back to SimFin publish dates for PEAD")
                announce_event_by_date = {d: list(ts) for d, ts in event_by_date.items()}

        # --- PEAD pre-computation (batch over actual announcement dates) ---
        # pead_by_date is keyed by entry_date (announce+1 business day) so the
        # outer loop can fire on the correct entry date directly.
        pead_by_date: dict[date, pl.DataFrame] = {}
        if has_pead:
            print("[Phase4] Pre-computing PEAD signal (batch)...")
            try:
                pead_prices = self.price_store.get(
                    tickers=universe,
                    start=start_date - timedelta(days=10),
                    end=end_date,
                )
                pead_event_src = announce_event_by_date if announce_event_by_date else event_by_date
                pead_full = PEADSignal().compute_batch(pead_prices, pead_event_src)
                if not pead_full.is_empty():
                    for (ed,), grp in pead_full.group_by(["entry_date"]):
                        pead_by_date[ed] = grp
                    print(f"[Phase4] PEAD: {pead_full.shape[0]} rows, "
                          f"{len(pead_by_date)} entry dates")
                else:
                    print("[Phase4] PEAD returned empty; PEAD experiments may yield no trades")
            except Exception as e:
                print(f"[Phase4] PEAD computation failed ({e}), has_pead=False")
                has_pead = False

        # --- per-experiment trade accumulator + IC tracker ---
        exp_trades: dict[str, list[pl.DataFrame]] = {e.name: [] for e in experiments}
        ic_log: dict[str, list[dict]] = {e.name: [] for e in experiments}

        # Calendar of all business days
        cal_dates = []
        d = start_date
        while d <= end_date:
            if d.weekday() < 5:
                cal_dates.append(d)
            d += timedelta(days=1)

        # Cache mkt_prices aggregate (expensive to recompute every event)
        mkt_prices = (
            all_prices
            .group_by("date")
            .agg(pl.col("adj_close").mean().alias("adj_close"))
            .sort("date")
        )

        # Union of SimFin publish dates (non-PEAD signals) and actual PEAD entry dates
        pead_entry_date_set: set[date] = set(pead_by_date.keys()) if has_pead else set()
        all_sorted_dates = sorted(set(sorted_dates) | pead_entry_date_set)
        n_dates = len(all_sorted_dates)
        print(f"[Phase3] Processing {n_dates} event dates "
              f"({len(sorted_dates)} SimFin + {len(pead_entry_date_set)} PEAD)...\n")

        for idx, entry_date in enumerate(all_sorted_dates):
            is_simfin_date = entry_date in event_by_date
            is_pead_entry = entry_date in pead_by_date

            if not is_simfin_date and not is_pead_entry:
                continue

            # Determine ticker pool for this date
            if is_simfin_date:
                win_tickers = event_by_date[entry_date]
            else:
                win_tickers = pead_by_date[entry_date]["ticker"].to_list()

            if len(win_tickers) < self.min_event_pool:
                continue

            # ESQS (only on SimFin publish dates; pure PEAD entry dates have no income data yet)
            if is_simfin_date:
                esqs_rows = []
                for t in win_tickers:
                    inp = get_esqs_inputs(t, entry_date, income_df)
                    if inp:
                        esqs_rows.append(inp)
                if len(esqs_rows) < self.min_event_pool:
                    continue
                esqs_raw = pl.DataFrame(esqs_rows)
                try:
                    esqs_df = ESQSSignal().compute_normalized(esqs_raw)
                except Exception:
                    continue
                present = esqs_df["ticker"].to_list()
            else:
                # Pure PEAD entry date: no SimFin data available yet; use PEAD tickers directly
                present = win_tickers
                esqs_df = pl.DataFrame({
                    "ticker": present,
                    "ESQS_z": [0.0] * len(present),
                })

            # IVRS latest (use date < entry_date: IVRS uses end-of-day adj_close;
            # since entry is at open of entry_date, only prior days are available)
            if not ivrs_df.is_empty():
                ivrs_latest = (
                    ivrs_df
                    .filter(pl.col("ticker").is_in(present) & (pl.col("date") < entry_date))
                    .sort("date")
                    .group_by("ticker")
                    .tail(1)
                    .select(["ticker", "IVRS_z"])
                )
            else:
                ivrs_latest = pl.DataFrame()

            # IFR latest (use date < entry_date: signed IFR uses same-day close;
            # since entry is at open, we can only use the previous day's signal)
            ifr_latest = (
                ifr_df
                .filter(pl.col("ticker").is_in(present) & (pl.col("date") < entry_date))
                .sort("date")
                .group_by("ticker")
                .tail(1)
                .select(["ticker", "IFR_z"])
            ) if not ifr_df.is_empty() else pl.DataFrame()

            # Base frame (all signals start at 0; joined below)
            base = esqs_df.select(["ticker", "ESQS_z"]).with_columns([
                pl.lit(0.0).alias("IVRS_z"),
                pl.lit(0.0).alias("IFR_z"),
                pl.lit(0.0).alias("FTS_z"),
                pl.lit(0.0).alias("RMV_z"),
                pl.lit(0.0).alias("PEAD_z"),
                pl.lit(0.0).alias("BSQ_z"),
                pl.lit(0.0).alias("SUE_z"),
            ])
            if not ivrs_latest.is_empty() and "IVRS_z" in ivrs_latest.columns:
                base = base.join(ivrs_latest, on="ticker", how="left", suffix="_new")
                if "IVRS_z_new" in base.columns:
                    base = base.with_columns(
                        pl.col("IVRS_z_new").fill_null(pl.col("IVRS_z")).alias("IVRS_z")
                    ).drop("IVRS_z_new")
            if not ifr_latest.is_empty():
                base = base.join(ifr_latest, on="ticker", how="left", suffix="_new")
                if "IFR_z_new" in base.columns:
                    base = base.with_columns(
                        pl.col("IFR_z_new").fill_null(pl.col("IFR_z")).alias("IFR_z")
                    ).drop("IFR_z_new")

            # PEAD signal for this event date
            # pead_by_date is keyed by entry_date (announce+1), so on actual PEAD entry
            # dates entry_date IS the correct trade entry; no further offset needed.
            pead_entry_date = entry_date  # default; overridden for pure-PEAD dates
            if has_pead and is_pead_entry:
                pead_cohort = pead_by_date[entry_date]
                base = base.join(
                    pead_cohort.select(["ticker", "PEAD_z"]),
                    on="ticker", how="left", suffix="_new"
                )
                if "PEAD_z_new" in base.columns:
                    base = base.with_columns(
                        pl.col("PEAD_z_new").fill_null(0.0).alias("PEAD_z")
                    ).drop("PEAD_z_new")
            elif is_simfin_date:
                # SimFin-only date: no corrected PEAD; PEAD experiments skip below
                pead_entry_date = _next_weekday(entry_date + timedelta(days=1))

            # BSQ eligibility for this cohort
            bsq_elig_df = pl.DataFrame()
            if needs_bsq and bsq_signal is not None:
                # PEAD_z < 0 short constraint only applies on actual PEAD entry
                # dates where a gap-down direction is known.  On SimFin-only
                # dates (or when PEAD isn't in the experiment) pass None so BSQ
                # short eligibility falls back to BSQ score alone.
                pead_z_map = (
                    dict(zip(base["ticker"].to_list(), base["PEAD_z"].to_list()))
                    if is_pead_entry else None
                )
                try:
                    bsq_elig_df = self._get_bsq_eligibility(
                        entry_date, present, income_df, balance_df,
                        cashflow_df, bsq_signal, pead_z_map
                    )
                except Exception:
                    pass
                # Expose BSQ_score as a direction signal (BSQ_z) for signal-based experiments
                if not bsq_elig_df.is_empty():
                    bsq_z_df = bsq_elig_df.select(["ticker", "BSQ_score"]).rename({"BSQ_score": "BSQ_z"})
                    base = base.join(bsq_z_df, on="ticker", how="left", suffix="_new")
                    if "BSQ_z_new" in base.columns:
                        base = base.with_columns(
                            pl.col("BSQ_z_new").fill_null(0.0).alias("BSQ_z")
                        ).drop("BSQ_z_new")

            # SUE signal for this cohort (requires paid consensus data)
            if has_sue and sue_signal is not None and consensus_df is not None:
                sue_rows = get_sue_inputs(present, entry_date, consensus_df)
                if len(sue_rows) >= 3:
                    sue_df = sue_signal.compute_normalized(sue_rows)
                    base = base.join(sue_df, on="ticker", how="left", suffix="_new")
                    if "SUE_z_new" in base.columns:
                        base = base.with_columns(
                            pl.col("SUE_z_new").fill_null(0.0).alias("SUE_z")
                        ).drop("SUE_z_new")

            # Compute revision direction map for this cohort (used in Phase 2 sizing)
            revision_dir_map: dict[str, float] = {}
            if has_sue and consensus_df is not None:
                for t in present:
                    revision_dir_map[t] = compute_revision_dir(t, entry_date, consensus_df)

            # Prices for exit window (generous buffer for 20d hold)
            max_hold = max(self.hold_days, 25)
            win_prices = self.price_store.get(
                tickers=present,
                start=entry_date,
                end=entry_date + timedelta(days=max_hold * 3),
            ).with_columns(
                (pl.col("close") * pl.col("volume")).alias("adv_20d_usd")
            )

            # IC computation (track all signals)
            ic_signal_cols = ["ESQS_z", "IFR_z", "FTS_z", "PEAD_z", "BSQ_z", "SUE_z"]
            ic_signals = base.select(
                ["ticker"] + [c for c in ic_signal_cols if c in base.columns]
            )
            ic_result = compute_ic(
                ic_signals, win_prices, entry_date,
                max(self.hold_days, 20), ic_signal_cols
            )

            # Same-day return map for PEAD confirmation gate (price reaction on event date)
            pead_gate_needed = any(e.pead_gate for e in experiments)
            pead_gate_map: dict[str, float] = {}
            if pead_gate_needed:
                prev_p = (
                    all_prices
                    .filter(pl.col("ticker").is_in(win_tickers) & (pl.col("date") <= entry_date))
                    .sort("date")
                    .group_by("ticker")
                    .tail(2)
                )
                for tkr in win_tickers:
                    closes = prev_p.filter(pl.col("ticker") == tkr)["adj_close"].to_list()
                    if len(closes) >= 2 and closes[-2] > 0:
                        pead_gate_map[tkr] = (closes[-1] - closes[-2]) / closes[-2]

            # Rolling HMM regime
            regime_ok = self._rolling_hmm.is_allowed(mkt_prices, entry_date)

            for exp in experiments:
                if exp.regime_filter and not regime_ok:
                    continue

                is_pead_exp = "PEAD_z" in exp.signals

                # PEAD experiments run only on actual PEAD entry dates.
                # Non-PEAD experiments run only on SimFin publish dates.
                if is_pead_exp and not is_pead_entry:
                    continue
                if not is_pead_exp and not is_simfin_date:
                    continue

                effective_entry = pead_entry_date if is_pead_exp else entry_date
                effective_hold = exp.hold_days if exp.hold_days != 10 else self.hold_days

                config = BacktestConfig(
                    hold_days=effective_hold,
                    initial_capital=self.initial_capital,
                    stop_loss_pct=exp.stop_loss_pct,
                    trailing_stop_pct=exp.trailing_stop_pct,
                    scaled_exit=exp.scaled_exit,
                    leg1_target=exp.leg1_target,
                    leg2_target=exp.leg2_target,
                )

                # Map RMV_z -> FTS_z
                weights = {}
                for sig, w in exp.weights.items():
                    mapped = "FTS_z" if sig == "RMV_z" else sig
                    weights[mapped] = weights.get(mapped, 0) + w

                scorer = CompositeScorer(weights=weights)
                scored = scorer.score(base)

                if exp.ivrs_multiplier and "IVRS_z" in scored.columns:
                    scored = scored.with_columns([
                        (pl.col("composite") *
                         (1.0 - 0.2 * pl.col("IVRS_z")).clip(0.5, 1.5)
                         ).alias("composite")
                    ])

                longs, shorts = scorer.select_long_short(scored, self.top_pct, self.bottom_pct)

                # |SUE_z| threshold: drop near-zero signals
                if exp.min_sue_z > 0 and "SUE_z" in base.columns:
                    sue_abs = dict(zip(base["ticker"].to_list(),
                                       [abs(v) for v in base["SUE_z"].to_list()]))
                    longs  = [t for t in longs  if sue_abs.get(t, 0.0) >= exp.min_sue_z]
                    shorts = [t for t in shorts if sue_abs.get(t, 0.0) >= exp.min_sue_z]

                # PEAD gate: only enter when same-day price return confirms SUE direction
                if exp.pead_gate and pead_gate_map:
                    threshold = exp.min_pead_ret
                    longs  = [t for t in longs
                               if pead_gate_map.get(t, 0.0) >= threshold]
                    shorts = [t for t in shorts
                               if pead_gate_map.get(t, 0.0) <= -threshold]

                # BSQ filter
                if exp.bsq_filter and not bsq_elig_df.is_empty():
                    longs, shorts = bsq_signal.apply_filter(longs, shorts, bsq_elig_df)

                if not longs or not shorts:
                    continue

                # Position sizing: vol-target, beta-neutral, or equal-weight
                if exp.vol_target > 0:
                    ticker_vols = self._compute_ticker_vols(
                        longs + shorts, all_prices, entry_date
                    )
                    # Inverse-vol base weights
                    lw = {t: 1.0 / max(ticker_vols.get(t, 0.30), 0.05) for t in longs}
                    sw = {t: 1.0 / max(ticker_vols.get(t, 0.30), 0.05) for t in shorts}

                    # SUE magnitude tilt: |SUE_z| ∈ [0.5, 3.0]
                    if "SUE_z" in exp.signals:
                        sue_z_map = dict(zip(base["ticker"].to_list(), base["SUE_z"].to_list()))
                        for t in longs:
                            lw[t] *= float(np.clip(abs(sue_z_map.get(t, 1.0)), 0.5, 3.0))
                        for t in shorts:
                            sw[t] *= float(np.clip(abs(sue_z_map.get(t, 1.0)), 0.5, 3.0))

                    # Revision direction tilt (E24/E25)
                    if exp.use_revision_weight and revision_dir_map:
                        filtered_longs = []
                        for t in longs:
                            rd = revision_dir_map.get(t, 0.0)
                            if rd >= -0.5:
                                lw[t] *= (1.0 + 0.5 * rd)
                                filtered_longs.append(t)
                        longs = filtered_longs
                        for t in shorts:
                            rd = revision_dir_map.get(t, 0.0)
                            sw[t] *= (1.0 - 0.5 * rd)
                        if not longs:
                            continue

                    # Normalise each side to sum=1
                    lt = sum(lw[t] for t in longs)
                    st = sum(sw.values())
                    long_w  = {t: lw[t] / lt for t in longs}
                    short_w = {t: v / st for t, v in sw.items()}

                    # Scale to hit vol target using corrected L/S formula
                    vol_tgt = exp.vol_target
                    if exp.concurrent_vol_adjust:
                        hold_cal = int(effective_hold * 1.45)
                        n_concurrent = sum(
                            1 for d in all_sorted_dates
                            if entry_date - timedelta(days=hold_cal) <= d <= entry_date
                        )
                        rho = exp.rho_cross_cohort
                        # Floor at 1.0: rho=-0.05 is invalid for n>21; effective diversification can't exceed single-cohort
                        eff_var = max(n_concurrent * (1.0 + (n_concurrent - 1) * rho), 1.0)
                        denom = np.sqrt(eff_var)
                        vol_tgt = exp.vol_target / denom
                    scale   = self._vol_target_scale(longs, shorts, ticker_vols, vol_tgt)
                    long_w  = {t: w * scale for t, w in long_w.items()}
                    short_w = {t: w * scale for t, w in short_w.items()}
                elif exp.beta_neutral and not spy_returns.is_empty():
                    betas = self._compute_betas(
                        longs + shorts, all_prices, spy_returns, entry_date
                    )
                    lw = {t: 1.0 / betas[t] for t in longs}
                    sw = {t: 1.0 / betas[t] for t in shorts}
                    lt = sum(lw.values())
                    st = sum(sw.values())
                    long_w = {t: v / lt for t, v in lw.items()}
                    short_w = {t: v / st for t, v in sw.items()}
                else:
                    nl, ns = len(longs), len(shorts)
                    long_w = {t: 1.0 / nl for t in longs}
                    short_w = {t: 1.0 / ns for t in shorts}

                signal_rows = (
                    [{"ticker": t, "entry_date": effective_entry,
                      "side": "long", "weight": long_w[t]} for t in longs] +
                    [{"ticker": t, "entry_date": effective_entry,
                      "side": "short", "weight": short_w[t]} for t in shorts]
                )
                signals_df = pl.DataFrame(signal_rows).with_columns(
                    pl.col("entry_date").cast(pl.Date)
                )

                # Prices window aligned to actual entry date
                if is_pead_exp and effective_entry != entry_date:
                    exp_prices = self.price_store.get(
                        tickers=present,
                        start=effective_entry,
                        end=effective_entry + timedelta(days=effective_hold * 3),
                    ).with_columns(
                        (pl.col("close") * pl.col("volume")).alias("adv_20d_usd")
                    )
                else:
                    exp_prices = win_prices

                engine = BacktestEngine(config=config)
                trades = engine.run(signals=signals_df, prices=exp_prices)
                if trades.shape[0] > 0:
                    exp_trades[exp.name].append(trades)
                    ic_log[exp.name].append({"date": entry_date, **ic_result})

            if (idx + 1) % 100 == 0:
                pct = (idx + 1) / n_dates * 100
                print(f"  [{pct:5.1f}%] {idx+1}/{n_dates}  ({entry_date})")

        print(f"\n[Phase3/4] Backtest done in {time.time()-t0:.1f}s. Computing metrics...\n")

        ablation = AblationRunner(experiments=experiments)

        for exp in experiments:
            all_trades = exp_trades[exp.name]
            if not all_trades:
                ablation.record(exp.name, {"sharpe": float("nan"), "note": "no_trades"})
                continue

            combined = pl.concat(all_trades)
            n_trades = combined.shape[0]
            combined = combined.with_columns(
                pl.col("exit_date").cast(pl.Utf8).str.to_date("%Y-%m-%d").alias("exit_dt")
            )
            pnl_map = dict(
                combined.group_by("exit_dt").agg(pl.col("pnl").sum()).iter_rows()
            )
            rets = np.array(
                [pnl_map.get(d, 0.0) / self.initial_capital for d in cal_dates]
            )
            metrics = self._pa.summarize(rets)

            wins = int((combined["pnl"] > 0).sum())
            win_pnl  = combined.filter(pl.col("pnl") > 0)["pnl"]
            loss_pnl = combined.filter(pl.col("pnl") < 0)["pnl"]
            avg_win  = float(win_pnl.mean())  if win_pnl.len()  > 0 else float("nan")
            avg_loss = float(loss_pnl.mean()) if loss_pnl.len() > 0 else float("nan")
            rr = abs(avg_win / avg_loss) if avg_loss != 0 and not np.isnan(avg_loss) else float("nan")
            metrics["n_trades"]      = n_trades
            metrics["win_rate"]      = round(wins / n_trades, 3) if n_trades > 0 else float("nan")
            metrics["avg_win"]       = round(avg_win,  2) if not np.isnan(avg_win)  else float("nan")
            metrics["avg_loss"]      = round(avg_loss, 2) if not np.isnan(avg_loss) else float("nan")
            metrics["rr_ratio"]      = round(rr, 3)       if not np.isnan(rr)       else float("nan")
            metrics["rmv_available"] = False
            metrics["regime_filter"] = exp.regime_filter
            metrics["ivrs_multiplier"] = exp.ivrs_multiplier
            metrics["bsq_filter"] = exp.bsq_filter
            metrics["beta_neutral"] = exp.beta_neutral

            ic_entries = ic_log[exp.name]
            for sig in ["ESQS_z", "IFR_z", "FTS_z", "PEAD_z", "BSQ_z", "SUE_z"]:
                vals = [
                    e[sig] for e in ic_entries
                    if not np.isnan(e.get(sig, float("nan")))
                ]
                metrics[f"IC_{sig}"] = round(float(np.mean(vals)), 4) if vals else float("nan")

            ablation.record(exp.name, metrics)

            sr = metrics.get("sharpe", float("nan"))
            ar = metrics.get("annual_return", float("nan"))
            dd = metrics.get("max_drawdown", float("nan"))
            if "SUE_z" in exp.signals:
                ic_key = "IC_SUE_z"
            elif "PEAD_z" in exp.signals:
                ic_key = "IC_PEAD_z"
            elif "BSQ_z" in exp.signals:
                ic_key = "IC_BSQ_z"
            else:
                ic_key = "IC_ESQS_z"
            ic_val  = metrics.get(ic_key, float("nan"))
            wr      = metrics.get("win_rate", float("nan"))
            rr_val  = metrics.get("rr_ratio",  float("nan"))
            vol_str = f"  VolTgt={exp.vol_target:.0%}" if exp.vol_target > 0 else ""
            print(f"  {exp.name:<32}  Sharpe={sr:+.2f}  AnnRet={ar:+.1%}  "
                  f"MaxDD={dd:.1%}  N={n_trades}  IC={ic_val:+.3f}  "
                  f"WR={wr:.1%}  RR={rr_val:.2f}{vol_str}")

        print(f"\n{'='*60}")
        print("[Phase3/4] Done.")
        return ablation


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ARIA Phase 3 ablation experiments")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--exp",   default=None, help="Comma-separated experiment names")
    parser.add_argument("--price-source", default="simfin", choices=["simfin", "yfinance"],
                        help="Price data source (simfin=CSV, yfinance=downloaded via API)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    exps  = None
    if args.exp:
        names = set(args.exp.split(","))
        exps = [e for e in ABLATION_MATRIX if e.name in names]

    runner = Phase3Runner(price_source=args.price_source)
    ablation = runner.run(start_date=start, end_date=end, experiments=exps)

    df = ablation.summary_df()
    cols = [
        "experiment", "sharpe", "annual_return", "annual_vol", "max_drawdown",
        "n_trades", "win_rate", "IC_ESQS_z", "IC_IFR_z", "IC_PEAD_z", "IC_BSQ_z", "IC_SUE_z",
        "bsq_filter", "beta_neutral", "vol_target",
    ]
    print("\n=== ABLATION SUMMARY ===")
    with pl.Config(tbl_rows=20, float_precision=3):
        print(df.select([c for c in cols if c in df.columns]))

    best = ablation.best_by("sharpe")
    if best:
        print(f"\nBest by Sharpe: {best.name}")
