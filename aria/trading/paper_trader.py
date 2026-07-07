#!/usr/bin/env python3
"""
ARIA E49 Paper Trader
=====================
Runs daily after market open (9:45-10:00 AM ET).

Implements the E49 PEAD strategy:
  - Detects earnings via SEC EDGAR 8-K filings (actual announcement day)
  - Computes cross-sectional PEAD_z scores across fresh reporters
  - Selects top 5% with >= 3% earnings-day reaction (long) / <= -3% (short)
  - Sizes positions using 6% annual vol target
  - Exits on -6% hard stop OR 30-day time stop
  - Sends structured Discord alerts

Setup:
  1. Set DISCORD_WEBHOOK_URL env var (create a webhook in Discord channel settings)
  2. Optionally set ARIA_CAPITAL (default 50000)
  3. Run: python -m aria.trading.paper_trader
  4. Schedule with Windows Task Scheduler at 9:45 AM ET on weekdays

Backtest results (E49, 2009-2024):
  WR=60.9%  RR=1.79  Sharpe=2.84  MaxDD=-12.0%  N=1790 trades
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import yfinance as yf

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).parent.parent   # project root (aria package dir)
_TRADING_DIR = _ROOT / "trading"

CFG = {
    # Strategy parameters (must match E49 backtest)
    "initial_capital":       float(os.environ.get("ARIA_CAPITAL", "50000")),
    "vol_target":            0.06,    # 6% annual vol per position
    "stop_pct":              0.06,    # 6% hard stop
    "hold_days":             30,      # max hold
    "min_pead_ret":          0.03,    # 3% earnings-day reaction gate
    "top_pct":               0.05,    # top/bottom 5%
    "max_positions":         10,
    "long_only":             True,    # set False to enable shorts

    # EDGAR
    "edgar_cache":           _TRADING_DIR / "edgar_cache",
    "cik_map":               _ROOT / "data" / "edgar" / "ticker_cik_map.json",
    "min_8k_after_qtr":      15,      # earliest 8-K days after quarter end
    "max_8k_after_qtr":      55,      # latest 8-K days after quarter end
    "lookback_days":         3,       # scan this many market days back for fresh 8-Ks

    # State / logs
    "state_file":            _TRADING_DIR / "state.json",
    "nav_history":           _TRADING_DIR / "nav_history.json",
    "log_file":              _TRADING_DIR / "paper_trader.log",

    # Discord
    "discord_webhook":       os.environ.get("DISCORD_WEBHOOK_URL", ""),
    "strategy_name":         "ARIA E49",
}

# US market holidays (major ones; extend as needed)
_HOLIDAYS: set[date] = {
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),  date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),  date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5),  date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}


def is_market_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _HOLIDAYS


def prev_market_day(d: date, n: int = 1) -> date:
    for _ in range(n):
        d -= timedelta(days=1)
        while not is_market_day(d):
            d -= timedelta(days=1)
    return d


def _day_label(d: date) -> str:
    """'Mon Jul 7' — cross-platform (no %-d)."""
    return d.strftime("%a %b ") + str(d.day)


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    entry_date: str      # ISO YYYY-MM-DD
    stop_price: float
    direction: str       # "long" | "short"


@dataclass
class ClosedTrade:
    ticker: str
    direction: str
    shares: float
    entry_price: float
    exit_price: float
    entry_date: str
    exit_date: str
    pnl: float
    pnl_pct: float
    held_days: int
    exit_reason: str     # "hard_stop" | "time_stop"


@dataclass
class State:
    cash: float
    initial_capital: float
    positions: list[Position] = field(default_factory=list)
    closed_trades: list[ClosedTrade] = field(default_factory=list)

    # ---- persistence ----

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "initial_capital": self.initial_capital,
            "positions": [asdict(p) for p in self.positions],
            "closed_trades": [asdict(t) for t in self.closed_trades],
        }

    @staticmethod
    def from_dict(d: dict) -> "State":
        s = State(cash=d["cash"], initial_capital=d["initial_capital"])
        s.positions = [Position(**p) for p in d.get("positions", [])]
        s.closed_trades = [ClosedTrade(**t) for t in d.get("closed_trades", [])]
        return s

    @staticmethod
    def load() -> "State":
        p = CFG["state_file"]
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return State.from_dict(json.load(f))
        return State(cash=CFG["initial_capital"], initial_capital=CFG["initial_capital"])

    def save(self) -> None:
        p = CFG["state_file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# EDGAR helpers
# ---------------------------------------------------------------------------

_SEC_HEADERS = {"User-Agent": "ARIA Research dandybee06@gmail.com"}
_RATE = 0.13     # 7.7 req/sec, safely under SEC 10/sec limit


def _http_get_json(url: str) -> dict:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=_SEC_HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {}
        except Exception:
            pass
        time.sleep(_RATE * (attempt + 2))
    return {}


def _load_cik_map() -> dict[str, str]:
    p = CFG["cik_map"]
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _load_universe() -> list[str]:
    """Return SimFin universe tickers (reads from research data)."""
    try:
        import polars as pl
        income_path = _ROOT / "data" / "simfin" / "us-income-quarterly.parquet"
        if income_path.exists():
            df = pl.read_parquet(income_path)
            return df["Ticker"].unique().to_list()
    except Exception:
        pass
    # Fallback: all tickers in CIK map
    cik_map = _load_cik_map()
    return list(cik_map.keys())[:800]  # cap at reasonable size


def _refresh_edgar_cache(cik: str, cache_file: pathlib.Path, today: date) -> list[dict]:
    """
    Fetch/refresh EDGAR submissions for a CIK.
    Cache format: {"refresh_date": "YYYY-MM-DD", "filings": [...]}
    Only fetches "recent" section (last ~1-2 years) — enough for paper trading.
    """
    # Check if already refreshed today
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("refresh_date") == today.isoformat():
                return cached.get("filings", [])
        except Exception:
            pass

    # Fetch from EDGAR (recent section only — faster)
    data = _http_get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    time.sleep(_RATE)
    if not data:
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    reports = recent.get("reportDate", [])

    filings = [
        {"form": f, "filingDate": d, "reportDate": r or ""}
        for f, d, r in zip(forms, dates, reports)
        if f in ("10-Q", "10-K", "8-K")
    ]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"refresh_date": today.isoformat(), "filings": filings}, indent=2),
        encoding="utf-8",
    )
    return filings


def get_fresh_earnings_tickers(
    universe: list[str],
    today: date,
) -> list[tuple[str, date]]:
    """
    Return (ticker, announcement_date) for earnings announced in the last
    lookback_days market days. Uses 8-K filing dates matched to 10-Q quarters.
    """
    cik_map = _load_cik_map()
    cutoff = prev_market_day(today, CFG["lookback_days"])
    results: list[tuple[str, date]] = []
    n_fetched = 0

    for ticker in universe:
        cik = cik_map.get(ticker.upper())
        if not cik:
            continue

        cache_file = CFG["edgar_cache"] / f"{cik}.json"

        # Determine if refresh is needed (not yet refreshed today)
        needs_refresh = True
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached.get("refresh_date") == today.isoformat():
                    needs_refresh = False
                    filings = cached.get("filings", [])
            except Exception:
                pass

        if needs_refresh:
            filings = _refresh_edgar_cache(cik, cache_file, today)
            n_fetched += 1
            if n_fetched % 50 == 1:
                _log(f"[EDGAR] Refreshing submissions... ({n_fetched} fetched so far)")

        # Match each 10-Q/10-K to its earnings 8-K
        quarterly = [
            (s["filingDate"], s["reportDate"])
            for s in filings
            if s.get("form") in ("10-Q", "10-K") and s.get("reportDate")
        ]
        eightk_dates = [s["filingDate"] for s in filings if s.get("form") == "8-K"]

        for _, rd_str in quarterly:
            try:
                rd = date.fromisoformat(rd_str)
            except ValueError:
                continue

            win_start = rd + timedelta(days=CFG["min_8k_after_qtr"])
            win_end   = rd + timedelta(days=CFG["max_8k_after_qtr"])

            def _safe_parse(s: str) -> Optional[date]:
                try:
                    return date.fromisoformat(s)
                except ValueError:
                    return None

            matching = sorted(
                fd for d in eightk_dates
                if (fd := _safe_parse(d)) and win_start <= fd <= win_end
            )
            if matching:
                ann_date = matching[0]
                if ann_date >= cutoff:
                    results.append((ticker, ann_date))
                break  # one announcement per ticker per quarter

    if n_fetched:
        _log(f"[EDGAR] Refreshed {n_fetched} tickers from SEC.")
    return results


# ---------------------------------------------------------------------------
# Price helpers (yfinance)
# ---------------------------------------------------------------------------

def _yf_history(ticker: str, start: date, end: date) -> "pd.DataFrame":
    import pandas as pd
    try:
        t = yf.Ticker(ticker)
        hist = t.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )
        if hist.empty:
            return pd.DataFrame()
        hist.index = pd.to_datetime(hist.index).normalize()
        return hist
    except Exception:
        import pandas as pd
        return pd.DataFrame()


def get_price_reaction(ticker: str, ann_date: date) -> Optional[float]:
    """Return single-day return on ann_date vs the prior close."""
    import pandas as pd
    hist = _yf_history(ticker, ann_date - timedelta(days=7), ann_date)
    if hist.empty:
        return None
    target = pd.Timestamp(ann_date)
    if target not in hist.index:
        return None
    pos = hist.index.get_loc(target)
    if pos == 0:
        return None
    prev = float(hist["Close"].iloc[pos - 1])
    cur  = float(hist["Close"].iloc[pos])
    return (cur - prev) / prev if prev > 0 else None


def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Batch price fetch — last close for each ticker."""
    prices: dict[str, float] = {}
    for tkr in tickers:
        try:
            hist = yf.Ticker(tkr).history(period="2d", auto_adjust=True)
            if not hist.empty:
                prices[tkr] = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return prices


def get_annualized_vol(ticker: str, lookback: int = 20) -> float:
    """Annualized close-to-close vol (20-day default)."""
    try:
        hist = yf.Ticker(ticker).history(period=f"{lookback + 10}d", auto_adjust=True)
        if len(hist) < 5:
            return 0.30
        rets = hist["Close"].pct_change().dropna().tail(lookback)
        return float(rets.std() * np.sqrt(252))
    except Exception:
        return 0.30


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def compute_signals(
    fresh_earnings: list[tuple[str, date]],
) -> tuple[list[dict], list[dict]]:
    """
    Compute cross-sectional PEAD_z, apply gate + top-pct selection.

    Returns:
        entries  — signals passing gate+percentile (ready to enter)
        watchlist — fresh earnings not meeting the gate (informational)
    """
    rows: list[dict] = []
    for ticker, ann_date in fresh_earnings:
        r = get_price_reaction(ticker, ann_date)
        if r is not None:
            rows.append({"ticker": ticker, "ann_date": ann_date, "reaction": r})

    if len(rows) < 3:
        return [], rows  # not enough data for z-scoring

    vals = np.array([r["reaction"] for r in rows])
    mu, sigma = float(np.mean(vals)), float(np.std(vals))
    for r in rows:
        r["pead_z"] = float(np.clip((r["reaction"] - mu) / sigma, -3.0, 3.0)) if sigma > 1e-10 else 0.0

    n = len(rows)
    k = max(1, int(math.ceil(n * CFG["top_pct"])))
    sorted_rows = sorted(rows, key=lambda x: x["pead_z"], reverse=True)

    entries: list[dict] = []
    gate = CFG["min_pead_ret"]

    # Longs: top k with reaction >= gate
    for r in sorted_rows[:k]:
        if r["reaction"] >= gate:
            entries.append({**r, "direction": "long"})

    # Shorts: bottom k with reaction <= -gate (only if long_only=False)
    if not CFG["long_only"]:
        for r in sorted_rows[-k:]:
            if r["reaction"] <= -gate:
                entries.append({**r, "direction": "short"})

    entry_set = {r["ticker"] for r in entries}
    watchlist = [r for r in rows if r["ticker"] not in entry_set and abs(r["reaction"]) < gate]

    return entries, watchlist


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def size_position(ticker: str, price: float, nav: float, cash: float) -> int:
    """
    Inverse-vol sizing targeting CFG['vol_target'] annual portfolio vol.
    Returns integer share count.
    """
    ann_vol = max(get_annualized_vol(ticker), 0.08)   # floor 8% to cap size
    daily_vol = ann_vol / math.sqrt(252)

    # Dollar budget: nav * vol_target = position_size * price * daily_vol
    target_daily_dollar_vol = nav * CFG["vol_target"] / math.sqrt(252)
    pos_dollars = target_daily_dollar_vol / daily_vol if daily_vol > 0 else 0
    pos_dollars = min(pos_dollars, cash * 0.95, nav * 0.20)  # caps: cash and 20% NAV

    shares = int(pos_dollars / price)
    return shares if shares >= 1 and shares * price <= cash else 0


# ---------------------------------------------------------------------------
# P&L and NAV
# ---------------------------------------------------------------------------

def compute_nav(
    state: State,
    prices: dict[str, float],
) -> tuple[float, float, float]:
    """Returns (nav, unrealized_pnl, realized_pnl)."""
    position_value = 0.0
    unrealized = 0.0
    for pos in state.positions:
        price = prices.get(pos.ticker, pos.entry_price)
        position_value += price * pos.shares
        if pos.direction == "long":
            unrealized += (price - pos.entry_price) * pos.shares
        else:
            unrealized += (pos.entry_price - price) * pos.shares
    realized = sum(t.pnl for t in state.closed_trades)
    nav = state.cash + position_value
    return nav, unrealized, realized


# ---------------------------------------------------------------------------
# Discord notifications
# ---------------------------------------------------------------------------

def _send_discord(message: str) -> None:
    webhook = CFG["discord_webhook"]
    if not webhook:
        print("[Discord] No webhook — printing:\n" + message + "\n")
        return
    if len(message) > 1980:
        message = message[:1977] + "..."
    try:
        payload = json.dumps({"content": message}).encode("utf-8")
        req = urllib.request.Request(
            webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as exc:
        _log(f"[Discord] Error: {exc}")
    time.sleep(0.6)   # Discord rate limit


def _fmt_summary(
    state: State, prices: dict[str, float],
    nav: float, nav_prev: float,
    unrealized: float, realized: float,
    today: date,
) -> str:
    today_pct = (nav - nav_prev) / nav_prev * 100 if nav_prev else 0
    lines = [
        f":bar_chart: **{CFG['strategy_name']} | Daily Report | {today.isoformat()}**",
        (f"NAV: ${nav:,.2f} ({today_pct:+.2f}% today)  |  Cash: ${state.cash:,.2f}"
         f"  |  Realized: ${realized:+,.2f}  Unrealized: ${unrealized:+,.2f}"),
        "",
        f"Open positions ({len(state.positions)}):",
    ]
    for pos in state.positions:
        price = prices.get(pos.ticker, pos.entry_price)
        pnl = (price - pos.entry_price) * pos.shares * (1 if pos.direction == "long" else -1)
        pct = pnl / (pos.entry_price * pos.shares) * 100
        held = (today - date.fromisoformat(pos.entry_date)).days
        gem = ":small_green_diamond:" if pnl >= 0 else ":small_orange_diamond:"
        lines.append(
            f"  {gem} {pos.ticker:<6} {pos.shares:.0f}sh @ ${pos.entry_price:.2f}"
            f"  now ${price:.2f}  ({pct:+.2f}% / ${pnl:+.0f})"
            f"  stop ${pos.stop_price:.2f}  held {held}d"
        )
    return "\n".join(lines)


def _fmt_portfolio(
    state: State, prices: dict[str, float],
    nav: float, nav_prev: float,
    unrealized: float, realized: float,
    today: date,
) -> str:
    today_pnl = nav - nav_prev
    today_pct = today_pnl / nav_prev * 100 if nav_prev else 0
    since_pct = (nav - state.initial_capital) / state.initial_capital * 100
    n = len(state.positions)
    free = CFG["max_positions"] - n

    hdr = [
        f":chart_with_upwards_trend: **{CFG['strategy_name']}  |  {_day_label(today)}**",
        "",
        ":moneybag: **PORTFOLIO**",
        "```",
        f"NAV        ${nav:>12,.2f}   {since_pct:+.2f}% since start",
        f"Today P&L  ${today_pnl:>+12,.2f}   {today_pct:+.2f}%",
        f"Unrealized ${unrealized:>+12,.2f}",
        f"Realized   ${realized:>+12,.2f}",
        f"Cash       ${state.cash:>12,.2f}",
        f"Slots       {n}/{CFG['max_positions']} used   {free} free",
        "```",
        "",
        f":open_file_folder: **{n} positions  ({free} slot{'s' if free != 1 else ''} free)**",
        "```",
        f"{'SYM':<6}  {'SHARES':>6}  {'ENTRY':>8}  {'NOW':>8}  {'P&L$':>7}  {'P&L%':>6}  {'STOP':>8}  HELD",
        "-" * 66,
    ]
    rows = []
    net = 0.0
    for pos in state.positions:
        price = prices.get(pos.ticker, pos.entry_price)
        pnl = (price - pos.entry_price) * pos.shares * (1 if pos.direction == "long" else -1)
        pct = pnl / (pos.entry_price * pos.shares) * 100
        held = (today - date.fromisoformat(pos.entry_date)).days
        net += pnl
        rows.append(
            f"{pos.ticker:<6}  {pos.shares:>6.0f}  ${pos.entry_price:>7.2f}"
            f"  ${price:>7.2f}  ${pnl:>+6.0f}  {pct:>+5.1f}%"
            f"  ${pos.stop_price:>7.2f}  {held}d"
        )
    ftr = ["-" * 66, f"{'NET':>56}  ${net:>+7.0f}", "```"]
    return "\n".join(hdr + rows + ftr)


def _fmt_signals(
    entries: list[dict], watchlist: list[dict], today: date,
) -> str:
    lines = [
        f":chart_with_upwards_trend: **{CFG['strategy_name']}  |  {_day_label(today)}  (signals)**",
        "",
    ]
    if entries:
        lines += [
            f":bell: **{len(entries)} NEW ENTRY SIGNAL{'S' if len(entries) > 1 else ''}**",
            "```",
            f"{'DIR':<5}  {'SYM':<6}  {'REACTION':>9}  {'PEAD_Z':>7}  ANN_DATE",
            "-" * 46,
        ]
        for s in entries:
            lines.append(
                f"{s['direction'].upper():<5}  {s['ticker']:<6}  {s['reaction']:>+8.1%}"
                f"  {s['pead_z']:>+6.2f}   {s['ann_date'].isoformat()}"
            )
        lines.append("```")
    else:
        lines.append(
            f":hourglass: **No new entry signals today**"
            f"  (need 8-K reaction >= {CFG['min_pead_ret']:.0%} in top {CFG['top_pct']:.0%})"
        )

    if watchlist:
        lines += [
            "",
            f":eyes: **WATCHLIST**  —  fresh earnings, reaction < {CFG['min_pead_ret']:.0%}",
            "```",
            f"{'SYM':<6}  {'REACTION':>9}  {'ANN_DATE':<12}  STATUS",
            "-" * 46,
        ]
        for w in watchlist[:8]:
            r = w["reaction"]
            status = "small move" if abs(r) < 0.01 else "below gate"
            lines.append(f"{w['ticker']:<6}  {r:>+8.1%}  {w['ann_date'].isoformat():<12}  {status}")
        lines.append("```")

    return "\n".join(lines)


def _fmt_performance(state: State, today: date) -> str:
    trades = state.closed_trades
    n = len(trades)
    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    wr = len(wins) / n if n else 0.0
    gross_win  = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    avg_hold = sum(t.held_days for t in trades) / n if n else 0.0
    realized = sum(t.pnl for t in trades)
    target = 30

    lines = [
        f":chart_with_upwards_trend: **{CFG['strategy_name']}  |  {_day_label(today)}  (performance)**",
        "",
    ]

    recent = sorted(trades, key=lambda t: t.exit_date, reverse=True)[:5]
    if recent:
        lines += [
            ":bar_chart: **LAST 5 CLOSED TRADES**",
            "```",
            f"{'':4}  {'SYM':<6}  {'ENTRY':>8}  {'EXIT':>8}  {'P&L%':>6}  {'P&L$':>8}  {'HELD':>5}  EXIT REASON",
            "-" * 70,
        ]
        for t in recent:
            tag = "[W]" if t.pnl > 0 else "[L]"
            lines.append(
                f"{tag}  {t.ticker:<6}  ${t.entry_price:>7.2f}  ${t.exit_price:>7.2f}"
                f"  {t.pnl_pct*100:>+5.1f}%  ${t.pnl:>+7.0f}  {t.held_days:>4}d  {t.exit_reason}"
            )
        lines.append("```")

    if n < 10:
        assessment = f":grey_question: EARLY SIGNAL: ACCUMULATING DATA  —  {n}/{target} trades"
    elif wr >= 0.60 and pf >= 1.20:
        assessment = f":white_check_mark: ON TARGET — WR and PF meeting goals  ({n}/{target} trades)"
    elif wr >= 0.55 or pf >= 1.20:
        assessment = f":yellow_circle: PARTIAL — one metric below target  ({n}/{target} trades)"
    else:
        assessment = f":red_circle: BELOW TARGET — review signal quality  ({n}/{target} trades)"

    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    lines += [
        "",
        f":trophy: **PERFORMANCE  —  {n} trades closed  (target: {target})**",
        "```",
        f"Record         {len(wins)}W / {len(losses)}L   ({n} total)",
        f"Win rate       {wr:.1%}   (target 60%+)",
        f"Profit factor  {pf_str}   (target >= 1.20)",
        f"Avg hold       {avg_hold:.0f}d",
        f"Realized P&L   ${realized:+,.2f}",
        "```",
        assessment,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NAV history
# ---------------------------------------------------------------------------

def _load_prev_nav() -> float:
    p = CFG["nav_history"]
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data:
                return float(sorted(data.items())[-1][1])
        except Exception:
            pass
    return CFG["initial_capital"]


def _save_nav(nav: float) -> None:
    p = CFG["nav_history"]
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    data[date.today().isoformat()] = round(nav, 2)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        p = CFG["log_file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run() -> None:
    today = date.today()
    _log(f"=== {CFG['strategy_name']} paper trader | {today.isoformat()} ===")

    if not is_market_day(today):
        _log("Not a market day — exiting.")
        return

    # ---- Load state --------------------------------------------------------
    state = State.load()
    nav_prev = _load_prev_nav()

    # ---- Current prices for open positions ---------------------------------
    pos_tickers = [p.ticker for p in state.positions]
    prices = get_current_prices(pos_tickers) if pos_tickers else {}

    # ---- Check exits -------------------------------------------------------
    to_close: list[tuple[Position, str]] = []
    for pos in state.positions:
        price = prices.get(pos.ticker, pos.entry_price)
        held  = (today - date.fromisoformat(pos.entry_date)).days
        if pos.direction == "long":
            if price <= pos.stop_price:
                to_close.append((pos, "hard_stop"))
            elif held >= CFG["hold_days"]:
                to_close.append((pos, "time_stop"))
        else:
            if price >= pos.stop_price:
                to_close.append((pos, "hard_stop"))
            elif held >= CFG["hold_days"]:
                to_close.append((pos, "time_stop"))

    for pos, reason in to_close:
        exit_price = prices.get(pos.ticker, pos.entry_price)
        pnl = (exit_price - pos.entry_price) * pos.shares * (1 if pos.direction == "long" else -1)
        pnl_pct = pnl / (pos.entry_price * pos.shares)
        state.cash += exit_price * pos.shares   # simplified: return full value
        state.closed_trades.append(ClosedTrade(
            ticker=pos.ticker, direction=pos.direction, shares=pos.shares,
            entry_price=pos.entry_price, exit_price=round(exit_price, 4),
            entry_date=pos.entry_date, exit_date=today.isoformat(),
            pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 6),
            held_days=(today - date.fromisoformat(pos.entry_date)).days,
            exit_reason=reason,
        ))
        state.positions = [p for p in state.positions if p is not pos]
        _log(f"CLOSE {pos.direction.upper()} {pos.ticker} [{reason}]: P&L ${pnl:+.2f}")

    # ---- Detect fresh earnings 8-Ks ----------------------------------------
    universe = _load_universe()
    _log(f"Scanning {len(universe)} universe tickers for fresh earnings...")
    fresh_earnings = get_fresh_earnings_tickers(universe, today)
    _log(f"Found {len(fresh_earnings)} fresh earnings announcements")

    # ---- Compute PEAD signals -----------------------------------------------
    entries, watchlist = compute_signals(fresh_earnings)

    # ---- Get prices for signal tickers -------------------------------------
    sig_tickers = [s["ticker"] for s in entries]
    new_prices = get_current_prices(sig_tickers)
    prices.update(new_prices)

    # ---- Recompute NAV after exits -----------------------------------------
    nav, unrealized, realized = compute_nav(state, prices)

    # ---- Open new positions -------------------------------------------------
    n_free = CFG["max_positions"] - len(state.positions)
    for sig in entries[:n_free]:
        tkr = sig["ticker"]
        if any(p.ticker == tkr for p in state.positions):
            continue   # already holding this ticker
        price = prices.get(tkr)
        if not price:
            continue
        shares = size_position(tkr, price, nav, state.cash)
        if shares <= 0:
            continue
        direction = sig["direction"]
        stop = round(price * (1 - CFG["stop_pct"] if direction == "long" else 1 + CFG["stop_pct"]), 2)
        pos = Position(
            ticker=tkr, shares=shares, entry_price=round(price, 4),
            entry_date=today.isoformat(), stop_price=stop, direction=direction,
        )
        state.positions.append(pos)
        state.cash -= price * shares
        _log(f"OPEN {direction.upper()} {tkr}: {shares}sh @ ${price:.2f}, stop ${stop:.2f}")

    # ---- Final NAV ----------------------------------------------------------
    nav, unrealized, realized = compute_nav(state, prices)

    # ---- Send Discord messages ---------------------------------------------
    msgs = [
        _fmt_summary(state, prices, nav, nav_prev, unrealized, realized, today),
        _fmt_portfolio(state, prices, nav, nav_prev, unrealized, realized, today),
        _fmt_signals(entries, watchlist, today),
        _fmt_performance(state, today),
    ]
    for msg in msgs:
        _send_discord(msg)

    # ---- Persist -----------------------------------------------------------
    state.save()
    _save_nav(nav)
    _log(f"Done. NAV=${nav:,.2f} | {len(state.positions)} open | Cash=${state.cash:,.2f}")


if __name__ == "__main__":
    run()
