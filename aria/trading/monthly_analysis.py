#!/usr/bin/env python3
"""
Monthly ARIA + RSI2 Trade Analysis via Claude API
==================================================
Runs on the 1st of each month (scheduled via Windows Task Scheduler).

Reads state from:
  - aria/trading/state.json         (ARIA PEAD paper trader)
  - aria/trading/nav_history.json   (ARIA NAV history)
  - options2/options-trading/research/paper_state.json  (RSI2 options paper trader)

Calls Claude API to produce a full analysis and trading plan.
Writes output to: aria/trading/monthly_analysis_YYYY-MM.md
Posts summary to ARIA Discord (pead channel).

Usage:
    python aria/trading/monthly_analysis.py           # run analysis
    python aria/trading/monthly_analysis.py --dry-run # print prompt only, no API call
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.request
from datetime import datetime

try:
    from dotenv import load_dotenv as _load_dotenv
    _THIS = pathlib.Path(__file__).parent.parent
    _load_dotenv(_THIS / ".env")
except ImportError:
    pass

# ── paths ─────────────────────────────────────────────────────────────────────
_TRADING_DIR = pathlib.Path(__file__).parent
_ARIA_STATE  = _TRADING_DIR / "state.json"
_ARIA_NAV    = _TRADING_DIR / "nav_history.json"
_RSI2_STATE  = pathlib.Path(__file__).parents[3] / "options2" / "options-trading" / "research" / "paper_state.json"


def _load_json(path: pathlib.Path) -> dict | list | None:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return None


def _send_discord(message: str, webhook_url: str):
    if not webhook_url:
        return
    try:
        payload = json.dumps({"content": message}).encode()
        req = urllib.request.Request(
            webhook_url, data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "DiscordBot (aria-monthly-analysis, 1.0)",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception as exc:
        print(f"  [discord] failed: {exc}")


def _build_prompt(aria: dict | None, nav_history: list | None, rsi2: dict | None, month_label: str) -> str:
    sections: list[str] = []

    sections.append(f"# Monthly Trading Review — {month_label}\n")
    sections.append(
        "You are a quantitative trading analyst reviewing paper trading results for two live strategies. "
        "Write a comprehensive monthly analysis covering: performance attribution, what worked and what did not, "
        "risk observations, and a concrete action plan for the coming month. "
        "Be specific and data-driven. Do not hedge with excessive caveats.\n"
    )

    # ── ARIA section ──────────────────────────────────────────────────────────
    if aria:
        trades = aria.get("trades", [])
        positions = aria.get("positions", {})
        nav = aria.get("nav", 50000)
        cash = aria.get("cash", nav)
        started = aria.get("started", "unknown")
        total_pnl = aria.get("total_realized_pnl_dollar", 0.0)
        wins = aria.get("total_wins", 0)
        losses = aria.get("total_losses", 0)

        recent_trades = trades[-30:] if len(trades) > 30 else trades
        wr = wins / (wins + losses) if (wins + losses) > 0 else 0

        sections.append("## Strategy 1: ARIA E49 (PEAD Post-Earnings Drift)\n")
        sections.append(f"- Started: {started}")
        sections.append(f"- NAV: ${nav:,.2f}  Cash: ${cash:,.2f}")
        sections.append(f"- Total realized P&L: ${total_pnl:+,.2f}")
        sections.append(f"- Win/loss record: {wins}W / {losses}L  (WR={wr:.1%})")
        sections.append(f"- Total closed trades: {len(trades)}  |  Open positions: {len(positions)}\n")

        if recent_trades:
            sections.append("### Last 30 closed trades (ARIA):")
            sections.append("| Symbol | Entry | Exit | P&L% | P&L$ | Reason | Hold |")
            sections.append("|--------|-------|------|------|------|--------|------|")
            for t in recent_trades:
                sections.append(
                    f"| {t.get('symbol','')} | {t.get('entry_date','')} | {t.get('exit_date','')} "
                    f"| {t.get('pnl_pct', 0)*100:+.2f}% | ${t.get('pnl_dollar', 0):+,.0f} "
                    f"| {t.get('exit_reason','')} | {t.get('hold_days', t.get('bars_held', ''))}d |"
                )
            sections.append("")

        if positions:
            sections.append("### Current open ARIA positions:")
            for sym, pos in positions.items():
                sections.append(
                    f"  - {sym}: entered {pos.get('entry_date','')}  "
                    f"direction={pos.get('direction','?')}  "
                    f"unrealized_pnl={pos.get('unrealized_pnl_pct', 0)*100:+.2f}%"
                )
            sections.append("")

        if nav_history:
            recent_nav = nav_history[-30:] if len(nav_history) > 30 else nav_history
            sections.append("### NAV history (last 30 entries):")
            for entry in recent_nav:
                sections.append(f"  - {entry.get('date','')}: ${entry.get('nav', 0):,.2f}")
            sections.append("")
    else:
        sections.append("## Strategy 1: ARIA E49 — no state file found\n")

    # ── RSI2 section ──────────────────────────────────────────────────────────
    if rsi2:
        trades2 = rsi2.get("trades", [])
        positions2 = rsi2.get("positions", {})
        nav2 = rsi2.get("nav", 50000)
        cash2 = rsi2.get("cash", nav2)
        started2 = rsi2.get("started", "unknown")
        total_pnl2 = rsi2.get("total_realized_pnl_dollar", 0.0)
        wins2 = rsi2.get("total_wins", 0)
        losses2 = rsi2.get("total_losses", 0)
        wr2 = wins2 / (wins2 + losses2) if (wins2 + losses2) > 0 else 0

        recent2 = trades2[-30:] if len(trades2) > 30 else trades2

        sections.append("## Strategy 2: RSI2 Bull Put Spreads (Options)\n")
        sections.append(f"- Started: {started2}")
        sections.append(f"- NAV: ${nav2:,.2f}  Cash: ${cash2:,.2f}")
        sections.append(f"- Total realized P&L: ${total_pnl2:+,.2f}")
        sections.append(f"- Win/loss record: {wins2}W / {losses2}L  (WR={wr2:.1%})")
        sections.append(f"- Total closed trades: {len(trades2)}  |  Open positions: {len(positions2)}\n")

        if recent2:
            sections.append("### Last 30 closed trades (RSI2):")
            sections.append("| Symbol | Entry | Exit | P&L% | P&L$ | Reason | Hold |")
            sections.append("|--------|-------|------|------|------|--------|------|")
            for t in recent2:
                sections.append(
                    f"| {t.get('symbol','')} | {t.get('entry_date','')} | {t.get('exit_date','')} "
                    f"| {t.get('pnl_pct', 0)*100:+.2f}% | ${t.get('pnl_dollar', 0):+,.0f} "
                    f"| {t.get('exit_reason','')} | {t.get('bars_held', '')}d |"
                )
            sections.append("")

        if positions2:
            sections.append("### Current open RSI2 positions:")
            for sym, pos in positions2.items():
                struct = pos.get("structure", "?")
                sections.append(
                    f"  - {sym}: [{struct}]  entered {pos.get('entry_date','')}  "
                    f"expiry={pos.get('expiry','?')}  "
                    f"unrealized={pos.get('unrealized_pnl_pct', 0)*100:+.2f}%"
                )
            sections.append("")
    else:
        sections.append("## Strategy 2: RSI2 — no state file found\n")

    sections.append(
        "## Your Task\n\n"
        "1. **Performance Attribution** — Which trades drove returns? Any patterns in losers?\n"
        "2. **Risk Analysis** — Max drawdown, loss streaks, concentration, position sizing issues?\n"
        "3. **Signal Quality** — Are the entry/exit signals performing as expected?\n"
        "4. **Cross-Strategy Observations** — Correlation, diversification, capital allocation?\n"
        "5. **Action Plan for Next Month** — Specific, concrete steps (e.g., 'reduce max positions "
        "from 10 to 8 until win rate recovers', 'tighten RSI entry from 10 to 8 if median RSI stays elevated').\n"
        "6. **Risk Alerts** — Flag any concerning patterns that need immediate attention.\n\n"
        "Format the output as a clean markdown document suitable for a trading journal."
    )

    return "\n".join(sections)


def call_claude(prompt: str, api_key: str, model: str = "claude-sonnet-4-6") -> str:
    payload = json.dumps({
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print prompt only, no API call")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model to use")
    args = parser.parse_args()

    now = datetime.now()
    month_label = now.strftime("%B %Y")
    month_slug  = now.strftime("%Y-%m")
    out_file    = _TRADING_DIR / f"monthly_analysis_{month_slug}.md"

    api_key      = os.environ.get("ANTHROPIC_API_KEY", "")
    webhook_url  = os.environ.get("DISCORD_WEBHOOK_URL", "")

    print(f"\n{'='*60}")
    print(f"  ARIA Monthly Analysis — {month_label}")
    print(f"{'='*60}")

    aria     = _load_json(_ARIA_STATE)
    nav_hist = _load_json(_ARIA_NAV)
    rsi2     = _load_json(_RSI2_STATE)

    if isinstance(nav_hist, dict):
        nav_hist = nav_hist.get("history", list(nav_hist.values()))

    print(f"  ARIA state:   {'found' if aria else 'NOT FOUND'} ({_ARIA_STATE})")
    print(f"  NAV history:  {'found' if nav_hist else 'NOT FOUND'} ({_ARIA_NAV})")
    print(f"  RSI2 state:   {'found' if rsi2 else 'NOT FOUND'} ({_RSI2_STATE})")

    if not aria and not rsi2:
        print("\n  No trade data found. Exiting.")
        sys.exit(1)

    prompt = _build_prompt(aria, nav_hist, rsi2, month_label)

    if args.dry_run:
        print("\n--- PROMPT (dry run) ---")
        print(prompt)
        return

    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY not set. Add it to aria/.env or system env vars.")
        sys.exit(1)

    print(f"\n  Calling Claude ({args.model}) ...")
    try:
        analysis = call_claude(prompt, api_key, args.model)
    except Exception as exc:
        print(f"  ERROR calling Claude API: {exc}")
        sys.exit(1)

    header = f"# Monthly Trading Analysis — {month_label}\n\n*Generated {now.strftime('%Y-%m-%d %H:%M ET')} by Claude ({args.model})*\n\n---\n\n"
    out_file.write_text(header + analysis, encoding="utf-8")
    print(f"\n  Analysis written to: {out_file}")

    # Post Discord summary (first 1800 chars to stay under 2000 limit)
    if webhook_url:
        summary_lines = analysis.split("\n")
        summary_block = ""
        for line in summary_lines:
            if len(summary_block) + len(line) > 1600:
                break
            summary_block += line + "\n"
        discord_msg = (
            f":bar_chart: **Monthly Analysis — {month_label}**\n"
            f"Full report saved to `trading/monthly_analysis_{month_slug}.md`\n\n"
            + summary_block
            + f"\n*... (see full report file for complete analysis)*"
        )
        _send_discord(discord_msg, webhook_url)
        print("  Discord summary sent.")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
