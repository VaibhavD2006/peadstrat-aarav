@echo off
REM ARIA E49 Paper Trader — Daily runner
REM Schedule with Windows Task Scheduler at 4:15 PM ET on weekdays.
REM Runs after market close so yfinance returns today's actual closing prices,
REM matching the backtest's close-to-close entry/exit assumptions exactly.
REM
REM Task Scheduler setup:
REM   Action:    Start a program
REM   Program:   C:\path\to\this\run_daily.bat
REM   Trigger:   Daily at 4:15 PM, Mon-Fri only
REM
REM Set your Discord webhook URL below (or set it as a system env var):
REM Set webhook here only if not already in system env vars (use setx to set permanently)
IF "%DISCORD_WEBHOOK_URL%"=="" SET DISCORD_WEBHOOK_URL=YOUR_WEBHOOK_URL_HERE

REM Optional: override starting capital (default 50000)
REM SET ARIA_CAPITAL=50000

REM Change to project root
cd /d "%~dp0.."

REM Activate virtual environment if present
IF EXIST ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Run the paper trader
python -m aria.trading.paper_trader >> trading\paper_trader.log 2>&1

echo Done. Check trading\paper_trader.log for details.
