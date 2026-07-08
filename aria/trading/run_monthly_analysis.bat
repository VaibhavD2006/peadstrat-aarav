@echo off
REM ARIA Monthly Analysis Runner
REM Scheduled via Windows Task Scheduler on the 1st of each month at 9:00 AM.
REM Calls Claude API to analyze ARIA + RSI2 paper trading results.
REM Writes report to aria/trading/monthly_analysis_YYYY-MM.md

REM Change to project root
cd /d "%~dp0.."

REM Activate virtual environment if present
IF EXIST ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Run the monthly analysis
python -m aria.trading.monthly_analysis >> trading\monthly_analysis.log 2>&1

echo Done. Check trading\monthly_analysis.log for details.
