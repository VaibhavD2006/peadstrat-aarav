import subprocess
import sys
import pytest

def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "aria.cli"] + args,
        capture_output=True, text=True,
        cwd=r"C:\Users\dandy\OneDrive\Documents\aaravstrat\aria"
    )

def test_run_backtest_prints_signals():
    result = _run(["run-backtest", "--signals", "ESQS,RMV", "--hold-days", "10"])
    assert result.returncode == 0
    assert "ESQS" in result.stdout
    assert "hold_days" in result.stdout

def test_event_study_prints_signal():
    result = _run(["event-study", "--signal", "ESQS"])
    assert result.returncode == 0
    assert "ESQS" in result.stdout

def test_missing_subcommand_exits_nonzero():
    result = _run([])
    assert result.returncode != 0
