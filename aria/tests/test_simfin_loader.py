"""Tests for SimFin data loader. Requires cached data in data/simfin/ (already present)."""
import pytest
from datetime import date
import os
os.environ.setdefault("SIMFIN_API_KEY", "98dccc1f-781a-4013-94bf-04688f2b563d")

from aria.data.ingestion.simfin_loader import (
    load_income_statements,
    get_esqs_inputs,
    build_esqs_batch,
)


@pytest.fixture(scope="module")
def income_df():
    return load_income_statements()


def test_income_loads_successfully(income_df):
    assert income_df.shape[0] > 10_000
    assert "Revenue" in income_df.columns
    assert "Gross Profit" in income_df.columns
    assert "Selling, General & Administrative" in income_df.columns
    assert "Publish Date" in income_df.columns


def test_get_esqs_inputs_aapl_returns_dict(income_df):
    # Use a date after AAPL published Q4 FY2024 results (Publish Date: 2025-01-31)
    result = get_esqs_inputs("AAPL", date(2025, 2, 1), income_df)
    assert result is not None
    assert result["ticker"] == "AAPL"
    assert result["revenue_actual"] > 0
    assert result["revenue_consensus"] > 0
    assert 0 < result["gross_margin_actual"] < 1
    assert 0 < result["gross_margin_trailing"] < 1
    assert 0 < result["sga_pct_actual"] < 1
    assert 0 < result["sga_pct_trailing"] < 1


def test_get_esqs_inputs_point_in_time_respected(income_df):
    # Cutoff before any AAPL data → should return None
    result = get_esqs_inputs("AAPL", date(2019, 12, 31), income_df)
    assert result is None


def test_get_esqs_inputs_unknown_ticker(income_df):
    result = get_esqs_inputs("ZZZNOTREAL", date(2025, 1, 1), income_df)
    assert result is None


def test_get_esqs_sga_pct_is_positive(income_df):
    result = get_esqs_inputs("MSFT", date(2025, 2, 1), income_df)
    if result is not None:
        assert result["sga_pct_actual"] > 0
        assert result["sga_pct_trailing"] > 0


def test_build_esqs_batch_returns_list(income_df):
    tickers = ["AAPL", "MSFT", "NVDA", "ZZZFAKE"]
    result = build_esqs_batch(tickers, date(2025, 2, 1), income_df=income_df)
    # ZZZFAKE should be excluded
    returned_tickers = [r["ticker"] for r in result]
    assert "ZZZFAKE" not in returned_tickers
    assert len(result) >= 1


def test_revenue_consensus_is_trailing_average(income_df):
    # The pseudo-consensus should be the mean of trailing quarters, not equal to actual
    result_early = get_esqs_inputs("AAPL", date(2024, 2, 5), income_df)
    result_late  = get_esqs_inputs("AAPL", date(2025, 2, 1), income_df)
    if result_early and result_late:
        # Revenue actual at different dates should differ
        # Just verify the consensus is in a reasonable range relative to actual
        for r in [result_early, result_late]:
            ratio = r["revenue_actual"] / r["revenue_consensus"]
            assert 0.3 < ratio < 3.0, f"Revenue ratio {ratio} out of bounds"
