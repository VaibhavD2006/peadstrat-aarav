"""Tests for SUE signal and sue_loader."""
import math
import pytest
import polars as pl
from datetime import date
from pathlib import Path
import tempfile

from aria.signals.sue import SUESignal, compute_sue_raw
from aria.data.ingestion.sue_loader import load_consensus, get_sue_inputs, _normalise_columns


# ---------------------------------------------------------------------------
# compute_sue_raw
# ---------------------------------------------------------------------------

def test_sue_raw_positive_beat():
    """Analyst predicted 1.00, reported 1.20 => positive surprise."""
    val = compute_sue_raw(actual_eps=1.20, consensus_eps=1.00)
    assert val > 0

def test_sue_raw_negative_miss():
    """Analyst predicted 1.00, reported 0.80 => negative surprise."""
    val = compute_sue_raw(actual_eps=0.80, consensus_eps=1.00)
    assert val < 0

def test_sue_raw_uses_forecast_std_when_provided():
    surprise = 0.10
    std = 0.05
    val = compute_sue_raw(actual_eps=1.10, consensus_eps=1.00, forecast_std=std)
    assert abs(val - (surprise / std)) < 1e-9

def test_sue_raw_falls_back_to_abs_consensus_when_std_too_small():
    """std below fallback_scale => use abs(consensus) as denominator."""
    val = compute_sue_raw(actual_eps=1.10, consensus_eps=1.00, forecast_std=0.001)
    assert abs(val - 0.10 / 1.00) < 1e-9

def test_sue_raw_near_zero_consensus():
    """consensus near zero => uses fallback_scale=0.01 denominator."""
    actual, consensus = 0.05, 0.001
    val = compute_sue_raw(actual_eps=actual, consensus_eps=consensus)
    expected = (actual - consensus) / 0.01  # denom = max(abs(0.001), 0.01) = 0.01
    assert abs(val - expected) < 1e-6


# ---------------------------------------------------------------------------
# SUESignal.compute_normalized
# ---------------------------------------------------------------------------

def test_compute_normalized_empty_returns_empty():
    result = SUESignal().compute_normalized([])
    assert result.is_empty()

def test_compute_normalized_z_scores_cross_sectionally():
    rows = [
        {"ticker": "A", "sue_raw": 2.0},
        {"ticker": "B", "sue_raw": 0.0},
        {"ticker": "C", "sue_raw": -2.0},
    ]
    result = SUESignal().compute_normalized(rows)
    z = dict(zip(result["ticker"].to_list(), result["SUE_z"].to_list()))
    assert z["A"] > 0
    assert z["C"] < 0
    assert abs(z["B"]) < 0.01  # mean is 0, B is at mean

def test_compute_normalized_clips_to_three():
    rows = [{"ticker": str(i), "sue_raw": float(i * 10)} for i in range(10)]
    result = SUESignal().compute_normalized(rows)
    assert result["SUE_z"].max() <= 3.0
    assert result["SUE_z"].min() >= -3.0

def test_compute_normalized_handles_nans():
    rows = [
        {"ticker": "A", "sue_raw": float("nan")},
        {"ticker": "B", "sue_raw": 1.0},
        {"ticker": "C", "sue_raw": 0.0},
        {"ticker": "D", "sue_raw": -1.0},
    ]
    result = SUESignal().compute_normalized(rows)
    z = dict(zip(result["ticker"].to_list(), result["SUE_z"].to_list()))
    assert z["A"] == 0.0  # NaN → remains at 0 (not z-scored)
    assert z["B"] > 0
    assert z["D"] < 0

def test_compute_normalized_fewer_than_3_returns_zeros():
    rows = [
        {"ticker": "A", "sue_raw": 1.0},
        {"ticker": "B", "sue_raw": -1.0},
    ]
    result = SUESignal().compute_normalized(rows)
    # < 3 valid → no z-scoring → all 0
    assert all(v == 0.0 for v in result["SUE_z"].to_list())


# ---------------------------------------------------------------------------
# sue_loader: column normalisation
# ---------------------------------------------------------------------------

def test_normalise_columns_renames_aliases():
    df = pl.DataFrame({
        "symbol": ["A"],
        "announcement_date": [date(2024, 1, 15)],
        "epsEstimated": [1.0],
        "epsActual": [1.1],
    })
    result = _normalise_columns(df)
    assert "ticker" in result.columns
    assert "report_date" in result.columns
    assert "consensus_eps" in result.columns
    assert "actual_eps" in result.columns

def test_normalise_columns_leaves_canonical_names_unchanged():
    df = pl.DataFrame({
        "ticker": ["A"],
        "report_date": [date(2024, 1, 15)],
        "consensus_eps": [1.0],
        "actual_eps": [1.1],
    })
    result = _normalise_columns(df)
    assert result.columns == df.columns


# ---------------------------------------------------------------------------
# sue_loader: load_consensus
# ---------------------------------------------------------------------------

def test_load_consensus_returns_none_when_file_absent():
    result = load_consensus(Path("nonexistent_path/no_file.csv"))
    assert result is None

def test_load_consensus_loads_valid_csv(tmp_path):
    csv = tmp_path / "eps.csv"
    csv.write_text(
        "ticker,report_date,fiscal_quarter_end,consensus_eps,actual_eps\n"
        "AAPL,2024-01-30,2023-12-31,2.10,2.18\n"
        "MSFT,2024-01-25,2023-12-31,2.65,2.93\n"
    )
    df = load_consensus(csv)
    assert df is not None
    assert df.shape[0] == 2
    assert "ticker" in df.columns
    assert "report_date" in df.columns

def test_load_consensus_raises_on_missing_required_cols(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("ticker,date\nAAPL,2024-01-30\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_consensus(csv)


# ---------------------------------------------------------------------------
# sue_loader: get_sue_inputs
# ---------------------------------------------------------------------------

def _make_consensus_df():
    return pl.DataFrame({
        "ticker":         ["AAPL", "AAPL", "MSFT", "GOOG"],
        "report_date":    [date(2024, 1, 1), date(2024, 4, 1),
                           date(2024, 1, 5), date(2024, 1, 10)],
        "consensus_eps":  [2.0, 2.1, 3.0, 1.5],
        "actual_eps":     [2.2, 2.3, 2.8, 1.6],
    })

def test_get_sue_inputs_returns_most_recent_per_ticker():
    df = _make_consensus_df()
    rows = get_sue_inputs(["AAPL", "MSFT"], date(2024, 5, 1), df)
    tickers = {r["ticker"] for r in rows}
    assert tickers == {"AAPL", "MSFT"}
    aapl_row = next(r for r in rows if r["ticker"] == "AAPL")
    # AAPL has two rows; April is most recent
    assert aapl_row["sue_raw"] == pytest.approx((2.3 - 2.1) / max(abs(2.1), 0.01))

def test_get_sue_inputs_excludes_future_announcements():
    df = _make_consensus_df()
    # as_of_date before all AAPL announcements
    rows = get_sue_inputs(["AAPL"], date(2023, 12, 31), df)
    assert rows == []

def test_get_sue_inputs_handles_tickers_with_no_data():
    df = _make_consensus_df()
    rows = get_sue_inputs(["NVDA"], date(2024, 6, 1), df)
    assert rows == []

def test_get_sue_inputs_correct_sue_sign():
    df = _make_consensus_df()
    # MSFT: actual (2.8) < consensus (3.0) → miss → negative SUE
    rows = get_sue_inputs(["MSFT"], date(2024, 3, 1), df)
    assert len(rows) == 1
    assert rows[0]["sue_raw"] < 0


# ---------------------------------------------------------------------------
# compute_sue_raw: hist_errors parameter
# ---------------------------------------------------------------------------

def test_sue_raw_uses_hist_errors_when_provided():
    """hist_errors with std computed from list → normalizer = std(hist_errors)."""
    import numpy as np
    hist = [0.10, -0.10, 0.20, -0.20]
    expected_norm = float(np.std(hist, ddof=1))
    val = compute_sue_raw(actual_eps=1.30, consensus_eps=1.00,
                          hist_errors=hist)
    assert abs(val - 0.30 / expected_norm) < 1e-6


def test_sue_raw_hist_errors_preferred_over_forecast_std():
    """hist_errors takes priority over forecast_std."""
    import numpy as np
    hist = [0.10, -0.10, 0.20, -0.20]
    expected_norm = float(np.std(hist, ddof=1))
    val = compute_sue_raw(actual_eps=1.30, consensus_eps=1.00,
                          forecast_std=0.50, hist_errors=hist)
    assert abs(val - 0.30 / expected_norm) < 1e-6


def test_sue_raw_falls_back_when_hist_errors_too_short():
    """Only 1 hist_error → not enough for std → falls back to abs(consensus)."""
    val_with_hist  = compute_sue_raw(actual_eps=1.10, consensus_eps=1.00,
                                     hist_errors=[0.05])
    val_without    = compute_sue_raw(actual_eps=1.10, consensus_eps=1.00)
    assert abs(val_with_hist - val_without) < 1e-9


def test_sue_raw_hist_errors_fallback_scale_prevents_tiny_std():
    """If std of hist_errors is < fallback_scale=0.01, uses fallback_scale."""
    hist = [0.001, -0.001, 0.001, -0.001]  # tiny std
    val = compute_sue_raw(actual_eps=1.10, consensus_eps=1.00,
                          hist_errors=hist, fallback_scale=0.01)
    assert abs(val - 0.10 / 0.01) < 1e-6  # normalizer clamped to 0.01


# ---------------------------------------------------------------------------
# sue_loader: compute_historical_errors
# ---------------------------------------------------------------------------

from aria.data.ingestion.sue_loader import compute_historical_errors

def _make_rich_consensus_df():
    """6 quarters of AAPL history for testing historical errors."""
    return pl.DataFrame({
        "ticker": ["AAPL"] * 6 + ["MSFT"],
        "report_date": [
            date(2022, 7, 1), date(2022, 10, 1),
            date(2023, 1, 1), date(2023, 4, 1),
            date(2023, 7, 1), date(2023, 10, 1),
            date(2023, 10, 15),
        ],
        "consensus_eps": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0],
        "actual_eps":    [1.1, 1.0, 1.3, 1.2, 1.5, 1.4, 2.2],
    })


def test_compute_historical_errors_returns_prior_errors():
    df = _make_rich_consensus_df()
    errors = compute_historical_errors("AAPL", date(2024, 1, 1), df, n_quarters=4)
    # Prior 4 quarters before 2024-01-01: reports on 2023-10-01, 2023-07-01, 2023-04-01, 2023-01-01
    # errors = actual - consensus: 1.4-1.5=-0.1, 1.5-1.4=+0.1, 1.2-1.3=-0.1, 1.3-1.2=+0.1
    assert len(errors) == 4
    assert all(abs(e) == pytest.approx(0.1) for e in errors)


def test_compute_historical_errors_excludes_as_of_date():
    """report_date equal to as_of_date is excluded (point-in-time safe)."""
    df = _make_rich_consensus_df()
    errors = compute_historical_errors("AAPL", date(2023, 10, 1), df, n_quarters=4)
    # as_of_date is 2023-10-01; row on that date is excluded (uses strict <)
    assert len(errors) == 4  # 4 prior quarters: 2022-07-01, 2022-10-01, 2023-01-01, 2023-04-01


def test_compute_historical_errors_returns_empty_for_unknown_ticker():
    df = _make_rich_consensus_df()
    errors = compute_historical_errors("UNKNOWN", date(2024, 1, 1), df)
    assert errors == []


def test_get_sue_inputs_uses_historical_errors_in_normalizer():
    """get_sue_inputs passes hist_errors to compute_sue_raw when >= 2 available."""
    import numpy as np
    df = _make_rich_consensus_df()
    # AAPL has 5 rows total; as_of_date=2024-01-01 → 5 prior rows
    # Most recent report_date for AAPL: 2023-10-01 (actual=1.4, consensus=1.5)
    # hist_errors: last 4 of prior quarters = [0.1, -0.1, 0.1, -0.1] (computed from prior < 2023-10-01)
    rows = get_sue_inputs(["AAPL"], date(2024, 1, 1), df)
    assert len(rows) == 1
    # hist_errors for AAPL at report_date=2023-10-01 using 4 prior quarters (before 2023-10-01):
    # 2022-07-01: 1.1-1.0=+0.1, 2022-10-01: 1.0-1.1=-0.1, 2023-01-01: 1.3-1.2=+0.1, 2023-04-01: 1.2-1.3=-0.1
    hist = [0.1, -0.1, 0.1, -0.1]
    expected_norm = max(float(np.std(hist, ddof=1)), 0.01)
    expected_sue = (1.4 - 1.5) / expected_norm
    assert abs(rows[0]["sue_raw"] - expected_sue) < 0.01


# ---------------------------------------------------------------------------
# sue_loader: compute_revision_dir
# ---------------------------------------------------------------------------

from aria.data.ingestion.sue_loader import compute_revision_dir

def _make_revision_df():
    """4 quarters of AAPL: consensus rising YoY."""
    return pl.DataFrame({
        "ticker": ["AAPL"] * 4,
        "report_date": [
            date(2022, 10, 1),   # Q3 2022
            date(2023, 1, 15),   # Q4 2022
            date(2023, 10, 1),   # Q3 2023 (~365 days after Q3 2022)
            date(2024, 1, 15),   # Q4 2023
        ],
        "consensus_eps": [1.00, 1.20, 1.30, 1.50],
        "actual_eps":    [1.10, 1.25, 1.35, 1.55],
    })


def test_revision_dir_positive_when_consensus_raised():
    """Q3 2023 consensus (1.30) > Q3 2022 consensus (1.00) → positive."""
    df = _make_revision_df()
    val = compute_revision_dir("AAPL", date(2023, 10, 1), df)
    assert val > 0, f"Expected positive revision, got {val}"
    assert val == pytest.approx((1.30 - 1.00) / 1.00, abs=0.01)


def test_revision_dir_negative_when_consensus_cut():
    """Prior year consensus was higher → negative revision."""
    df = pl.DataFrame({
        "ticker": ["AAPL"] * 2,
        "report_date": [date(2022, 10, 1), date(2023, 10, 1)],
        "consensus_eps": [1.50, 1.20],   # cut YoY
        "actual_eps":    [1.55, 1.25],
    })
    val = compute_revision_dir("AAPL", date(2023, 10, 1), df)
    assert val < 0


def test_revision_dir_returns_zero_for_missing_prior():
    """No prior-year row → returns 0.0."""
    df = _make_revision_df()
    val = compute_revision_dir("AAPL", date(2022, 10, 1), df)
    assert val == 0.0


def test_revision_dir_clipped_to_minus_one_plus_one():
    """Very large change is clipped to [-1, 1]."""
    df = pl.DataFrame({
        "ticker": ["AAPL"] * 2,
        "report_date": [date(2022, 10, 1), date(2023, 10, 1)],
        "consensus_eps": [0.10, 5.00],  # 50x increase → raw=49 → clipped to 1.0
        "actual_eps":    [0.10, 5.00],
    })
    val = compute_revision_dir("AAPL", date(2023, 10, 1), df)
    assert val == pytest.approx(1.0)


def test_revision_dir_returns_zero_for_unknown_ticker():
    df = _make_revision_df()
    val = compute_revision_dir("UNKNOWN", date(2023, 10, 1), df)
    assert val == 0.0
