import pytest
from datetime import date
from aria.data.ingestion.earnings import EarningsLoader, EarningsEvent

def test_entry_bmo_is_same_day():
    loader = EarningsLoader()
    ev = EarningsEvent("AAPL", date(2024, 1, 25), "BMO", "Q1FY24")
    assert loader.entry_date(ev) == date(2024, 1, 25)

def test_entry_amc_is_next_business_day():
    loader = EarningsLoader()
    ev = EarningsEvent("AAPL", date(2024, 1, 25), "AMC", "Q1FY24")  # Thursday
    assert loader.entry_date(ev) == date(2024, 1, 26)               # Friday

def test_entry_amc_friday_rolls_to_monday():
    loader = EarningsLoader()
    ev = EarningsEvent("AAPL", date(2024, 1, 26), "AMC", "Q1FY24")  # Friday
    assert loader.entry_date(ev) == date(2024, 1, 29)               # Monday

def test_exit_is_10_business_days():
    loader = EarningsLoader()
    entry = date(2024, 1, 26)   # Friday
    exit_ = loader.exit_date(entry, hold_days=10)
    assert exit_ == date(2024, 2, 8)
