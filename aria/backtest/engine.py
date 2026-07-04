from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import polars as pl
import numpy as np
from aria.backtest.costs import TransactionCostModel

@dataclass
class BacktestConfig:
    hold_days: int = 10
    initial_capital: float = 100_000_000
    cost_model: TransactionCostModel = field(default_factory=TransactionCostModel)
    max_gap_pct: float = 0.03

@dataclass
class Position:
    ticker: str
    entry_date: date
    exit_date: date
    entry_price: float
    side: str
    weight: float
    capital: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def _get_price(self, prices: pl.DataFrame, ticker: str, d: date,
                   price_col: str = "open") -> Optional[float]:
        row = prices.filter((pl.col("ticker") == ticker) & (pl.col("date") == d))
        if row.is_empty():
            return None
        return float(row[price_col][0])

    def _get_adv(self, prices: pl.DataFrame, ticker: str, d: date) -> float:
        rows = prices.filter((pl.col("ticker") == ticker) & (pl.col("date") <= d)).tail(20)
        if "adv_20d_usd" in rows.columns and not rows.is_empty():
            val = rows["adv_20d_usd"][-1]
            if val is not None:
                return float(val)
        return 5_000_000_000.0

    def run(self, signals: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
        records = []
        cost = self.config.cost_model

        for row in signals.iter_rows(named=True):
            ticker     = row["ticker"]
            entry_date = row["entry_date"]
            side       = row["side"]
            weight     = row["weight"]

            entry_price = self._get_price(prices, ticker, entry_date, "open")
            if entry_price is None:
                continue

            future = (prices
                      .filter((pl.col("ticker") == ticker) & (pl.col("date") > entry_date))
                      .sort("date"))
            if future.shape[0] < self.config.hold_days:
                continue

            exit_date  = future["date"][self.config.hold_days - 1]
            exit_price = float(future["close"][self.config.hold_days - 1])

            capital    = self.config.initial_capital * weight
            adv        = self._get_adv(prices, ticker, entry_date)
            cost_entry = capital * cost.total_cost_bps(capital, adv, True) / 10_000
            cost_exit  = capital * cost.total_cost_bps(capital, adv, True) / 10_000
            borrow     = (capital * cost.daily_borrow_cost_bps() / 10_000 *
                          self.config.hold_days) if side == "short" else 0.0

            direction    = 1.0 if side == "long" else -1.0
            gross_return = (exit_price - entry_price) / entry_price
            pnl = capital * direction * gross_return - cost_entry - cost_exit - borrow

            records.append({
                "ticker":       ticker,
                "entry_date":   str(entry_date),
                "exit_date":    str(exit_date),
                "side":         side,
                "entry_price":  entry_price,
                "exit_price":   exit_price,
                "gross_return": float(gross_return),
                "pnl":          float(pnl),
                "capital":      capital,
                "weight":       weight,
            })

        if not records:
            return pl.DataFrame({
                "ticker": [], "entry_date": [], "exit_date": [], "side": [],
                "entry_price": [], "exit_price": [], "gross_return": [], "pnl": [],
                "capital": [], "weight": [],
            })
        return pl.DataFrame(records)
