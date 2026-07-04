from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
import pandas as pd

@dataclass
class EarningsEvent:
    ticker: str
    announce_date: date
    announce_time: str          # "BMO" = before market open, "AMC" = after market close
    fiscal_quarter: str
    revenue_actual: Optional[float] = None
    revenue_consensus: Optional[float] = None
    eps_actual: Optional[float] = None
    eps_consensus: Optional[float] = None

class EarningsLoader:
    def entry_date(self, event: EarningsEvent) -> date:
        if event.announce_time.upper() in ("BMO", "BTO", "BEFORE"):
            return event.announce_date
        return self._next_business_day(event.announce_date)

    def exit_date(self, entry: date, hold_days: int = 10) -> date:
        bd = pd.bdate_range(start=entry, periods=hold_days)
        return bd[-1].date()

    def _next_business_day(self, d: date) -> date:
        next_day = d + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day
