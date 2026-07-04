import numpy as np

class TransactionCostModel:
    def __init__(self,
                 spread_bps_large: float = 5.0,
                 spread_bps_mid: float = 10.0,
                 market_impact_mult: float = 10.0,
                 participation_cap: float = 0.10,
                 borrow_cost_annual_bps: float = 30.0):
        self.spread_large = spread_bps_large
        self.spread_mid   = spread_bps_mid
        self.impact_mult  = market_impact_mult
        self.part_cap     = participation_cap
        self.borrow_annual = borrow_cost_annual_bps

    def spread_cost_bps(self, is_large_cap: bool) -> float:
        return self.spread_large if is_large_cap else self.spread_mid

    def market_impact_bps(self, order_usd: float, adv_20d_usd: float) -> float:
        if adv_20d_usd <= 0:
            return self.impact_mult * 10.0  # fallback for zero ADV
        participation = order_usd / (adv_20d_usd * self.part_cap)
        return self.impact_mult * np.sqrt(max(participation, 0.0))

    def total_cost_bps(self, order_usd: float, adv_20d_usd: float,
                       is_large_cap: bool = True) -> float:
        return self.spread_cost_bps(is_large_cap) + self.market_impact_bps(order_usd, adv_20d_usd)

    def daily_borrow_cost_bps(self) -> float:
        return self.borrow_annual / 252.0
