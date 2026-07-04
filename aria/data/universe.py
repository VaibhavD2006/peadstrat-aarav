import polars as pl

class UniverseFilter:
    def __init__(self, min_adv_usd: float = 5_000_000, min_price: float = 5.0):
        self.min_adv_usd = min_adv_usd
        self.min_price = min_price

    def apply(self, prices: pl.DataFrame) -> pl.DataFrame:
        return prices.filter(
            (pl.col("adv_20d_usd") >= self.min_adv_usd) &
            (pl.col("close") >= self.min_price)
        )

    def compute_adv(self, prices: pl.DataFrame, window: int = 20) -> pl.DataFrame:
        return prices.sort(["ticker", "date"]).with_columns(
            (pl.col("close") * pl.col("volume"))
            .rolling_mean(window_size=window)
            .over("ticker")
            .alias("adv_20d_usd")
        )
