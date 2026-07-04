from pathlib import Path
import polars as pl


class DataLake:
    def __init__(self, base_dir: str = "data/parquet"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = self.base / f"{key}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write(self, key: str, df: pl.DataFrame) -> None:
        df.write_parquet(self._path(key))

    def read(self, key: str) -> pl.DataFrame:
        return pl.read_parquet(self._path(key))

    def exists(self, key: str) -> bool:
        return self._path(key).exists()
