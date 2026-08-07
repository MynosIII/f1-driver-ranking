from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS race_decompositions AS SELECT * FROM read_parquet(?) LIMIT 0;
"""


class AnalyticalStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.parquet = root / "race_decompositions.parquet"
        self.database = root / "historical_xw.duckdb"

    def write_decompositions(self, rows: list[dict[str, Any]]) -> Path:
        frame = pd.DataFrame(rows)
        frame.to_parquet(self.parquet, index=False)
        parquet_sql_path = str(self.parquet.resolve()).replace("'", "''")
        with duckdb.connect(str(self.database)) as connection:
            connection.execute(
                f"CREATE OR REPLACE VIEW race_decompositions AS "
                f"SELECT * FROM read_parquet('{parquet_sql_path}')"
            )
        return self.parquet

    def query(self, sql: str) -> pd.DataFrame:
        with duckdb.connect(str(self.database), read_only=True) as connection:
            return connection.execute(sql).df()
