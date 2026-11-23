"""SQLite 覆盖存储。

- 重交航次: 单事务内先删该航次全部旧分片, 再插入新分片;
  事务中途任何失败均回滚, 旧覆盖继续可查。
- 查询: 单条只读事务 (单一已提交快照), 只取指定日 + 指定格。
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator

from .geometry import Shard, merge_intervals

SCHEMA = """
CREATE TABLE IF NOT EXISTS coverage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    voyage_id   TEXT    NOT NULL,
    day         TEXT    NOT NULL,
    lat_cell    INTEGER NOT NULL,
    lon_cell    INTEGER NOT NULL,
    start_epoch REAL    NOT NULL,
    end_epoch   REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_coverage_query
    ON coverage (day, lat_cell, lon_cell);
CREATE INDEX IF NOT EXISTS idx_coverage_voyage
    ON coverage (voyage_id);
"""


def _db_path() -> str:
    return os.environ.get("COVERAGE_DB", "/data/coverage.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _transaction(readonly: bool = False) -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        if readonly:
            conn.execute("BEGIN")
        else:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    os.makedirs(os.path.dirname(_db_path()) or ".", exist_ok=True)
    conn = _connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def replace_voyage(voyage_id: str, shards: list[Shard]) -> int:
    """在一个事务内替换航次的全部覆盖, 返回写入的分片数。"""
    rows = [
        (voyage_id, s.day, s.lat_cell, s.lon_cell, s.start_epoch, s.end_epoch)
        for s in shards
    ]
    with _transaction() as conn:
        conn.execute("DELETE FROM coverage WHERE voyage_id = ?", (voyage_id,))
        conn.executemany(
            "INSERT INTO coverage "
            "(voyage_id, day, lat_cell, lon_cell, start_epoch, end_epoch) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def query_coverage(day: str, lat_cell: int, lon_cell: int) -> list[tuple[float, float]]:
    """单一已提交快照内查询某日某格的覆盖, 返回合并后的时段。"""
    with _transaction(readonly=True) as conn:
        result = conn.execute(
            "SELECT start_epoch, end_epoch FROM coverage "
            "WHERE day = ? AND lat_cell = ? AND lon_cell = ? "
            "ORDER BY start_epoch",
            (day, lat_cell, lon_cell),
        ).fetchall()
    return merge_intervals([(r["start_epoch"], r["end_epoch"]) for r in result])


def list_voyages() -> list[dict]:
    with _transaction(readonly=True) as conn:
        rows = conn.execute(
            "SELECT voyage_id, day, lat_cell, lon_cell, COUNT(*) AS shards, "
            "MIN(start_epoch) AS min_ts, MAX(end_epoch) AS max_ts "
            "FROM coverage GROUP BY voyage_id, day, lat_cell, lon_cell "
            "ORDER BY voyage_id, day, lat_cell, lon_cell"
        ).fetchall()
    return [dict(r) for r in rows]
