from __future__ import annotations

"""Persistence layer for Atlan Brain Kernel memory chains.

The goal is to let long-running agents preserve their experiential data and to
allow post-hoc analysis without modifying the core cognitive algorithms.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import sqlite3
from typing import Iterable, List, Tuple

MemoryRecord = Tuple[int, str | None, str, float]


class MemoryStore(ABC):
    """Abstract base class representing a persistence backend."""

    @abstractmethod
    def append(self, record: MemoryRecord) -> None:  # noqa: D401, N802
        """Append a single memory *record*."""

    @abstractmethod
    def iterate(self) -> Iterable[MemoryRecord]:  # noqa: D401
        """Yield all stored records in insertion order."""

    @abstractmethod
    def flush(self) -> None:  # noqa: D401
        """Force pending writes to be persisted (if applicable)."""


class InMemoryStore(MemoryStore):
    """Simple list-backed store—keeps everything in RAM."""

    def __init__(self) -> None:
        self._data: List[MemoryRecord] = []

    # ------------------------------------------------------------------
    # API implementations
    # ------------------------------------------------------------------

    def append(self, record: MemoryRecord) -> None:  # noqa: D401
        self._data.append(record)

    def iterate(self) -> Iterable[MemoryRecord]:  # noqa: D401
        yield from self._data

    def flush(self) -> None:  # noqa: D401
        # Nothing to flush for in-memory store
        return None

    # Convenience property
    @property
    def data(self) -> List[MemoryRecord]:  # noqa: D401
        return self._data


class SQLiteStore(MemoryStore):
    """SQLite-based durable store (thread-safe for our simple use-case)."""

    def __init__(self, db_path: str | Path = "memory_chain.sqlite") -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # MemoryStore API
    # ------------------------------------------------------------------

    def append(self, record: MemoryRecord) -> None:  # noqa: D401
        with self.conn:
            self.conn.execute(
                "INSERT INTO memory_chain (tick, source_symbol, target_symbol, energy) VALUES (?, ?, ?, ?)",
                record,
            )

    def iterate(self) -> Iterable[MemoryRecord]:  # noqa: D401
        cur = self.conn.execute(
            "SELECT tick, source_symbol, target_symbol, energy FROM memory_chain ORDER BY id ASC"
        )
        yield from cur.fetchall()

    def flush(self) -> None:  # noqa: D401
        self.conn.commit()

    # ------------------------------------------------------------------
    # Resource handling
    # ------------------------------------------------------------------

    def close(self) -> None:  # noqa: D401
        """Explicitly close the DB connection."""
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:  # noqa: D401
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:  # noqa: D401
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tick INTEGER NOT NULL,
                    source_symbol TEXT,
                    target_symbol TEXT NOT NULL,
                    energy REAL NOT NULL
                )
                """
            ) 