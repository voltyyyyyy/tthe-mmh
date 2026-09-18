"""Process-safe SQLite inference-budget ledger.

The ledger enforces a hard finite denominator at call boundaries.  Callers
reserve a conservative maximum before launching a call and commit actual usage
afterwards; inflight reservations count against both the total and the memory
allocation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


class BudgetExceeded(RuntimeError):
    """The requested reservation would exceed a hard cap."""


class BudgetViolation(RuntimeError):
    """A caller committed more than its reservation or used an invalid state."""


class Reservation:
    def __init__(self, ledger: "BudgetLedger", reservation_id: str, amount: float) -> None:
        self.ledger = ledger
        self.reservation_id = reservation_id
        self.reserved_amount = float(amount)
        self._closed = False

    def commit(self, actual_amount: float | None = None, *, metadata: Mapping[str, Any] | None = None) -> None:
        if self._closed:
            raise BudgetViolation(f"reservation already closed: {self.reservation_id}")
        self.ledger.commit(self.reservation_id, actual_amount, metadata=metadata)
        self._closed = True

    def release(self, reason: str = "not_run") -> None:
        if not self._closed:
            self.ledger.release(self.reservation_id, reason=reason)
            self._closed = True

    def __enter__(self) -> "Reservation":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._closed:
            self.release("call_failed" if exc_type else "completed_without_commit")
        return False


class BudgetLedger:
    """SQLite-backed ledger with total and per-category hard caps."""

    MEMORY_CATEGORIES = frozenset({
        "memory_validation", "applicability", "embedding", "memory_retrieval",
    })

    def __init__(
        self,
        path: str | Path,
        total_budget: float,
        *,
        memory_fraction: float = 0.10,
        category_caps: Mapping[str, float] | None = None,
        memory_categories: frozenset[str] | None = None,
    ) -> None:
        self.path = str(path)
        total = float(total_budget)
        if total <= 0:
            raise ValueError("total_budget must be finite and positive")
        self.total_budget = total
        self.memory_fraction = float(memory_fraction)
        if not 0.0 <= self.memory_fraction <= 1.0:
            raise ValueError("memory_fraction must be in [0, 1]")
        self.category_caps = {str(key): float(value) for key, value in (category_caps or {}).items()}
        self.memory_categories = frozenset(memory_categories or self.MEMORY_CATEGORIES)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self._create_schema()
        self._ensure_metadata()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                yield
            except BaseException:
                self.connection.rollback()
                raise
            else:
                self.connection.commit()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS budget_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS budget_entries (
                    reservation_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL,
                    call_id TEXT,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    closed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_budget_status_category
                    ON budget_entries(status, category);
                """
            )

    def _ensure_metadata(self) -> None:
        with self._transaction():
            row = self.connection.execute(
                "SELECT value FROM budget_meta WHERE key='total_budget'"
            ).fetchone()
            if row is None:
                self.connection.execute(
                    "INSERT INTO budget_meta(key,value) VALUES('total_budget',?)",
                    (json.dumps(self.total_budget),),
                )
                self.connection.execute(
                    "INSERT INTO budget_meta(key,value) VALUES('memory_fraction',?)",
                    (json.dumps(self.memory_fraction),),
                )
            else:
                existing = float(json.loads(row["value"]))
                if abs(existing - self.total_budget) > 1e-9:
                    raise ValueError(
                        f"budget ledger already has total_budget={existing}; "
                        f"resume with the same denominator or a fresh ledger"
                    )

    @property
    def memory_allocation(self) -> float:
        return self.total_budget * self.memory_fraction

    def _sum_for(self, category: str | None = None, statuses: tuple[str, ...] = ("reserved", "committed")) -> float:
        clauses = ["status IN (%s)" % ",".join("?" for _ in statuses)]
        values: list[Any] = list(statuses)
        if category is not None:
            clauses.append("category=?")
            values.append(category)
        row = self.connection.execute(
            f"SELECT COALESCE(SUM(amount),0.0) AS amount FROM budget_entries WHERE {' AND '.join(clauses)}",
            values,
        ).fetchone()
        return float(row["amount"])

    def _memory_used(self) -> float:
        placeholders = ",".join("?" for _ in self.memory_categories)
        row = self.connection.execute(
            f"SELECT COALESCE(SUM(amount),0.0) AS amount FROM budget_entries "
            f"WHERE status IN ('reserved','committed') AND category IN ({placeholders})",
            tuple(sorted(self.memory_categories)),
        ).fetchone()
        return float(row["amount"])

    def remaining(self, category: str | None = None) -> float:
        used = self._sum_for(category)
        category_remaining = float("inf")
        if category is not None and category in self.category_caps:
            category_remaining = self.category_caps[category] - used
        total_remaining = self.total_budget - self._sum_for()
        memory_remaining = self.memory_allocation - self._memory_used()
        values = [total_remaining, category_remaining]
        if category in self.memory_categories:
            values.append(memory_remaining)
        return max(0.0, min(values))

    def reserve(
        self,
        category: str,
        amount: float,
        *,
        call_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Reservation:
        amount = float(amount)
        category = str(category)
        if amount < 0:
            raise ValueError("reservation amount must be non-negative")
        if amount == 0:
            amount = 0.0
        with self._transaction():
            if not self._can_reserve(category, amount):
                raise BudgetExceeded(
                    f"budget exhausted for category={category!r}: requested={amount} "
                    f"remaining={self.remaining(category)}"
                )
            reservation_id = f"rsv-{uuid.uuid4().hex}"
            self.connection.execute(
                """INSERT INTO budget_entries
                   (reservation_id,category,amount,status,call_id,metadata,created_at)
                   VALUES(?,?,?,?,?,?,datetime('now'))""",
                (reservation_id, category, amount, "reserved", call_id,
                 json.dumps(dict(metadata or {}), sort_keys=True)),
            )
        return Reservation(self, reservation_id, amount)

    def _can_reserve(self, category: str, amount: float) -> bool:
        total_used = self._sum_for()
        if total_used + amount > self.total_budget + 1e-9:
            return False
        if category in self.category_caps:
            if self._sum_for(category) + amount > self.category_caps[category] + 1e-9:
                return False
        if category in self.memory_categories:
            if self._memory_used() + amount > self.memory_allocation + 1e-9:
                return False
        return True

    def commit(
        self,
        reservation_id: str,
        actual_amount: float | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Commit actual usage.  Returns False for an already-closed reservation.

        ``actual_amount`` must not exceed the reserved conservative bound.
        """
        with self._transaction():
            row = self.connection.execute(
                "SELECT * FROM budget_entries WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown reservation: {reservation_id}")
            if row["status"] != "reserved":
                return False
            reserved = float(row["amount"])
            actual = reserved if actual_amount is None else float(actual_amount)
            if actual < 0:
                raise BudgetViolation("actual_amount must be non-negative")
            if actual > reserved + 1e-9:
                raise BudgetViolation(
                    f"actual usage {actual} exceeds reserved bound {reserved}; "
                    "reserve conservatively before the call"
                )
            merged = json.loads(row["metadata"])
            merged.update(dict(metadata or {}))
            self.connection.execute(
                """UPDATE budget_entries
                   SET amount=?, status='committed', metadata=?, closed_at=datetime('now')
                   WHERE reservation_id=?""",
                (actual, json.dumps(merged, sort_keys=True), reservation_id),
            )
        return True

    def release(self, reservation_id: str, *, reason: str = "") -> bool:
        with self._transaction():
            row = self.connection.execute(
                "SELECT * FROM budget_entries WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown reservation: {reservation_id}")
            if row["status"] != "reserved":
                return False
            metadata = json.loads(row["metadata"])
            if reason:
                metadata["release_reason"] = reason
            self.connection.execute(
                """UPDATE budget_entries
                   SET status='released', metadata=?, closed_at=datetime('now')
                   WHERE reservation_id=?""",
                (json.dumps(metadata, sort_keys=True), reservation_id),
            )
        return True

    def spend(
        self,
        category: str,
        amount: float,
        *,
        call_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        reservation = self.reserve(category, amount, call_id=call_id, metadata=metadata)
        reservation.commit(amount)

    def report(self) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT category,status,COALESCE(SUM(amount),0.0) AS amount "
            "FROM budget_entries GROUP BY category,status ORDER BY category,status"
        ).fetchall()
        categories: dict[str, dict[str, float]] = {}
        for row in rows:
            categories.setdefault(row["category"], {})[row["status"]] = float(row["amount"])
        return {
            "total_budget": self.total_budget,
            "memory_fraction": self.memory_fraction,
            "memory_allocation": self.memory_allocation,
            "spent": self._sum_for(statuses=("committed",)),
            "reserved": self._sum_for(statuses=("reserved",)),
            "remaining": max(0.0, self.total_budget - self._sum_for()),
            "memory_spent": self._memory_used(),
            "memory_remaining": max(0.0, self.memory_allocation - self._memory_used()),
            "categories": categories,
        }

    def list_entries(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM budget_entries ORDER BY created_at, reservation_id"
        ).fetchall()
        return [
            {
                "reservation_id": row["reservation_id"],
                "category": row["category"],
                "amount": float(row["amount"]),
                "status": row["status"],
                "call_id": row["call_id"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]
