"""Artifact/config-aware replay cache.

A cached result can save a call but is never promoted to fresh independent
evidence.  The cache key contains the immutable artifact hashes, task/test
fingerprint, execution configuration, and sampling identity.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .types import canonical_json, utc_now


class ObservationCache:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        with self.connection:
            self.connection.execute(
                """CREATE TABLE IF NOT EXISTS observation_cache (
                       cache_key TEXT PRIMARY KEY,
                       value TEXT NOT NULL,
                       created_at TEXT NOT NULL,
                       last_hit_at TEXT
                   )"""
            )

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

    @staticmethod
    def key(
        *,
        parent_artifact_hash: str,
        child_artifact_hash: str,
        task_id: str,
        test_fingerprint: str,
        execution_config_fingerprint: str,
        sampling_identity: str,
    ) -> str:
        return canonical_json({
            "parent_artifact_hash": parent_artifact_hash,
            "child_artifact_hash": child_artifact_hash,
            "task_id": task_id,
            "test_fingerprint": test_fingerprint,
            "execution_config_fingerprint": execution_config_fingerprint,
            "sampling_identity": sampling_identity,
        })

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with self._transaction():
            row = self.connection.execute(
                "SELECT value FROM observation_cache WHERE cache_key=?", (cache_key,)
            ).fetchone()
            if row is None:
                return None
            self.connection.execute(
                "UPDATE observation_cache SET last_hit_at=? WHERE cache_key=?",
                (utc_now(), cache_key),
            )
        value = json.loads(row["value"])
        value["cache_hit"] = True
        return value

    def put(self, cache_key: str, value: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(value), sort_keys=True)
        with self._transaction():
            self.connection.execute(
                """INSERT INTO observation_cache(cache_key,value,created_at,last_hit_at)
                   VALUES(?,?,?,NULL)
                   ON CONFLICT(cache_key) DO UPDATE SET value=excluded.value""",
                (cache_key, payload, utc_now()),
            )

    def get_or_compute(
        self,
        cache_key: str,
        compute: Callable[[], Mapping[str, Any]],
        *,
        cost_category: str = "memory_validation",
        ledger: Any | None = None,
        estimated_cost: float = 0.0,
        call_id: str | None = None,
    ) -> dict[str, Any]:
        cached = self.get(cache_key)
        if cached is not None:
            result = dict(cached)
            result.setdefault("details", {})
            result["details"]["cache_hit"] = True
            result["details"]["fresh_run"] = False
            return result
        if ledger is not None:
            reservation = ledger.reserve(cost_category, estimated_cost, call_id=call_id,
                                         metadata={"cache_key": cache_key})
            try:
                raw = dict(compute())
            except BaseException:
                reservation.release("compute_failed")
                raise
            actual = float(raw.pop("cost", estimated_cost))
            reservation.commit(actual, metadata={"cache_key": cache_key})
            result = raw
        else:
            result = dict(compute())
        result.setdefault("details", {})
        result["details"]["cache_hit"] = False
        result["details"]["fresh_run"] = True
        self.put(cache_key, result)
        return result

    def stats(self) -> dict[str, int]:
        total = self.connection.execute("SELECT COUNT(*) AS n FROM observation_cache").fetchone()["n"]
        hits = self.connection.execute(
            "SELECT COUNT(*) AS n FROM observation_cache WHERE last_hit_at IS NOT NULL"
        ).fetchone()["n"]
        return {"entries": int(total), "hit_entries": int(hits)}
