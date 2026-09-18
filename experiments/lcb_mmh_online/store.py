"""Versioned SQLite source of truth for the online-MMH experiment.

The paper-level rules/patches remain in the existing ``SQLiteStore`` schema so
the legacy engine can be reused.  This subclass adds the online-specific
tables in the same database file, avoiding an unrepairable JSON/SQLite
two-file commit protocol.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from meta_memory import SQLiteStore

from .types import Application, ApplicationStatus, MemoryEvidence


ONLINE_SCHEMA_VERSION = 2


class OnlineStore(SQLiteStore):
    """Extend ``SQLiteStore`` rather than replacing it.

    ``SQLiteStore.__init__`` creates the legacy tables; we then create the
    application/evidence tables on the same connection and transaction.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        super().__init__(path)
        self._create_online_schema()
        persisted = self.get_metadata("online_schema_version", None)
        if persisted is None:
            self.set_metadata("online_schema_version", ONLINE_SCHEMA_VERSION)
            self.connection.commit()
        elif int(persisted) > ONLINE_SCHEMA_VERSION:
            raise ValueError(f"online MMH schema is newer than this code: {persisted}")

    def _create_online_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS online_applications (
                    application_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_batch INTEGER NOT NULL,
                    created_round TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_online_applications_status
                    ON online_applications(status, created_batch);
                CREATE TABLE IF NOT EXISTS online_evidence (
                    uniqueness_key TEXT PRIMARY KEY,
                    application_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    test_fingerprint TEXT NOT NULL,
                    batch_id INTEGER NOT NULL,
                    round_id INTEGER NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_online_evidence_application
                    ON online_evidence(application_id, batch_id);
                CREATE INDEX IF NOT EXISTS idx_online_evidence_task
                    ON online_evidence(task_id, test_fingerprint);
                CREATE TABLE IF NOT EXISTS online_recoveries (
                    application_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    test_fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data TEXT NOT NULL,
                    PRIMARY KEY(application_id, task_id, test_fingerprint)
                );
                CREATE TABLE IF NOT EXISTS online_stream (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def put_application(self, application: Application) -> None:
        self.connection.execute(
            """INSERT INTO online_applications
               (application_id,status,created_batch,created_round,data)
               VALUES(?,?,?,?,?)
               ON CONFLICT(application_id) DO UPDATE SET
                 status=excluded.status, created_batch=excluded.created_batch,
                 created_round=excluded.created_round, data=excluded.data""",
            (application.application_id, application.status.value, application.created_batch,
             application.created_round, json.dumps(application.to_dict(), sort_keys=True)),
        )

    def get_application(self, application_id: str) -> Application | None:
        row = self.connection.execute(
            "SELECT data FROM online_applications WHERE application_id=?", (application_id,)
        ).fetchone()
        return Application.from_mapping(json.loads(row["data"])) if row else None

    def list_applications(self, status: ApplicationStatus | str | None = None) -> list[Application]:
        if status is None:
            rows = self.connection.execute(
                "SELECT data FROM online_applications ORDER BY created_batch, application_id"
            ).fetchall()
        else:
            value = status.value if isinstance(status, ApplicationStatus) else str(status)
            rows = self.connection.execute(
                "SELECT data FROM online_applications WHERE status=? ORDER BY created_batch, application_id",
                (value,),
            ).fetchall()
        return [Application.from_mapping(json.loads(row["data"])) for row in rows]

    def add_online_evidence(self, evidence: MemoryEvidence) -> bool:
        """Persist one unique application/task/configuration observation."""
        result = self.connection.execute(
            """INSERT OR IGNORE INTO online_evidence
               (uniqueness_key,application_id,outcome,task_id,test_fingerprint,batch_id,round_id,data)
               VALUES(?,?,?,?,?,?,?,?)""",
            (evidence.uniqueness_key, evidence.application_id, evidence.outcome.value,
             evidence.task_id, evidence.test_fingerprint, evidence.batch_id, evidence.round_id,
             json.dumps(evidence.to_dict(), sort_keys=True)),
        )
        return result.rowcount == 1

    def evidence_for_application(self, application_id: str) -> list[MemoryEvidence]:
        rows = self.connection.execute(
            "SELECT data FROM online_evidence WHERE application_id=? ORDER BY batch_id,round_id,task_id",
            (application_id,),
        ).fetchall()
        return [MemoryEvidence.from_mapping(json.loads(row["data"])) for row in rows]

    def list_online_evidence(self) -> list[MemoryEvidence]:
        rows = self.connection.execute(
            "SELECT data FROM online_evidence ORDER BY batch_id,round_id,task_id"
        ).fetchall()
        return [MemoryEvidence.from_mapping(json.loads(row["data"])) for row in rows]

    def put_recovery(self, application_id: str, task_id: str, test_fingerprint: str,
                     details: Mapping[str, Any]) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO online_recoveries
               (application_id,task_id,test_fingerprint,created_at,data)
               VALUES(?,?,?,datetime('now'),?)""",
            (application_id, task_id, test_fingerprint, json.dumps(dict(details), sort_keys=True)),
        )

    def has_recovery(self, application_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM online_recoveries WHERE application_id=? LIMIT 1", (application_id,)
        ).fetchone()
        return row is not None

    def get_stream(self, key: str, default: object = None) -> object:
        row = self.connection.execute("SELECT value FROM online_stream WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def set_stream(self, key: str, value: object) -> None:
        self.connection.execute(
            """INSERT INTO online_stream(key,value) VALUES(?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (key, json.dumps(value, sort_keys=True)),
        )

    def close(self) -> None:
        super().close()
