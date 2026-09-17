"""Small transactional SQLite store for MMH state and audit history."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from .types import Patch, PatchOperation, PatchStatus, Precedent, Rule, RuleTier, ValidationEvidence


class SQLiteStore:
    """Persistence boundary for MMH.

    Whole Rule/Patch objects are JSON encoded so the paper-level schema can evolve,
    while indexed columns retain efficient access to lifecycle state.  Every state
    transition is performed under ``transaction`` to survive interrupted runs.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rules (
                    rule_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_patch_id TEXT,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rules_tier ON rules(tier, status);
                CREATE TABLE IF NOT EXISTS patches (
                    patch_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    created_round INTEGER NOT NULL,
                    resolved_round INTEGER,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_patches_status ON patches(status);
                CREATE TABLE IF NOT EXISTS precedents (
                    patch_id TEXT PRIMARY KEY,
                    outcome INTEGER NOT NULL,
                    operation TEXT NOT NULL,
                    resolved_round INTEGER NOT NULL,
                    embedding TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patch_id TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    subset_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    original_failure_recovered INTEGER,
                    applicable INTEGER NOT NULL,
                    details TEXT NOT NULL,
                    UNIQUE(patch_id, subset_id, round_id),
                    FOREIGN KEY(patch_id) REFERENCES patches(patch_id)
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_patch ON evidence(patch_id, round_id);
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def transaction(self) -> Iterator[None]:
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
    def _encode(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def put_rule(self, rule: Rule) -> None:
        self.connection.execute(
            """INSERT INTO rules(rule_id,tier,status,source_patch_id,data) VALUES(?,?,?,?,?)
            ON CONFLICT(rule_id) DO UPDATE SET tier=excluded.tier,status=excluded.status,
            source_patch_id=excluded.source_patch_id,data=excluded.data""",
            (rule.rule_id, rule.tier.value, rule.status.value, rule.source_patch_id, self._encode(rule.to_dict())),
        )

    def get_rule(self, rule_id: str) -> Rule | None:
        row = self.connection.execute("SELECT data FROM rules WHERE rule_id=?", (rule_id,)).fetchone()
        return Rule.from_dict(json.loads(row["data"])) if row else None

    def list_rules(self, tier: RuleTier | None = None, include_pending: bool = True) -> list[Rule]:
        clauses: list[str] = []
        values: list[str] = []
        if tier is not None:
            clauses.append("tier=?")
            values.append(tier.value)
        if not include_pending:
            clauses.append("status != ?")
            values.append(PatchStatus.PENDING.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(f"SELECT data FROM rules{where} ORDER BY rule_id", values).fetchall()
        return [Rule.from_dict(json.loads(row["data"])) for row in rows]

    def delete_rule(self, rule_id: str) -> None:
        self.connection.execute("DELETE FROM rules WHERE rule_id=?", (rule_id,))

    def put_patch(self, patch: Patch) -> None:
        self.connection.execute(
            """INSERT INTO patches(patch_id,status,operation,created_round,resolved_round,data)
            VALUES(?,?,?,?,?,?) ON CONFLICT(patch_id) DO UPDATE SET status=excluded.status,
            operation=excluded.operation,created_round=excluded.created_round,
            resolved_round=excluded.resolved_round,data=excluded.data""",
            (patch.patch_id, patch.status.value, patch.operation.value, patch.created_round,
             patch.resolved_round, self._encode(patch.to_dict())),
        )

    def get_patch(self, patch_id: str) -> Patch | None:
        row = self.connection.execute("SELECT data FROM patches WHERE patch_id=?", (patch_id,)).fetchone()
        return Patch.from_dict(json.loads(row["data"])) if row else None

    def list_patches(self, status: PatchStatus | None = None) -> list[Patch]:
        if status is None:
            rows = self.connection.execute("SELECT data FROM patches ORDER BY patch_id").fetchall()
        else:
            rows = self.connection.execute("SELECT data FROM patches WHERE status=? ORDER BY patch_id", (status.value,)).fetchall()
        return [Patch.from_dict(json.loads(row["data"])) for row in rows]

    def add_evidence(self, evidence: ValidationEvidence) -> bool:
        """Add unique evidence. Returns false for a resumed/duplicate observation."""
        result = self.connection.execute(
            """INSERT OR IGNORE INTO evidence(patch_id,success,subset_id,round_id,
            original_failure_recovered,applicable,details) VALUES(?,?,?,?,?,?,?)""",
            (evidence.patch_id, int(evidence.success), evidence.subset_id, evidence.round_id,
             None if evidence.original_failure_recovered is None else int(evidence.original_failure_recovered),
             int(evidence.applicable), self._encode(evidence.details)),
        )
        return result.rowcount == 1

    def evidence_for_patch(self, patch_id: str) -> list[ValidationEvidence]:
        rows = self.connection.execute(
            "SELECT * FROM evidence WHERE patch_id=? ORDER BY round_id,id", (patch_id,)
        ).fetchall()
        return [
            ValidationEvidence(
                patch_id=row["patch_id"], success=bool(row["success"]), subset_id=row["subset_id"],
                round_id=row["round_id"],
                original_failure_recovered=(None if row["original_failure_recovered"] is None else bool(row["original_failure_recovered"])),
                applicable=bool(row["applicable"]), details=json.loads(row["details"]),
            )
            for row in rows
        ]

    def add_precedent(self, precedent: Precedent) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO precedents(patch_id,outcome,operation,resolved_round,embedding)
            VALUES(?,?,?,?,?)""",
            (precedent.patch_id, int(precedent.outcome), precedent.operation.value,
             precedent.resolved_round, self._encode(list(precedent.context_embedding))),
        )

    def list_precedents(self) -> list[Precedent]:
        rows = self.connection.execute("SELECT * FROM precedents ORDER BY resolved_round,patch_id").fetchall()
        return [
            Precedent(row["patch_id"], tuple(json.loads(row["embedding"])), bool(row["outcome"]),
                      PatchOperation(row["operation"]), row["resolved_round"])
            for row in rows
        ]

    def get_metadata(self, key: str, default: object = None) -> object:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def set_metadata(self, key: str, value: object) -> None:
        self.connection.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, self._encode(value)),
        )
