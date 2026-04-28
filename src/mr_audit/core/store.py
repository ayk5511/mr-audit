"""Storage backends for audit records.

Two backends, both vendor-neutral and locally-runnable:

  - SQLiteStore:   SQLite database; queryable, robust, append-only by convention.
                   Suited for production-style use with millions of records.
  - JSONLStore:    JSON Lines file; append-only by file structure.
                   Suited for human inspection, lightweight pipelines, easy diffing.

Both backends preserve insertion order (the order in which records are written),
which is what hash-chain integrity depends on.

Backends are intentionally write-only-append. The API does not expose a
"delete" or "update" operation. To inspect records, use read_all().
"""
from __future__ import annotations

import abc
import json
import sqlite3
from pathlib import Path

from .hash import seal_record
from .schema import AuditRecord, CodeState, ComputeState, DataState, ModelState, Prediction


class AuditStore(abc.ABC):
    """Abstract backend for audit records."""

    @abc.abstractmethod
    def append(self, record: AuditRecord) -> AuditRecord:
        """Seal (compute hash + chain) and persist `record`. Return sealed record."""

    @abc.abstractmethod
    def read_all(self) -> list[AuditRecord]:
        """Return all records in insertion order."""

    @abc.abstractmethod
    def latest_record_id(self) -> str | None:
        """Return the record_id of the most recent record, or None if empty."""


class SQLiteStore(AuditStore):
    """SQLite-backed audit store.

    Schema (table `audit_records`):
        id              INTEGER PRIMARY KEY AUTOINCREMENT  -- insertion order
        record_id       TEXT NOT NULL UNIQUE
        prev_record_hash TEXT
        timestamp_utc   TEXT NOT NULL
        json            TEXT NOT NULL                       -- canonical JSON
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS audit_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id TEXT NOT NULL UNIQUE,
        prev_record_hash TEXT,
        timestamp_utc TEXT NOT NULL,
        json TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_record_id ON audit_records(record_id);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_records(timestamp_utc);
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    def append(self, record: AuditRecord) -> AuditRecord:
        # Seal: chain + compute record_id
        prev = self.latest_record_id()
        from dataclasses import replace
        chained = replace(record, prev_record_hash=prev)
        sealed = seal_record(chained)

        self._conn.execute(
            "INSERT INTO audit_records (record_id, prev_record_hash, timestamp_utc, json) "
            "VALUES (?, ?, ?, ?)",
            (
                sealed.record_id,
                sealed.prev_record_hash,
                sealed.timestamp_utc,
                sealed.to_json(),
            ),
        )
        self._conn.commit()
        return sealed

    def read_all(self) -> list[AuditRecord]:
        cur = self._conn.execute(
            "SELECT json FROM audit_records ORDER BY id ASC"
        )
        return [_record_from_json(row[0]) for row in cur.fetchall()]

    def latest_record_id(self) -> str | None:
        cur = self._conn.execute(
            "SELECT record_id FROM audit_records ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._conn.close()


class JSONLStore(AuditStore):
    """JSON Lines audit store. One record per line, append-only.

    Append throughput is good. Random access requires a full scan; suited for
    lightweight pipelines or human inspection.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, record: AuditRecord) -> AuditRecord:
        prev = self.latest_record_id()
        from dataclasses import replace
        chained = replace(record, prev_record_hash=prev)
        sealed = seal_record(chained)

        with self.path.open("a", encoding="utf-8") as f:
            f.write(sealed.to_json())
            f.write("\n")
        return sealed

    def read_all(self) -> list[AuditRecord]:
        if not self.path.exists():
            return []
        records: list[AuditRecord] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(_record_from_json(line))
        return records

    def latest_record_id(self) -> str | None:
        # Linear scan; fine for moderate sizes
        records = self.read_all()
        return records[-1].record_id if records else None


def _record_from_json(s: str) -> AuditRecord:
    """Reconstruct an AuditRecord from its canonical JSON form."""
    d = json.loads(s)
    return AuditRecord(
        schema_version=d["schema_version"],
        timestamp_utc=d["timestamp_utc"],
        code=CodeState(**d["code"]),
        data=DataState(**{**d["data"], "feature_names": tuple(d["data"].get("feature_names", []))}),
        model=ModelState(**d["model"]),
        compute=ComputeState(**d["compute"]),
        prediction=Prediction(**d["prediction"]),
        random_seed=d.get("random_seed"),
        prev_record_hash=d.get("prev_record_hash"),
        record_id=d.get("record_id", ""),
    )
