"""Tests for SQLiteStore and JSONLStore."""
from __future__ import annotations


import pytest

from mr_audit.core.hash import verify_chain
from mr_audit.core.schema import (
    SCHEMA_VERSION,
    AuditRecord,
    CodeState,
    ComputeState,
    DataState,
    ModelState,
    Prediction,
)
from mr_audit.core.store import JSONLStore, SQLiteStore


def make_record(data_hash="abc") -> AuditRecord:
    return AuditRecord(
        schema_version=SCHEMA_VERSION,
        timestamp_utc="2026-04-28T12:00:00+00:00",
        code=CodeState(library_versions={"numpy": "2.1.0"}),
        data=DataState(input_data_hash=data_hash, feature_names=("a", "b")),
        model=ModelState(name="M", version="1.0.0", parameters={"alpha": 0.1}),
        compute=ComputeState(python_version="3.14.0"),
        prediction=Prediction(value=1.5, horizon_days=5),
        random_seed=42,
    )


@pytest.fixture(params=["sqlite", "jsonl"])
def store(tmp_path, request):
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "test.db")
    return JSONLStore(tmp_path / "test.jsonl")


def test_empty_store_has_no_latest(store):
    assert store.latest_record_id() is None
    assert store.read_all() == []


def test_append_and_read_back(store):
    r = make_record()
    sealed = store.append(r)
    assert sealed.record_id != ""
    records = store.read_all()
    assert len(records) == 1
    assert records[0].record_id == sealed.record_id


def test_chain_links_through_appends(store):
    sealed1 = store.append(make_record(data_hash="d1"))
    sealed2 = store.append(make_record(data_hash="d2"))
    sealed3 = store.append(make_record(data_hash="d3"))
    records = store.read_all()
    assert len(records) == 3
    assert records[0].prev_record_hash is None
    assert records[1].prev_record_hash == sealed1.record_id
    assert records[2].prev_record_hash == sealed2.record_id
    valid, _ = verify_chain(records)
    assert valid


def test_records_persist_across_reopen(tmp_path):
    """Both backends must persist between sessions."""
    db_path = tmp_path / "persist.db"
    s1 = SQLiteStore(db_path)
    s1.append(make_record(data_hash="d1"))
    s1.append(make_record(data_hash="d2"))
    s1.close()

    s2 = SQLiteStore(db_path)
    records = s2.read_all()
    assert len(records) == 2

    jsonl_path = tmp_path / "persist.jsonl"
    s3 = JSONLStore(jsonl_path)
    s3.append(make_record(data_hash="d1"))
    s3.append(make_record(data_hash="d2"))

    s4 = JSONLStore(jsonl_path)
    records = s4.read_all()
    assert len(records) == 2


def test_record_id_unique_in_sqlite(tmp_path):
    """SQLiteStore enforces UNIQUE on record_id at the DB level."""
    s = SQLiteStore(tmp_path / "unique.db")
    s.append(make_record(data_hash="d1"))
    s.append(make_record(data_hash="d2"))
    # Different data hash -> different record_id, both succeed
    records = s.read_all()
    assert records[0].record_id != records[1].record_id


def test_jsonl_one_line_per_record(tmp_path):
    """JSONL format must be one record per line."""
    p = tmp_path / "format.jsonl"
    s = JSONLStore(p)
    s.append(make_record(data_hash="d1"))
    s.append(make_record(data_hash="d2"))
    s.append(make_record(data_hash="d3"))
    lines = p.read_text().strip().split("\n")
    assert len(lines) == 3
    # Each line must be valid JSON
    import json
    for line in lines:
        json.loads(line)
