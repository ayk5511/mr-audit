"""Tests for the hash chain (mr_audit.core.hash + mr_audit.core.schema)."""
from __future__ import annotations

from dataclasses import replace


from mr_audit.core.hash import compute_record_id, seal_record, verify_chain
from mr_audit.core.schema import (
    SCHEMA_VERSION,
    AuditRecord,
    CodeState,
    ComputeState,
    DataState,
    ModelState,
    Prediction,
)


def make_record(prev_hash=None, data_hash="abc123") -> AuditRecord:
    return AuditRecord(
        schema_version=SCHEMA_VERSION,
        timestamp_utc="2026-04-28T12:00:00+00:00",
        code=CodeState(),
        data=DataState(input_data_hash=data_hash),
        model=ModelState(name="TestModel", version="1.0.0"),
        compute=ComputeState(python_version="3.14.0"),
        prediction=Prediction(value=1.0),
        random_seed=42,
        prev_record_hash=prev_hash,
    )


def test_compute_record_id_is_sha256_hex_64_chars():
    rec = make_record()
    rid = compute_record_id(rec)
    assert isinstance(rid, str)
    assert len(rid) == 64
    int(rid, 16)  # parses as hex


def test_compute_record_id_excludes_record_id_field():
    """A record's record_id field must not affect its own hash."""
    rec = make_record()
    rid1 = compute_record_id(rec)
    rec_with_id = replace(rec, record_id="dummy")
    rid2 = compute_record_id(rec_with_id)
    assert rid1 == rid2


def test_seal_record_sets_record_id():
    rec = make_record()
    sealed = seal_record(rec)
    assert sealed.record_id != ""
    assert sealed.record_id == compute_record_id(rec)


def test_chain_of_three_records_verifies():
    r1 = seal_record(make_record(prev_hash=None, data_hash="d1"))
    r2 = seal_record(make_record(prev_hash=r1.record_id, data_hash="d2"))
    r3 = seal_record(make_record(prev_hash=r2.record_id, data_hash="d3"))
    valid, errors = verify_chain([r1, r2, r3])
    assert valid is True
    assert errors == []


def test_tampered_record_breaks_chain():
    r1 = seal_record(make_record(prev_hash=None, data_hash="d1"))
    r2 = seal_record(make_record(prev_hash=r1.record_id, data_hash="d2"))
    r3 = seal_record(make_record(prev_hash=r2.record_id, data_hash="d3"))

    # Tamper with r2's data hash but leave its record_id unchanged
    r2_tampered = replace(r2, data=DataState(input_data_hash="d2_TAMPERED"))
    valid, errors = verify_chain([r1, r2_tampered, r3])
    assert valid is False
    assert len(errors) >= 1
    assert any("record_id mismatch" in e for e in errors)


def test_genesis_record_must_have_none_prev_hash():
    r1 = seal_record(make_record(prev_hash="not-genesis"))
    valid, errors = verify_chain([r1])
    assert valid is False
    assert any("genesis record has non-None prev_record_hash" in e for e in errors)


def test_chain_break_detected_at_link():
    r1 = seal_record(make_record(prev_hash=None, data_hash="d1"))
    r2 = seal_record(make_record(prev_hash=r1.record_id, data_hash="d2"))
    # r3 claims to follow r1 (skipping r2)
    r3 = seal_record(make_record(prev_hash=r1.record_id, data_hash="d3"))
    valid, errors = verify_chain([r1, r2, r3])
    assert valid is False
    assert any("prev_record_hash mismatch" in e for e in errors)


def test_canonical_json_is_deterministic():
    """Hash must be identical across repeated calls."""
    rec = make_record()
    h1 = compute_record_id(rec)
    h2 = compute_record_id(rec)
    h3 = compute_record_id(rec)
    assert h1 == h2 == h3


def test_different_predictions_give_different_hashes():
    r1 = make_record()
    r2 = replace(r1, prediction=Prediction(value=2.0))
    assert compute_record_id(r1) != compute_record_id(r2)
