"""SHA-256 hash chain for audit-record integrity.

Each AuditRecord's record_id is the SHA-256 of its canonical-JSON serialisation
(excluding record_id itself). Each record carries the prev_record_hash
(=previous record's record_id), forming a Merkle-style chain.

Why SHA-256: NIST-approved (FIPS 180-4); 256-bit output; ubiquitous; no known
practical preimage or collision attacks. Adequate for the integrity-protection
use case here. Aligns with EU AI Act Article 19's "automatically generated
logs" intent, though the regulation does not specify a hash algorithm.
"""
from __future__ import annotations

import hashlib

from .schema import AuditRecord


def sha256_hex(data: bytes) -> str:
    """Return the hex-encoded SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def compute_record_id(record: AuditRecord) -> str:
    """Compute the canonical hash of an audit record.

    The record's record_id field is excluded from the hash input (it cannot
    contain its own digest). All other fields are serialised to canonical JSON
    (sorted keys, minimal whitespace) and SHA-256-hashed.
    """
    canonical = record.to_json(include_record_id=False)
    return sha256_hex(canonical.encode("utf-8"))


def seal_record(record: AuditRecord) -> AuditRecord:
    """Return a new AuditRecord with record_id set to the canonical hash.

    This is the function called by the store on write; it should be the only
    place where record_id is populated.
    """
    rid = compute_record_id(record)
    # dataclasses are frozen; replace via __dataclass_fields__-aware copy
    from dataclasses import replace
    return replace(record, record_id=rid)


def verify_chain(records: list[AuditRecord]) -> tuple[bool, list[str]]:
    """Verify hash chain integrity for a sequence of records.

    Returns (is_valid, list_of_errors).
    Errors are strings describing the first defect for each broken record.
    Empty list means the chain is intact.

    Checks:
      1. Each record's record_id matches its canonical hash.
      2. Each record's prev_record_hash matches the previous record's record_id.
      3. The first record's prev_record_hash is None (genesis).
    """
    errors: list[str] = []
    for i, rec in enumerate(records):
        # Check: record_id is canonical hash of the rest
        expected_id = compute_record_id(rec)
        if rec.record_id != expected_id:
            errors.append(
                f"record[{i}]: record_id mismatch (stored={rec.record_id[:16]}..., "
                f"expected={expected_id[:16]}...)"
            )
            continue
        # Check: prev_record_hash matches predecessor
        if i == 0:
            if rec.prev_record_hash is not None:
                errors.append(
                    f"record[{i}]: genesis record has non-None prev_record_hash"
                )
        else:
            expected_prev = records[i - 1].record_id
            if rec.prev_record_hash != expected_prev:
                errors.append(
                    f"record[{i}]: prev_record_hash mismatch "
                    f"(stored={rec.prev_record_hash[:16] if rec.prev_record_hash else 'None'}..., "
                    f"expected={expected_prev[:16]}...)"
                )
    return (len(errors) == 0, errors)
