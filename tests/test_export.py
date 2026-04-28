"""Tests for export bundle creation and verification."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path


from mr_audit import AuditContext, auditable
from mr_audit.export import export_bundle, verify_bundle


def _populate_log(db: Path, n: int = 5) -> None:
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x):
        return x * 2.0

    with AuditContext(db) as ctx:
        for i in range(n):
            predict(x=float(i))


def test_export_creates_zip_with_expected_files(tmp_path):
    db = tmp_path / "log.db"
    bundle = tmp_path / "bundle.zip"
    _populate_log(db, n=3)

    from mr_audit.core.store import SQLiteStore
    records = SQLiteStore(db).read_all()
    export_bundle(records, bundle, source_log_path=str(db))

    assert bundle.exists()
    with zipfile.ZipFile(bundle, "r") as z:
        names = set(z.namelist())
    assert names == {
        "manifest.json",
        "manifest.sha256",
        "audit_records.jsonl",
        "chain_verification.json",
        "README.txt",
    }


def test_bundle_verification_passes_for_clean_export(tmp_path):
    db = tmp_path / "log.db"
    bundle = tmp_path / "bundle.zip"
    _populate_log(db, n=10)

    from mr_audit.core.store import SQLiteStore
    records = SQLiteStore(db).read_all()
    export_bundle(records, bundle)

    result = verify_bundle(bundle)
    assert result["overall_valid"] is True
    assert result["manifest_sha_match"] is True
    assert result["chain_valid"] is True
    assert result["n_records"] == 10
    assert all(result["files_sha_match"].values())
    assert result["errors"] == []


def test_tampered_bundle_detected(tmp_path):
    """If a record in audit_records.jsonl is changed, verification must fail."""
    db = tmp_path / "log.db"
    bundle = tmp_path / "bundle.zip"
    _populate_log(db, n=5)

    from mr_audit.core.store import SQLiteStore
    records = SQLiteStore(db).read_all()
    export_bundle(records, bundle)

    # Tamper: replace audit_records.jsonl with an altered version
    with zipfile.ZipFile(bundle, "r") as z:
        original = z.read("audit_records.jsonl").decode("utf-8")

    tampered = original.replace('"value": 0.0', '"value": 99.0', 1)
    if tampered == original:
        # If 0.0 not present (e.g., float repr edge case), tamper differently
        tampered = original + "\n"

    # Rewrite the zip with tampered records
    contents: dict[str, bytes] = {}
    with zipfile.ZipFile(bundle, "r") as z:
        for name in z.namelist():
            contents[name] = z.read(name)
    contents["audit_records.jsonl"] = tampered.encode("utf-8")

    bundle.unlink()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in contents.items():
            z.writestr(name, data)

    result = verify_bundle(bundle)
    assert result["overall_valid"] is False
    assert any("audit_records.jsonl" in e for e in result["errors"])


def test_manifest_sha_matches_manifest_content(tmp_path):
    """manifest.sha256 must hash manifest.json bit-identically."""
    db = tmp_path / "log.db"
    bundle = tmp_path / "bundle.zip"
    _populate_log(db, n=3)

    from mr_audit.core.store import SQLiteStore
    records = SQLiteStore(db).read_all()
    export_bundle(records, bundle)

    import hashlib

    with zipfile.ZipFile(bundle, "r") as z:
        manifest_bytes = z.read("manifest.json")
        sha_text = z.read("manifest.sha256").decode("utf-8").strip().split()[0]
    actual_sha = hashlib.sha256(manifest_bytes).hexdigest()
    assert sha_text == actual_sha


def test_round_trip_bundle_to_records(tmp_path):
    """audit_records.jsonl in the bundle must be parseable back to records."""
    db = tmp_path / "log.db"
    bundle = tmp_path / "bundle.zip"
    _populate_log(db, n=4)

    from mr_audit.core.store import SQLiteStore
    original_records = SQLiteStore(db).read_all()
    export_bundle(original_records, bundle)

    with zipfile.ZipFile(bundle, "r") as z:
        jsonl = z.read("audit_records.jsonl").decode("utf-8")

    lines = [line for line in jsonl.split("\n") if line.strip()]
    assert len(lines) == len(original_records)
    parsed = [json.loads(line) for line in lines]
    for orig, p in zip(original_records, parsed):
        assert p["record_id"] == orig.record_id
        assert p["model"]["name"] == orig.model.name


def test_empty_log_export(tmp_path):
    """Empty log should still produce a valid (empty) bundle."""
    bundle = tmp_path / "empty.zip"
    export_bundle([], bundle, source_log_path="empty")
    result = verify_bundle(bundle)
    assert result["overall_valid"] is True
    assert result["n_records"] == 0
