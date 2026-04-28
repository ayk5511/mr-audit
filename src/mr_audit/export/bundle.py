"""Audit bundle export.

The "regulator export bundle" is a self-contained zip archive that lets an
external reviewer (regulator, model-validation team, internal audit) verify
every prediction without access to the original system.

Bundle contents:

    audit_bundle.zip
    ├── manifest.json          # bundle metadata + SHA-256 of every contents file
    ├── audit_records.jsonl    # all records, canonical-JSON, one per line
    ├── chain_verification.json # hash-chain verification result at export time
    ├── README.txt             # human-readable description + replay instructions
    └── manifest.sha256        # external-tool-friendly checksum

Design choices:

    1. JSON Lines for portability. SQLite is great for storage but JSONL is
       what auditors and downstream tools (jq, pandas, R) handle best.
    2. The manifest carries SHA-256 of every other bundle file. The manifest
       itself has its own SHA-256 in manifest.sha256 (a second file) so the
       bundle's integrity can be checked with off-the-shelf tools without
       any mr-audit-specific code.
    3. The bundle is timestamped and tamper-evident at export time. We do
       not provide cryptographic signing in this 0.0.x release; signing
       (e.g., GPG, Sigstore) is left for the firm to add per its policy.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ..core.hash import verify_chain
from ..core.schema import AuditRecord


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _records_to_jsonl(records: list[AuditRecord]) -> bytes:
    lines = [r.to_json() for r in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _chain_verification_payload(records: list[AuditRecord]) -> bytes:
    valid, errors = verify_chain(records)
    payload = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_records": len(records),
        "chain_valid": valid,
        "errors": errors,
    }
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _readme_text(n_records: int, source_path: str) -> bytes:
    text = f"""mr-audit Export Bundle
=======================

This bundle was produced by mr-audit (https://github.com/ayk5511/mr-audit)
on {datetime.now(timezone.utc).isoformat()} UTC.

Source log:    {source_path}
Records:       {n_records}

Contents
--------
manifest.json           Bundle metadata + SHA-256 of every contents file
audit_records.jsonl     All audit records, one per line, canonical JSON
chain_verification.json Hash-chain verification at export time
manifest.sha256         Off-the-shelf integrity check on manifest.json
README.txt              This file

Verification (without mr-audit installed)
------------------------------------------
1. Compute SHA-256 of manifest.json:
       sha256sum manifest.json
   Compare to manifest.sha256.

2. For each file listed in manifest.json's `files` field, compute SHA-256:
       sha256sum <filename>
   Compare to the digest in manifest.json.

Replay (with mr-audit installed)
---------------------------------
    pip install mr-audit
    mr-audit verify <path-to-bundle.zip>
    mr-audit replay <path-to-bundle.zip> --record-id <id>
"""
    return text.encode("utf-8")


def export_bundle(
    records: list[AuditRecord],
    output_path: str | Path,
    *,
    source_log_path: str = "(unspecified)",
) -> Path:
    """Write a regulator-export bundle from an in-memory list of records.

    Args:
        records: chronologically-ordered list of AuditRecord (e.g., from
                 store.read_all()).
        output_path: where to write the .zip archive.
        source_log_path: human-readable identifier of the source log, recorded
                         in the manifest and README. Optional metadata only.

    Returns:
        Path to the written bundle.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the four primary artifacts
    records_jsonl = _records_to_jsonl(records)
    chain_json = _chain_verification_payload(records)
    readme = _readme_text(len(records), source_log_path)

    files: dict[str, bytes] = {
        "audit_records.jsonl": records_jsonl,
        "chain_verification.json": chain_json,
        "README.txt": readme,
    }

    # Build manifest with hashes of each file
    manifest = {
        "schema": "mr-audit-bundle-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_log_path": source_log_path,
        "n_records": len(records),
        "files": {
            name: {
                "sha256": _sha256_hex(content),
                "size_bytes": len(content),
            }
            for name, content in files.items()
        },
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    manifest_sha = _sha256_hex(manifest_bytes)

    # Write the zip
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", manifest_bytes)
        z.writestr("manifest.sha256", manifest_sha + "  manifest.json\n")
        for name, content in files.items():
            z.writestr(name, content)

    return output_path


def verify_bundle(bundle_path: str | Path) -> dict:
    """Verify an exported bundle without using any mr-audit private state.

    Returns a dict with verification results suitable for human or machine
    consumption. The bundle is considered intact if:

        - manifest.sha256 matches sha256(manifest.json)
        - every file listed in manifest.json has matching sha256 inside the zip
        - chain_verification.json says chain_valid: true
    """
    bundle_path = Path(bundle_path)
    result: dict = {
        "bundle_path": str(bundle_path),
        "manifest_sha_match": False,
        "files_sha_match": {},
        "chain_valid": None,
        "n_records": None,
        "errors": [],
    }
    with zipfile.ZipFile(bundle_path, "r") as z:
        manifest_bytes = z.read("manifest.json")
        manifest_sha_file = z.read("manifest.sha256").decode("utf-8").strip().split()[0]
        actual_manifest_sha = _sha256_hex(manifest_bytes)
        result["manifest_sha_match"] = manifest_sha_file == actual_manifest_sha
        if not result["manifest_sha_match"]:
            result["errors"].append(
                f"manifest.sha256 mismatch: file says {manifest_sha_file}, "
                f"actual is {actual_manifest_sha}"
            )

        manifest = json.loads(manifest_bytes)
        for name, info in manifest["files"].items():
            try:
                content = z.read(name)
                actual = _sha256_hex(content)
                expected = info["sha256"]
                ok = actual == expected
                result["files_sha_match"][name] = ok
                if not ok:
                    result["errors"].append(
                        f"{name}: hash mismatch (file={actual}, manifest={expected})"
                    )
            except KeyError:
                result["files_sha_match"][name] = False
                result["errors"].append(f"{name}: missing from bundle")

        chain_payload = json.loads(z.read("chain_verification.json"))
        result["chain_valid"] = chain_payload.get("chain_valid")
        result["n_records"] = chain_payload.get("n_records")
        if not result["chain_valid"]:
            result["errors"].extend(chain_payload.get("errors", []))

    result["overall_valid"] = (
        result["manifest_sha_match"]
        and all(result["files_sha_match"].values())
        and result["chain_valid"]
    )
    return result
