"""Audit-record schema for mr-audit.

Defines the structured record written for every ML prediction. The schema is
versioned (SCHEMA_VERSION) so that future evolutions remain backward-readable.

Each AuditRecord captures, per the SR 11-7 / EU AI Act Article 19 requirements:

  - Code state:    git commit hash, library versions
  - Data state:    hash of input features, data-source identifier
  - Model state:   model name, version, parameters, artifact hash
  - Compute state: python version, platform
  - Random state:  random seed
  - Prediction:    value, horizon, predicted-for date
  - Chain link:    record_id (SHA-256 of canonical JSON of this record),
                   prev_record_hash (record_id of the previous record in the
                   chain, or None for the genesis record)

The chain link makes the log Merkle-style: tampering with any record breaks
the hash continuity from that record onward.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "0.1.0"
"""Increment on backward-incompatible schema changes."""


@dataclass(frozen=True)
class CodeState:
    """Identifies the calling code's version and library environment."""
    code_commit_hash: str | None = None
    """Git commit SHA of the calling repository, or None if not in a git repo."""
    code_dirty: bool = False
    """True if the working tree had uncommitted changes at predict time."""
    library_versions: dict[str, str] = field(default_factory=dict)
    """Map of dependency name to version string (e.g., {'numpy': '2.1.0'})."""


@dataclass(frozen=True)
class DataState:
    """Identifies the input data."""
    input_data_hash: str
    """SHA-256 hex digest of the canonical-JSON-serialised input features."""
    input_data_source: str | None = None
    """Free-form identifier of the data source (e.g., 'yfinance:2024-04-28')."""
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    """Ordered names of the input features."""
    feature_values_summary: dict[str, float] | None = None
    """Optional summary stats (mean/std) for the feature batch; None if disabled."""


@dataclass(frozen=True)
class ModelState:
    """Identifies the model and its parameters."""
    name: str
    """Human-readable model name (e.g., 'GJR-GARCH')."""
    version: str
    """Semantic version of the model (e.g., '1.0.0')."""
    parameters: dict[str, Any] = field(default_factory=dict)
    """Full hyperparameter set."""
    model_artifact_hash: str | None = None
    """SHA-256 of the serialised model artifact, if available."""


@dataclass(frozen=True)
class ComputeState:
    """Identifies the compute environment."""
    python_version: str = ""
    platform: str = ""
    process_id: int | None = None


@dataclass(frozen=True)
class Prediction:
    """The prediction itself."""
    value: float | list[float]
    """Predicted value (scalar or vector)."""
    horizon_days: int | None = None
    """Forecast horizon in days, if applicable."""
    predicted_for_date: str | None = None
    """ISO 8601 date the prediction is for."""


@dataclass(frozen=True)
class AuditRecord:
    """One audit-trail entry. Immutable; identified by record_id (its own hash).

    The serialisation order is fixed by the dataclass field order so that the
    canonical JSON used for hashing is deterministic.
    """
    schema_version: str
    timestamp_utc: str
    """ISO 8601 timestamp at UTC, e.g., '2026-04-28T15:23:01.123456+00:00'."""

    code: CodeState
    data: DataState
    model: ModelState
    compute: ComputeState
    prediction: Prediction
    random_seed: int | None = None
    prev_record_hash: str | None = None
    """SHA-256 hex of the previous record's record_id, or None for genesis."""

    record_id: str = ""
    """SHA-256 hex of canonical-JSON of all fields except record_id itself.
    Set by the hash chain at write time; do not set manually.
    """

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def to_canonical_dict(self) -> dict:
        """Return the dict used for hashing (excludes record_id)."""
        d = asdict(self)
        d.pop("record_id", None)
        return d

    def to_json(self, *, include_record_id: bool = True) -> str:
        """Canonical JSON serialisation: sorted keys, no extra whitespace.

        This is the form whose SHA-256 is the record_id.
        """
        d = asdict(self)
        if not include_record_id:
            d.pop("record_id", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(obj: Any) -> Any:
    """Fallback JSON encoder for types not handled by default."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "__fspath__"):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
