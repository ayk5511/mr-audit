"""mr-audit: An audit-trail framework for ML pipelines under SR 11-7
and the EU AI Act.
"""
from mr_audit.core.auditable import AuditContext, auditable
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
from mr_audit.core.store import AuditStore, JSONLStore, SQLiteStore

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    # API
    "AuditContext",
    "auditable",
    # Schema types
    "AuditRecord",
    "CodeState",
    "ComputeState",
    "DataState",
    "ModelState",
    "Prediction",
    # Storage
    "AuditStore",
    "SQLiteStore",
    "JSONLStore",
    # Hash chain
    "compute_record_id",
    "seal_record",
    "verify_chain",
]
