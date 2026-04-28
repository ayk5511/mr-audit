"""Replay: rebuild any prediction from the audit log.

The replay contract:
  Given a sealed AuditRecord, plus the deterministic, idempotent prediction
  function `predict_fn(*args, **kwargs)`, calling predict_fn with the inputs
  recorded must produce the same prediction value the record stores.

This is the primary "regulator inspection" capability: any cell of any table
in any downstream report must be reconstructable from the log alone.

Important caveats:
  1. The prediction function must be deterministic conditional on its inputs
     (including random_seed). Stochastic models without seed control cannot
     be replayed bit-identically.
  2. The library versions and platform must match (or be compatible). The
     audit record captures these so a mismatch is detectable.
  3. The input data must be reconstructable. mr-audit stores the input HASH,
     not the input values themselves (to keep logs compact and to avoid
     recording potentially sensitive data). The replayer must therefore
     supply the input data; replay verifies the hash matches.

The verify_replay function returns a ReplayResult with all comparison detail.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

from ..core.schema import AuditRecord


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of replaying one record."""
    record_id: str
    input_hash_match: bool
    """True if the inputs supplied for replay hash to the same value as logged."""
    prediction_match: bool
    """True if the replayed prediction matches the logged prediction within tol."""
    logged_value: Any
    replayed_value: Any
    library_version_warnings: list[str]
    """Non-fatal: libraries whose versions differ from the logged record."""
    notes: list[str]


def replay(
    record: AuditRecord,
    predict_fn: Callable,
    inputs: dict[str, Any] | None = None,
    *,
    args: tuple = (),
    kwargs: dict | None = None,
    tol: float = 1e-9,
) -> ReplayResult:
    """Replay one prediction from an audit record.

    Args:
        record:        the AuditRecord to replay
        predict_fn:    the prediction function (NOT decorated with @auditable;
                       use the unwrapped version)
        inputs:        deprecated convenience kwarg; same effect as kwargs=inputs
        args, kwargs:  positional / keyword args for predict_fn. Their hash must
                       match the record's data.input_data_hash for replay to
                       be considered valid.
        tol:           absolute tolerance for prediction equality

    Returns:
        ReplayResult with input_hash_match, prediction_match, value comparison,
        and any library-version warnings.

    The function does NOT modify the record. It is read-only.
    """
    if kwargs is None:
        kwargs = inputs.copy() if inputs else {}
    if inputs is not None and not kwargs:
        kwargs = inputs.copy()

    notes: list[str] = []

    # 1. Verify input hash
    public_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
    canonical = json.dumps(
        {"args": [_to_jsonable(a) for a in args],
         "kwargs": {k: _to_jsonable(v) for k, v in public_kwargs.items()}},
        sort_keys=True, separators=(",", ":"), default=_to_jsonable,
    )
    actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    input_match = actual_hash == record.data.input_data_hash
    if not input_match:
        notes.append(
            f"Input hash mismatch: replay produced {actual_hash[:16]}..., "
            f"record contains {record.data.input_data_hash[:16]}..."
        )

    # 2. Library-version comparison
    lib_warnings: list[str] = []
    try:
        import sys
        for libname, logged_ver in record.code.library_versions.items():
            mod = sys.modules.get(libname)
            if mod is None:
                lib_warnings.append(
                    f"library '{libname}' not loaded at replay time (logged: {logged_ver})"
                )
                continue
            current_ver = getattr(mod, "__version__", "unknown")
            if current_ver != logged_ver:
                lib_warnings.append(
                    f"library '{libname}' version differs: replay={current_ver}, logged={logged_ver}"
                )
    except Exception as e:
        lib_warnings.append(f"library-version check failed: {e}")

    # 3. Re-run prediction
    replayed_value = predict_fn(*args, **kwargs)

    # 4. Compare predictions
    pred_match = _values_match(record.prediction.value, replayed_value, tol=tol)
    if not pred_match:
        notes.append(
            f"Prediction mismatch: logged={record.prediction.value}, replayed={replayed_value}"
        )

    return ReplayResult(
        record_id=record.record_id,
        input_hash_match=input_match,
        prediction_match=pred_match,
        logged_value=record.prediction.value,
        replayed_value=_to_jsonable(replayed_value),
        library_version_warnings=lib_warnings,
        notes=notes,
    )


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return repr(obj)


def _values_match(a: Any, b: Any, *, tol: float) -> bool:
    """Approximate equality for predictions (scalars or lists)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), abs_tol=tol)
    if hasattr(b, "tolist"):
        b = b.tolist()
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(
            math.isclose(float(x), float(y), abs_tol=tol)
            for x, y in zip(a, b)
        )
    return a == b
