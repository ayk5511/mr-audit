"""User-facing API: AuditContext + @auditable decorator.

Typical usage:

    from mr_audit import AuditContext, auditable

    @auditable(model_name="GJR-GARCH", model_version="1.0.0")
    def predict_volatility(features, params):
        return forecast

    with AuditContext(log_path="audit/run.db"):
        forecast = predict_volatility(features, params)

The context manager establishes the active store; the decorator pulls the
calling-code state and writes one record per invocation.

Design choice: the calling code uses kwargs for model identification rather
than reflecting on the function. Reflection-based introspection is fragile
across decorators and async wrappers; the explicit kwargs are honest and
auditable.
"""
from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .schema import (
    SCHEMA_VERSION,
    AuditRecord,
    CodeState,
    ComputeState,
    DataState,
    ModelState,
    Prediction,
)
from .store import AuditStore, JSONLStore, SQLiteStore

# Active store, scoped via contextvars (works in async + threaded contexts).
_active_store: contextvars.ContextVar[AuditStore | None] = contextvars.ContextVar(
    "_mr_audit_active_store", default=None
)


class AuditContext:
    """Context manager establishing the active audit store.

    Auto-detects backend from path extension:
      .db, .sqlite, .sqlite3 -> SQLiteStore
      .jsonl, .ndjson        -> JSONLStore

    Multiple contexts cannot be nested concurrently in the same thread; the
    inner-most context wins.
    """

    def __init__(
        self,
        log_path: str | Path,
        *,
        backend: AuditStore | None = None,
    ) -> None:
        self.log_path = Path(log_path)
        if backend is not None:
            self.store: AuditStore = backend
        else:
            self.store = self._auto_backend(self.log_path)
        self._token: Any = None

    @staticmethod
    def _auto_backend(path: Path) -> AuditStore:
        ext = path.suffix.lower()
        if ext in (".db", ".sqlite", ".sqlite3"):
            return SQLiteStore(path)
        if ext in (".jsonl", ".ndjson"):
            return JSONLStore(path)
        # Default: SQLite for any other extension
        return SQLiteStore(path)

    def __enter__(self) -> "AuditContext":
        self._token = _active_store.set(self.store)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _active_store.reset(self._token)
            self._token = None
        if hasattr(self.store, "close"):
            self.store.close()


def auditable(
    *,
    model_name: str,
    model_version: str = "0.0.0",
    horizon_days: int | None = None,
    feature_extractor: Callable[..., dict[str, float]] | None = None,
    summarise_features: bool = True,
) -> Callable:
    """Decorator that records an audit entry on every call.

    Args:
        model_name: human-readable model name, e.g. "GJR-GARCH".
        model_version: semantic version, e.g. "1.0.0".
        horizon_days: forecast horizon, recorded with the prediction.
        feature_extractor: optional callable taking the same args as the
            decorated function and returning a dict of {name: value}. Used to
            populate DataState.feature_values_summary. If None, the args are
            JSON-canonicalised and only their hash is stored (no plaintext).
        summarise_features: if True and feature_extractor is provided, store
            mean/std summaries; if False, store full feature_values_summary.
            Defaults to True for privacy and log-size reasons.

    The decorated function is called normally; the audit record is written to
    the active store (set by AuditContext). If no active store, the decorator
    is a no-op (the function runs but no log is written).
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            store = _active_store.get()
            # Private kwargs (prefix _) are mr-audit metadata; never passed to fn
            private_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
            public_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}

            if store is None:
                # No active context; call through unrecorded (without private kwargs)
                return fn(*args, **public_kwargs)

            # 1. Compute input data hash (uses public kwargs only)
            data_hash, feature_summary = _hash_inputs(
                args, public_kwargs, feature_extractor, summarise_features
            )

            # 2. Capture code state
            code_state = _capture_code_state()

            # 3. Capture compute state
            compute_state = _capture_compute_state()

            # 4. Run the function (without private kwargs; they are mr-audit-only)
            prediction_value = fn(*args, **public_kwargs)

            # 5. Build the record
            record = AuditRecord(
                schema_version=SCHEMA_VERSION,
                timestamp_utc=AuditRecord.now_iso(),
                code=code_state,
                data=DataState(
                    input_data_hash=data_hash,
                    input_data_source=private_kwargs.get("_data_source"),
                    feature_names=tuple(feature_summary.keys()) if feature_summary else (),
                    feature_values_summary=feature_summary,
                ),
                model=ModelState(
                    name=model_name,
                    version=model_version,
                    parameters=_extract_params(public_kwargs),
                ),
                compute=compute_state,
                prediction=Prediction(
                    value=_coerce_prediction(prediction_value),
                    horizon_days=horizon_days,
                    predicted_for_date=private_kwargs.get("_predicted_for_date"),
                ),
                random_seed=private_kwargs.get("_random_seed"),
            )

            # 6. Persist (store seals + chains)
            store.append(record)
            return prediction_value

        return wrapper

    return decorator


# -- helpers ----------------------------------------------------------------


def _capture_code_state() -> CodeState:
    """Capture git commit hash and key library versions from the calling env."""
    commit = _git_commit_hash()
    dirty = _git_dirty() if commit else False
    libs = _library_versions()
    return CodeState(
        code_commit_hash=commit,
        code_dirty=dirty,
        library_versions=libs,
    )


def _git_commit_hash() -> str | None:
    """Return the git commit SHA of the current working directory's repo, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _git_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def _library_versions() -> dict[str, str]:
    """Capture versions of the key ML/numerical libraries currently imported."""
    libs: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "torch", "sklearn", "arch", "statsmodels"):
        if name in sys.modules:
            mod = sys.modules[name]
            libs[name] = getattr(mod, "__version__", "unknown")
    return libs


def _capture_compute_state() -> ComputeState:
    return ComputeState(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        process_id=os.getpid(),
    )


def _hash_inputs(
    args: tuple,
    kwargs: dict,
    feature_extractor: Callable | None,
    summarise: bool,
) -> tuple[str, dict[str, float] | None]:
    """Compute SHA-256 of canonical-JSON of inputs; optionally extract summary.

    Caller is expected to pass public kwargs only (no _-prefixed metadata).
    """
    hash_input = {
        "args": [_to_jsonable(a) for a in args],
        "kwargs": {k: _to_jsonable(v) for k, v in kwargs.items()},
    }
    canonical = json.dumps(
        hash_input, sort_keys=True, separators=(",", ":"), default=_to_jsonable_default
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    summary: dict[str, float] | None = None
    if feature_extractor is not None:
        try:
            features = feature_extractor(*args, **kwargs)
            if summarise:
                # Reduce to {name: value} for summary; numeric only
                summary = {k: float(v) for k, v in features.items() if isinstance(v, (int, float))}
            else:
                summary = features
        except Exception:
            summary = None
    return digest, summary


def _extract_params(kwargs: dict) -> dict[str, Any]:
    """Extract hyperparameters from kwargs (anything not prefixed with _)."""
    return {k: _to_jsonable(v) for k, v in kwargs.items() if not k.startswith("_")}


def _to_jsonable(obj: Any) -> Any:
    """Coerce common types to JSON-serialisable forms."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return repr(obj)


def _to_jsonable_default(obj: Any) -> Any:
    """JSON encoder fallback for arbitrary objects."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return repr(obj)


def _coerce_prediction(value: Any) -> float | list[float]:
    """Coerce prediction to a JSON-serialisable scalar or list."""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise TypeError(
        f"Prediction must be numeric scalar or array-like; got {type(value).__name__}"
    )
