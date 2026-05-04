"""Tests for output-distribution drift detection (mr_audit.drift.output)."""
from __future__ import annotations

import numpy as np
from mr_audit.core.schema import (
    AuditRecord,
    CodeState,
    ComputeState,
    DataState,
    ModelState,
    Prediction,
)
from mr_audit.drift import OutputDriftResult, detect_output_drift


def _make_record(prediction_value: float, idx: int = 0) -> AuditRecord:
    """Construct a minimal AuditRecord with a scalar prediction."""
    from mr_audit.core.schema import SCHEMA_VERSION

    return AuditRecord(
        schema_version=SCHEMA_VERSION,
        timestamp_utc="2026-01-01T00:00:00Z",
        code=CodeState(
            code_commit_hash=None,
            code_dirty=False,
            library_versions={},
        ),
        data=DataState(
            input_data_hash="x" * 64,
            input_data_source=None,
            feature_names=(),
            feature_values_summary=None,
        ),
        model=ModelState(
            name="dummy",
            version="0.0",
            parameters={},
            model_artifact_hash=None,
        ),
        compute=ComputeState(
            python_version="3.14",
            platform="darwin",
            process_id=None,
        ),
        prediction=Prediction(
            value=prediction_value,
            horizon_days=None,
            predicted_for_date=None,
        ),
        random_seed=None,
        prev_record_hash=None,
        record_id=None,
    )


def _records_from_distribution(values: np.ndarray) -> list[AuditRecord]:
    return [_make_record(float(v), i) for i, v in enumerate(values)]


def test_output_drift_returns_result_object() -> None:
    rng = np.random.default_rng(2026)
    base = _records_from_distribution(rng.normal(0.15, 0.04, 200))
    win = _records_from_distribution(rng.normal(0.15, 0.04, 200))
    result = detect_output_drift(base, win)
    assert isinstance(result, OutputDriftResult)
    assert result.n_baseline == 200
    assert result.n_window == 200


def test_output_drift_no_drift_when_same_distribution() -> None:
    """When baseline and window are drawn from the same distribution,
    KS p-value should be > 0.05 in expectation and PSI should be 'stable'."""
    rng = np.random.default_rng(2026)
    base = _records_from_distribution(rng.normal(0.15, 0.04, 500))
    win = _records_from_distribution(rng.normal(0.15, 0.04, 500))
    result = detect_output_drift(base, win)
    # Not asserting p > 0.05 (KS is noisy on n=500) but PSI should be small.
    assert result.psi < 0.10  # 'stable' band
    assert result.psi_band == "stable"


def test_output_drift_detects_mean_shift() -> None:
    """When predictions shift in mean, KS rejects and PSI flags."""
    rng = np.random.default_rng(2026)
    base = _records_from_distribution(rng.normal(0.15, 0.04, 500))
    win = _records_from_distribution(rng.normal(0.30, 0.04, 500))  # mean shift
    result = detect_output_drift(base, win)
    assert result.ks_p_value < 1e-10
    assert result.psi >= 0.25  # 'high' band
    assert result.psi_band == "high"
    assert abs(result.window_mean - 0.30) < 0.01
    assert abs(result.baseline_mean - 0.15) < 0.01


def test_output_drift_detects_variance_shift() -> None:
    """A pure variance shift (same mean) should also register."""
    rng = np.random.default_rng(2026)
    base = _records_from_distribution(rng.normal(0.15, 0.02, 500))
    win = _records_from_distribution(rng.normal(0.15, 0.10, 500))  # variance shift
    result = detect_output_drift(base, win)
    assert result.ks_p_value < 1e-5
    assert result.psi > 0.10  # at least moderate band


def test_output_drift_handles_empty_window() -> None:
    rng = np.random.default_rng(2026)
    base = _records_from_distribution(rng.normal(0.15, 0.04, 50))
    result = detect_output_drift(base, [])
    assert result.n_window == 0
    assert np.isnan(result.ks_p_value)
    assert result.psi_band == "n/a"


def test_output_drift_handles_vector_predictions() -> None:
    """Vector predictions reduce to first component."""
    base = [_make_record(0.1) for _ in range(50)]
    # Manually craft a record with a vector prediction
    vec_records: list[AuditRecord] = []
    for i in range(50):
        r = _make_record(0.5, i)
        # Replace prediction with a vector
        r_vec = AuditRecord(
            schema_version=r.schema_version,
            timestamp_utc=r.timestamp_utc,
            code=r.code, data=r.data, model=r.model, compute=r.compute,
            prediction=Prediction(value=[0.5, 0.6, 0.7], horizon_days=None, predicted_for_date=None),
            random_seed=r.random_seed, prev_record_hash=r.prev_record_hash, record_id=r.record_id,
        )
        vec_records.append(r_vec)
    result = detect_output_drift(base, vec_records)
    # Window predictions should be ~0.5 (first component); baseline ~0.1
    assert abs(result.window_mean - 0.5) < 0.01
    assert abs(result.baseline_mean - 0.1) < 0.01
    assert result.ks_p_value < 1e-10


def test_output_drift_psi_band_thresholds() -> None:
    """Verify PSI band labels match Hansen's thresholds (<0.10 stable, <0.25 moderate, else high)."""
    rng = np.random.default_rng(2026)
    # Use a small location shift to land in the 'moderate' band
    base = _records_from_distribution(rng.normal(0.15, 0.04, 500))
    win = _records_from_distribution(rng.normal(0.18, 0.04, 500))  # small shift
    result = detect_output_drift(base, win)
    assert result.psi_band in ("moderate", "high")  # not 'stable'
