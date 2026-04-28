"""Tests for drift detection (KS + PSI)."""
from __future__ import annotations

import numpy as np

from mr_audit.core.schema import (
    SCHEMA_VERSION,
    AuditRecord,
    CodeState,
    ComputeState,
    DataState,
    ModelState,
    Prediction,
)
from mr_audit.drift import detect_drift, split_by_calendar_window
from mr_audit.drift.input import _psi, _psi_band


def make_record_with_features(features: dict[str, float], ts: str) -> AuditRecord:
    return AuditRecord(
        schema_version=SCHEMA_VERSION,
        timestamp_utc=ts,
        code=CodeState(),
        data=DataState(
            input_data_hash="h",
            feature_names=tuple(features.keys()),
            feature_values_summary=features,
        ),
        model=ModelState(name="M", version="1.0.0"),
        compute=ComputeState(),
        prediction=Prediction(value=1.0),
    )


def test_no_drift_when_distributions_match():
    """At n=2000, sampling noise alone keeps PSI in the stable/moderate band.

    Note: PSI thresholds (0.10/0.25) are calibrated for credit-risk samples
    with n>1000; at small samples (n=100), bucket-count variance alone
    produces PSI ~ 0.1-0.4. The KS test is more reliable at small n.
    """
    rng = np.random.default_rng(42)
    n = 2000
    baseline = [
        make_record_with_features(
            {"x": float(rng.normal(0, 1))},
            f"2026-01-{(i % 28)+1:02d}T12:00:00+00:00",
        )
        for i in range(n)
    ]
    window = [
        make_record_with_features(
            {"x": float(rng.normal(0, 1))},
            f"2026-02-{(i % 28)+1:02d}T12:00:00+00:00",
        )
        for i in range(n)
    ]
    results = detect_drift(baseline, window, features=["x"])
    assert len(results) == 1
    r = results[0]
    assert r.feature == "x"
    assert r.n_baseline == n
    assert r.n_window == n
    # Same generative distribution -> KS should NOT reject; p_value > 0.05
    assert r.ks_p_value > 0.05
    # PSI in stable band at this sample size
    assert r.psi_band in ("stable", "moderate")


def test_clear_drift_detected():
    rng = np.random.default_rng(42)
    baseline = [
        make_record_with_features({"x": float(rng.normal(0, 1))}, f"2026-01-{(i % 28)+1:02d}T12:00:00+00:00")
        for i in range(200)
    ]
    # Window distribution is shifted by +3 standard deviations
    window = [
        make_record_with_features({"x": float(rng.normal(3, 1))}, f"2026-02-{(i % 28)+1:02d}T12:00:00+00:00")
        for i in range(200)
    ]
    results = detect_drift(baseline, window, features=["x"])
    r = results[0]
    # KS p-value should be very small (clear rejection)
    assert r.ks_p_value < 1e-6
    # PSI should be in 'high' band
    assert r.psi_band == "high"


def test_psi_bands():
    assert _psi_band(0.05) == "stable"
    assert _psi_band(0.15) == "moderate"
    assert _psi_band(0.30) == "high"
    assert _psi_band(float("nan")) == "n/a"


def test_psi_zero_for_identical_samples():
    """PSI between identical samples should be very close to zero."""
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 500)
    psi = _psi(sample, sample)
    assert abs(psi) < 1e-3


def test_split_by_month():
    records = [
        make_record_with_features({"x": 1.0}, "2026-01-15T12:00:00+00:00"),
        make_record_with_features({"x": 1.0}, "2026-01-28T12:00:00+00:00"),
        make_record_with_features({"x": 1.0}, "2026-02-05T12:00:00+00:00"),
        make_record_with_features({"x": 1.0}, "2026-02-20T12:00:00+00:00"),
        make_record_with_features({"x": 1.0}, "2026-03-01T12:00:00+00:00"),
    ]
    groups = split_by_calendar_window(records, window="month")
    assert list(groups.keys()) == ["2026-01", "2026-02", "2026-03"]
    assert len(groups["2026-01"]) == 2
    assert len(groups["2026-02"]) == 2
    assert len(groups["2026-03"]) == 1


def test_split_by_year():
    records = [
        make_record_with_features({"x": 1.0}, "2025-06-15T12:00:00+00:00"),
        make_record_with_features({"x": 1.0}, "2026-01-15T12:00:00+00:00"),
        make_record_with_features({"x": 1.0}, "2026-12-31T12:00:00+00:00"),
    ]
    groups = split_by_calendar_window(records, window="year")
    assert list(groups.keys()) == ["2025", "2026"]


def test_drift_handles_empty_baseline():
    """If baseline is empty, drift result should still be returned with NaN stats."""
    window = [make_record_with_features({"x": 1.0}, "2026-01-01T12:00:00+00:00")]
    results = detect_drift([], window, features=["x"])
    assert len(results) == 1
    r = results[0]
    assert r.n_baseline == 0
    assert np.isnan(r.ks_p_value)
    assert r.psi_band == "n/a"


def test_drift_auto_detects_features():
    """If features=None, all features are tested."""
    baseline = [make_record_with_features({"a": 0.0, "b": 1.0}, "2026-01-01T12:00:00+00:00")]
    window = [make_record_with_features({"a": 0.0, "b": 1.0}, "2026-02-01T12:00:00+00:00")]
    results = detect_drift(baseline, window, features=None)
    feats = sorted(r.feature for r in results)
    assert feats == ["a", "b"]
