"""Drift-detection module.

Two complementary surfaces:

- ``input.detect_drift`` and ``DriftResult``: covariate-shift detection
  on feature values recorded in DataState.feature_values_summary.

- ``output.detect_output_drift`` and ``OutputDriftResult``: concept-drift
  detection on Prediction.value across windows.
"""
from mr_audit.drift.input import (
    DriftResult,
    detect_drift,
    split_by_calendar_window,
)
from mr_audit.drift.output import (
    OutputDriftResult,
    detect_output_drift,
)

__all__ = [
    "DriftResult",
    "detect_drift",
    "split_by_calendar_window",
    "OutputDriftResult",
    "detect_output_drift",
]
