"""Drift-detection module."""
from mr_audit.drift.input import (
    DriftResult,
    detect_drift,
    split_by_calendar_window,
)

__all__ = ["DriftResult", "detect_drift", "split_by_calendar_window"]
