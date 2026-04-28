"""Input-distribution drift detection.

Two methods, both well-established in the model-risk literature:

  - KS test (Kolmogorov-Smirnov, two-sample): null hypothesis that two
    distributions are identical. Returns p-value; small p-value indicates
    drift. Cited extensively in SR 11-7 ongoing-monitoring practice.

  - PSI (Population Stability Index): scalar measure of distribution shift.
    PSI < 0.1 = stable; 0.1 ≤ PSI < 0.25 = moderate shift; PSI ≥ 0.25 = high
    shift. The thresholds are heuristic but widely adopted in credit-risk and
    bank model-risk teams.

The module operates on AuditRecord sequences. Given a window of records
(e.g., a calendar month), it returns drift statistics for each feature whose
summary is captured in DataState.feature_values_summary.

Design choice: we report drift on numeric feature SUMMARIES (means by
default), not on raw feature values, because mr-audit stores summaries by
default for log-size and privacy reasons. This is a conservative choice
that may miss tail-distribution shifts while capturing mean shifts. A
production deployment would want raw feature drift on a sample.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..core.schema import AuditRecord


@dataclass(frozen=True)
class DriftResult:
    feature: str
    n_baseline: int
    n_window: int
    ks_statistic: float
    ks_p_value: float
    psi: float
    psi_band: str
    """One of 'stable', 'moderate', 'high'."""


def _feature_series(records: list[AuditRecord], feature: str) -> np.ndarray:
    vals: list[float] = []
    for r in records:
        if r.data.feature_values_summary is None:
            continue
        if feature in r.data.feature_values_summary:
            v = r.data.feature_values_summary[feature]
            if isinstance(v, (int, float)) and not np.isnan(v):
                vals.append(float(v))
    return np.asarray(vals)


def _psi(baseline: np.ndarray, window: np.ndarray, n_buckets: int = 10) -> float:
    """Compute PSI between two samples using equal-frequency bucketing on baseline.

    PSI = sum_i (p_window_i - p_baseline_i) * log(p_window_i / p_baseline_i)

    Empty buckets are handled with Laplace (add-half) smoothing to avoid
    blow-ups from log(0). This is the standard credit-risk implementation:
    add 0.5 to each bucket count before normalising. The extra ~5% of mass
    bias is dwarfed by genuine drift signals.
    """
    if len(baseline) < n_buckets or len(window) == 0:
        return float("nan")
    # Equal-frequency bucketing on baseline
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    if len(edges) < 2:
        return float("nan")
    edges[0] = -np.inf
    edges[-1] = np.inf

    base_counts, _ = np.histogram(baseline, bins=edges)
    win_counts, _ = np.histogram(window, bins=edges)

    # Laplace (add-half) smoothing: prevents log(0) without distorting PSI for
    # well-populated buckets. With ~10 samples per baseline bucket, +0.5 is ~5%
    # bias; with empty buckets, +0.5 keeps the contribution finite.
    n_b = len(base_counts)
    p_base = (base_counts + 0.5) / (base_counts.sum() + 0.5 * n_b)
    p_win = (win_counts + 0.5) / (win_counts.sum() + 0.5 * n_b)

    return float(np.sum((p_win - p_base) * np.log(p_win / p_base)))


def _psi_band(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "high"


def detect_drift(
    baseline_records: list[AuditRecord],
    window_records: list[AuditRecord],
    *,
    features: list[str] | None = None,
) -> list[DriftResult]:
    """Detect distribution drift between baseline and window for each feature.

    Args:
        baseline_records: reference records (e.g., training period or first month).
        window_records:   records to test (e.g., recent month).
        features:         feature names to check. If None, the union of
                          features present in either set is used.

    Returns:
        One DriftResult per feature.
    """
    if features is None:
        feats: set[str] = set()
        for r in baseline_records + window_records:
            if r.data.feature_values_summary:
                feats.update(r.data.feature_values_summary.keys())
        features = sorted(feats)

    results: list[DriftResult] = []
    for f in features:
        b = _feature_series(baseline_records, f)
        w = _feature_series(window_records, f)
        if len(b) == 0 or len(w) == 0:
            results.append(DriftResult(
                feature=f, n_baseline=len(b), n_window=len(w),
                ks_statistic=float("nan"), ks_p_value=float("nan"),
                psi=float("nan"), psi_band="n/a",
            ))
            continue
        ks_stat, ks_p = stats.ks_2samp(b, w)
        psi = _psi(b, w)
        results.append(DriftResult(
            feature=f, n_baseline=len(b), n_window=len(w),
            ks_statistic=float(ks_stat), ks_p_value=float(ks_p),
            psi=psi, psi_band=_psi_band(psi),
        ))
    return results


def split_by_calendar_window(
    records: list[AuditRecord],
    window: str = "month",
) -> dict[str, list[AuditRecord]]:
    """Group records into calendar windows for time-series drift analysis.

    Args:
        records: chronologically-ordered records (any order tolerated).
        window:  'day' | 'week' | 'month' | 'year'.

    Returns:
        Dict from window-key (e.g., '2026-04') to list of records.
    """
    groups: dict[str, list[AuditRecord]] = defaultdict(list)
    for r in records:
        # timestamp_utc is ISO 8601, e.g., '2026-04-28T15:23:01+00:00'
        key = _window_key(r.timestamp_utc, window)
        if key is not None:
            groups[key].append(r)
    return dict(sorted(groups.items()))


def _window_key(ts_iso: str, window: str) -> str | None:
    """Extract a window-key string from an ISO timestamp."""
    if not ts_iso or "T" not in ts_iso:
        return None
    date_part = ts_iso.split("T")[0]  # e.g. '2026-04-28'
    parts = date_part.split("-")
    if len(parts) != 3:
        return None
    y, m, d = parts
    if window == "day":
        return f"{y}-{m}-{d}"
    if window == "week":
        # ISO week from datetime
        from datetime import date
        dt = date(int(y), int(m), int(d))
        iso_year, iso_week, _ = dt.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    if window == "month":
        return f"{y}-{m}"
    if window == "year":
        return y
    raise ValueError(f"Unknown window: {window}")
