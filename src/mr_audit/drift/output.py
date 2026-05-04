"""Output-distribution drift detection.

The companion to ``input.py``. Where input drift asks ``has the
distribution of inputs my model sees changed?'', output drift asks
``has the distribution of predictions my model produces changed?''
The two questions are complementary:

  - Input drift catches covariate shift before the model has had a
    chance to misbehave on it.
  - Output drift catches concept drift, label drift, and silent
    failures of the model whose prediction distribution has shifted
    even though inputs look stable.

The same statistical machinery (Kolmogorov-Smirnov two-sample test;
Population Stability Index with Laplace smoothing) applies, but the
input is now ``Prediction.value'' across records rather than a
DataState feature summary.

Both methods are well-established in model-risk practice for output
monitoring; SR 11-7 §IV.C (Ongoing Monitoring) and EU AI Act Article
12 both reference monitoring of model output behaviour over time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..core.schema import AuditRecord


@dataclass(frozen=True)
class OutputDriftResult:
    """Result of output-distribution drift detection.

    Attributes:
        n_baseline:    Number of baseline-window records with a numeric
                       prediction value.
        n_window:      Number of test-window records with a numeric
                       prediction value.
        ks_statistic:  KS two-sample statistic on prediction values.
        ks_p_value:    KS two-sample p-value.
        psi:           Population Stability Index of window vs baseline.
        psi_band:      One of 'stable', 'moderate', 'high', 'n/a'.
        baseline_mean: Mean of baseline predictions.
        window_mean:   Mean of window predictions.
        baseline_std:  Std of baseline predictions.
        window_std:    Std of window predictions.
    """

    n_baseline: int
    n_window: int
    ks_statistic: float
    ks_p_value: float
    psi: float
    psi_band: str
    baseline_mean: float
    window_mean: float
    baseline_std: float
    window_std: float


def _scalar_predictions(records: list[AuditRecord]) -> np.ndarray:
    """Extract scalar Prediction.value from a list of records.

    Vector predictions are reduced to their first component for
    distribution comparison purposes; callers needing per-component
    drift should slice the records themselves and call this once per
    component.
    """
    vals: list[float] = []
    for r in records:
        v = r.prediction.value
        if isinstance(v, (int, float)):
            if not np.isnan(float(v)):
                vals.append(float(v))
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            first = v[0]
            if isinstance(first, (int, float)) and not np.isnan(float(first)):
                vals.append(float(first))
    return np.asarray(vals, dtype=np.float64)


def _psi(baseline: np.ndarray, window: np.ndarray, n_buckets: int = 10) -> float:
    """PSI with equal-frequency bucketing on baseline and Laplace smoothing.

    Identical implementation to drift.input._psi; duplicated here so
    the output module has no cross-module private dependency.
    """
    if len(baseline) < n_buckets or len(window) == 0:
        return float("nan")
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    if len(edges) < 2:
        return float("nan")
    edges[0] = -np.inf
    edges[-1] = np.inf

    base_counts, _ = np.histogram(baseline, bins=edges)
    win_counts, _ = np.histogram(window, bins=edges)

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


def detect_output_drift(
    baseline_records: list[AuditRecord],
    window_records: list[AuditRecord],
) -> OutputDriftResult:
    """Detect prediction-distribution drift between two record windows.

    Args:
        baseline_records: reference window (e.g., training-period predictions
            or the first calendar month of production).
        window_records:   the window to test (e.g., the most recent month).

    Returns:
        OutputDriftResult with KS test, PSI, and summary statistics.
    """
    b = _scalar_predictions(baseline_records)
    w = _scalar_predictions(window_records)

    if len(b) == 0 or len(w) == 0:
        return OutputDriftResult(
            n_baseline=len(b), n_window=len(w),
            ks_statistic=float("nan"), ks_p_value=float("nan"),
            psi=float("nan"), psi_band="n/a",
            baseline_mean=float("nan"), window_mean=float("nan"),
            baseline_std=float("nan"), window_std=float("nan"),
        )

    ks_stat, ks_p = stats.ks_2samp(b, w)
    psi = _psi(b, w)

    return OutputDriftResult(
        n_baseline=int(len(b)),
        n_window=int(len(w)),
        ks_statistic=float(ks_stat),
        ks_p_value=float(ks_p),
        psi=float(psi),
        psi_band=_psi_band(psi),
        baseline_mean=float(b.mean()),
        window_mean=float(w.mean()),
        baseline_std=float(b.std(ddof=1)) if len(b) > 1 else 0.0,
        window_std=float(w.std(ddof=1)) if len(w) > 1 else 0.0,
    )
