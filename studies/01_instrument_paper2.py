"""Empirical demonstration: instrument the Paper 2 vol forecaster with mr-audit.

This script demonstrates retrospective instrumentation. We do not re-run
Paper 2's forecasting pipeline; we take its committed forecast panel
(forecasts_5d.parquet, 980 days x 7 models = 6860 forecast cells) and write
each cell as an mr-audit record. The result is a realistic-scale audit log
that can be used to demonstrate the four use cases:

  1. Reproducibility audit (replay any forecast)
  2. Drift detection (monthly KS + PSI on input features)
  3. Model risk validation report (per-model statistics)
  4. Regulator export bundle (zip with manifest + chain verification)

The "input features" we record per cell are the lagged actual realized
volatility values [actual_t-1, actual_t-2, ..., actual_t-5] in their
day-ordered position. These are the most natural "inputs" for a 5-day-ahead
forecast and let us run meaningful drift analysis over the test period
(2022-01-03 to 2025-11-26, including the high-vol 2022 regime and the
lower-vol 2023+ regime documented in Khan2026Vol).

Output:
  studies/audit_logs/paper2_demo.db    (SQLite, ~6860 records)
  studies/audit_logs/instrumentation_log.json (run metadata)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from mr_audit import AuditContext, auditable

ROOT = Path(__file__).resolve().parent
PAPER2_FORECASTS = (
    Path("/Users/akmbp/Documents/EB1A-Profile/papers/paper2-volatility/results/forecasts_5d.parquet")
)
LOG_DIR = ROOT / "audit_logs"
LOG_PATH = LOG_DIR / "paper2_demo.db"
META_PATH = LOG_DIR / "instrumentation_log.json"

MODELS = ["GARCH", "EGARCH", "GJR-GARCH", "HAR-RV", "LightGBM", "XGBoost", "Ensemble"]
LOOKBACK_DAYS = 5  # number of lagged realised-vol values used as input features


def feature_extractor(*, lagged_rvs, **_):
    """Return per-feature values of the lagged realised-volatility window.

    Used by mr-audit to populate DataState.feature_values_summary.
    Accepts arbitrary additional kwargs (model_params, etc.) and ignores them.
    """
    return {f"rv_lag_{i+1}": float(v) for i, v in enumerate(lagged_rvs)}


def make_predict_for(model_name: str):
    """Decorator factory: returns one @auditable predictor per model.

    The model name is captured at decoration time so each model has its own
    auditable wrapper that records model.name correctly.
    """

    @auditable(
        model_name=model_name,
        model_version="paper2-v1",
        horizon_days=5,
        feature_extractor=feature_extractor,
        summarise_features=False,
    )
    def predict_volatility(*, lagged_rvs, model_params=None):
        """Mock prediction function: returns the previously-fit forecast value."""
        if model_params is None:
            return 0.0
        return float(model_params.get("_forecast_value", 0.0))

    return predict_volatility


# Build one predictor per model (7 total)
PREDICTORS = {m: make_predict_for(m) for m in MODELS}


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()  # rebuild fresh

    print(f"Loading Paper 2 forecast panel: {PAPER2_FORECASTS}")
    df = pd.read_parquet(PAPER2_FORECASTS)
    print(f"  shape: {df.shape}")
    print(f"  date range: {df.index.min().date()} -> {df.index.max().date()}")

    # Compute lagged realised vol per row (need 5 days of prior actuals)
    actual = df["actual"].values
    n = len(df)

    print(f"\nWriting audit records for {len(MODELS)} models x {n} days...")
    start = time.time()
    n_written = 0
    with AuditContext(LOG_PATH):
        for i in range(LOOKBACK_DAYS, n):
            date = df.index[i]
            lagged = tuple(float(actual[i - k - 1]) for k in range(LOOKBACK_DAYS))
            for model_name in MODELS:
                forecast = float(df.iloc[i][model_name])
                PREDICTORS[model_name](
                    lagged_rvs=lagged,
                    model_params={"_forecast_value": forecast},
                    _data_source="ayk5511/volatility-forecasting:forecasts_5d.parquet",
                    _predicted_for_date=date.isoformat()[:10],
                    _random_seed=2026,
                )
                n_written += 1
            if (i - LOOKBACK_DAYS + 1) % 200 == 0:
                elapsed = time.time() - start
                rate = n_written / elapsed if elapsed > 0 else 0
                print(f"  day {i+1}/{n}  records={n_written:>5}  rate={rate:.0f}/s")

    elapsed = time.time() - start
    log_size_kb = LOG_PATH.stat().st_size // 1024
    print(f"\nDone. {n_written} records in {elapsed:.1f}s ({n_written/elapsed:.0f}/s).")
    print(f"Log size: {log_size_kb} KB")

    meta = {
        "source_panel": str(PAPER2_FORECASTS),
        "n_days_in_panel": int(n),
        "n_days_audited": int(n - LOOKBACK_DAYS),
        "n_models": len(MODELS),
        "models": MODELS,
        "n_records_written": n_written,
        "lookback_days": LOOKBACK_DAYS,
        "audit_log_path": str(LOG_PATH),
        "elapsed_seconds": round(elapsed, 3),
        "write_rate_records_per_second": round(n_written / elapsed, 1) if elapsed > 0 else None,
        "log_size_kb": int(log_size_kb),
        "log_size_bytes_per_record": int(LOG_PATH.stat().st_size / n_written) if n_written else None,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {META_PATH}")


if __name__ == "__main__":
    sys.exit(main() or 0)
