"""mr-audit on a statsmodels SARIMAX macroeconomic forecaster.

Demonstrates mr-audit on time-series econometric models. Same one-decorator
integration; SR 11-7 §VI model-inventory and §V monitoring requirements
apply identically.

Scenario: a central-bank or treasury function forecasting unemployment-rate
direction quarter-ahead from FRED-style synthetic data. Predictions go to
a monetary-policy committee; SR 11-7 §V requires per-prediction logging.

Synthetic data only (no FRED API call required).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

from mr_audit import AuditContext, auditable, verify_chain

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026)


def main():
    # --- Synthetic quarterly macro series, 1990 Q1 to 2025 Q4 (144 quarters) ---
    n = 144
    dates = pd.date_range("1990-03-31", periods=n, freq="QE")
    # AR(1) with seasonal drift to mimic unemployment-rate behaviour
    series = np.zeros(n)
    series[0] = 5.0
    for t in range(1, n):
        season = 0.3 * np.sin(2 * np.pi * t / 4)
        series[t] = 5.0 + 0.85 * (series[t - 1] - 5.0) + season + rng.normal(0, 0.4)
    panel = pd.Series(series, index=dates, name="unemp_pct")

    print(f"Synthetic series: {len(panel)} quarters, {panel.index[0].date()} to {panel.index[-1].date()}")
    print(f"  Mean: {panel.mean():.2f}%, Std: {panel.std():.2f}%")

    # --- ARIMA(1,0,1) on de-seasoned series; SARIMAX added too much complexity for this synthetic ---
    from statsmodels.tsa.arima.model import ARIMA

    # De-season by demeaning each quarterly position
    season = panel.groupby(panel.index.quarter).transform("mean")
    de_seasoned = panel - season + panel.mean()

    train_n = 120
    train, test = de_seasoned.iloc[:train_n], de_seasoned.iloc[train_n:]
    model_spec = {"order": (1, 0, 1)}
    fit = ARIMA(train, **model_spec).fit()
    print(f"\nARIMA{model_spec['order']} trained on {train_n} quarters of de-seasoned series")
    print(f"  AR(1) coef: {fit.params.get('ar.L1', float('nan')):.3f}")
    print(f"  MA(1) coef: {fit.params.get('ma.L1', float('nan')):.3f}")

    @auditable(
        model_name="ARIMA-Unemployment-1Q",
        model_version="1.0.0",
        horizon_days=90,
        feature_extractor=lambda *, history, **_: {f"lag_{i}": float(v) for i, v in enumerate(history[-4:])},
    )
    def predict_next_quarter(*, history, model_params=None, **_):
        """Forecast 1 quarter ahead from the latest history."""
        if model_params is None:
            model_params = {"order": (1, 0, 1)}
        order = tuple(model_params["order"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = ARIMA(np.asarray(history), order=order).fit()
            return float(f.forecast(steps=1)[0])

    # --- Walk-forward audited forecasts on the test set ---
    log_path = "examples/audit_logs/sarimax_macro.db"
    print(f"\nWalk-forward forecasting {len(test)} quarters with audit logging...")
    with AuditContext(log_path) as ctx:
        for i, (date, _) in enumerate(test.items()):
            history_for_predict = panel.iloc[: train_n + i].tolist()
            predict_next_quarter(
                history=history_for_predict,
                model_params=model_spec,
                _data_source="synthetic_macro_AR1_seasonal:seed=2026",
                _predicted_for_date=date.date().isoformat(),
                _random_seed=2026,
            )
        records = ctx.store.read_all()

    print(f"  Audit records written: {len(records)}")
    valid, errors = verify_chain(records)
    print(f"  Chain valid: {valid}")

    # Sample record
    rec = records[0]
    print("\nExample record [0]:")
    print(f"  model:     {rec.model.name} v{rec.model.version}")
    print(f"  predicted: {rec.prediction.value:.3f}% (for {rec.prediction.predicted_for_date})")
    print(f"  history-tail logged: {rec.data.feature_values_summary}")
    print(f"  ARIMA order in params: {rec.model.parameters['model_params']}")


if __name__ == "__main__":
    main()
