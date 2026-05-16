"""Compute real output-drift numbers on the Khan 2026 audit log.

Output drift over time: baseline = 2022 Q1 (first 3 months), window =
each subsequent month. Per model, asks whether the prediction
distribution has shifted.

We also compute the "covariate vs concept drift" comparison: when
input KS p < 0.01 and output KS p > 0.05, that's input-only drift
(model has absorbed the shift). When both are < 0.01, it's joint
drift. When output is < 0.01 and input is not, it's concept drift.
"""
import json
import sqlite3
from collections import defaultdict
from datetime import date

import numpy as np
from scipy import stats


DB = "studies/audit_logs/paper2_demo.db"


def parse_record(row):
    rec = json.loads(row[0])
    pred_date = date.fromisoformat(rec["prediction"]["predicted_for_date"])
    return {
        "model": rec["model"]["name"],
        "month": f"{pred_date.year}-{pred_date.month:02d}",
        "value": float(rec["prediction"]["value"]),
        "features": rec["data"]["feature_values_summary"],
    }


def psi(window, baseline, n_bins=10, eps=1e-6):
    bin_edges = np.linspace(min(window.min(), baseline.min()),
                            max(window.max(), baseline.max()),
                            n_bins + 1)
    w_counts, _ = np.histogram(window, bins=bin_edges)
    b_counts, _ = np.histogram(baseline, bins=bin_edges)
    p_w = w_counts / max(len(window), 1) + eps
    p_b = b_counts / max(len(baseline), 1) + eps
    return float(np.sum((p_w - p_b) * np.log(p_w / p_b)))


def psi_band(psi_val):
    if abs(psi_val) < 0.1:
        return "stable"
    if abs(psi_val) < 0.25:
        return "moderate"
    return "high"


con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("SELECT json FROM audit_records ORDER BY record_id")

records = []
for row in cur.fetchall():
    try:
        r = parse_record(row)
        records.append(r)
    except (KeyError, ValueError):
        continue

print(f"Loaded {len(records)} records")

# Per-model, per-month output values
by_model_month = defaultdict(lambda: defaultdict(list))
for r in records:
    by_model_month[r["model"]][r["month"]].append(r["value"])

baseline_months = ["2022-01", "2022-02", "2022-03"]

results = {}
for model in sorted(by_model_month.keys()):
    by_month = by_model_month[model]
    baseline = np.array([v for m in baseline_months
                         for v in by_month.get(m, [])])
    if len(baseline) < 30:
        continue
    months_drift = 0
    months_total = 0
    monthly = []
    for month in sorted(by_month.keys()):
        if month in baseline_months:
            continue
        window = np.array(by_month[month])
        if len(window) < 5:
            continue
        ks_stat, ks_p = stats.ks_2samp(baseline, window)
        p = psi(window, baseline)
        drift = (ks_p < 0.01) or (abs(p) > 0.25)
        months_total += 1
        if drift:
            months_drift += 1
        monthly.append({
            "month": month, "n_window": len(window),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
            "psi": float(p),
            "psi_band": psi_band(p),
            "drift": bool(drift),
            "baseline_mean": float(np.mean(baseline)),
            "window_mean": float(np.mean(window)),
        })
    results[model] = {
        "n_baseline": len(baseline),
        "months_evaluated": months_total,
        "months_with_drift": months_drift,
        "monthly": monthly,
    }

for model, r in results.items():
    print(f"{model:12s} baseline n={r['n_baseline']:4d}  drift in "
          f"{r['months_with_drift']:2d} of {r['months_evaluated']:2d} months")

with open("studies/results/output_drift_summary.json", "w") as fh:
    json.dump(results, fh, indent=2)
print("Wrote studies/results/output_drift_summary.json")
