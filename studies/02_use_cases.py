"""Demonstrate the four mr-audit use cases on the Paper 2 audit log.

Use case 1: Reproducibility audit
    Pick 5 random records and replay them. Verify input hash + prediction
    match exactly.

Use case 2: Drift detection
    Split records into calendar months. For each month, run KS + PSI on the
    lagged-RV input features against the baseline (first 3 months of test
    period). Tabulate the months in which drift is detected.

Use case 3: Model risk validation report
    For each of the 7 models, compute log-level summary statistics from the
    audit log alone (no external data): record count, prediction mean/std,
    latest-record metadata, library versions.

Use case 4: Regulator export bundle
    Build a zipped, manifest-signed audit bundle from the SQLite log and
    verify it from the bundle alone (no mr-audit-internal state).

Outputs every result to studies/results/ as JSON for the audit script.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from mr_audit.core.hash import verify_chain
from mr_audit.core.store import SQLiteStore
from mr_audit.drift import detect_drift
from mr_audit.export import export_bundle, verify_bundle
from mr_audit.replay import replay

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "audit_logs" / "paper2_demo.db"
RESULTS = ROOT / "results"
BUNDLE_PATH = ROOT / "audit_logs" / "regulator_bundle.zip"

random.seed(2026)


# Re-declare the predict function from 01 (for replay).
# Signature must match the version used at instrumentation time, exactly.
def predict_volatility(*, lagged_rvs, model_params=None):
    if model_params is None:
        return 0.0
    return float(model_params.get("_forecast_value", 0.0))


def use_case_1_replay(records: list) -> dict:
    print("\n=== USE CASE 1: Reproducibility audit ===")
    sample_indices = sorted(random.sample(range(len(records)), 5))
    results = []
    for idx in sample_indices:
        rec = records[idx]
        # Reconstruct exact kwargs from the record's stored parameters
        # (rec.model.parameters already contains the full public-kwarg set).
        kwargs = dict(rec.model.parameters)
        # The stored lagged_rvs is a list (JSON-coerced); the original was a
        # tuple. The hash canonical-JSON serialises both as the same array, so
        # either form re-hashes identically.
        res = replay(rec, predict_volatility, kwargs=kwargs, tol=1e-12)
        results.append({
            "record_index": idx,
            "record_id_short": rec.record_id[:16],
            "model": rec.model.name,
            "date": rec.prediction.predicted_for_date,
            "input_hash_match": res.input_hash_match,
            "prediction_match": res.prediction_match,
            "logged": res.logged_value,
            "replayed": res.replayed_value,
        })
        marker = "OK" if res.input_hash_match and res.prediction_match else "FAIL"
        print(f"  [{marker}] record[{idx}] model={rec.model.name:<10}  date={rec.prediction.predicted_for_date}  pred={res.logged_value:.6f}")
    n_ok = sum(1 for r in results if r["input_hash_match"] and r["prediction_match"])
    print(f"  {n_ok}/{len(results)} replays match exactly")
    return {"sample_size": len(results), "n_match": n_ok, "details": results}


def use_case_2_drift(records: list) -> dict:
    print("\n=== USE CASE 2: Drift detection ===")
    # Bucket by predicted_for_date (the date the prediction is for), not by
    # the instrumentation timestamp. The instrumentation happened today; the
    # underlying forecasts cover 2022-01 to 2025-11. Drift analysis must use
    # the prediction date.
    monthly: dict[str, list] = defaultdict(list)
    for r in records:
        date_str = r.prediction.predicted_for_date or ""
        if len(date_str) >= 7:
            monthly[date_str[:7]].append(r)
    monthly = dict(sorted(monthly.items()))
    months = sorted(monthly.keys())
    print(f"  {len(months)} months in audit log: {months[0]} ... {months[-1]}")

    # Baseline: first 3 calendar months of the test period
    baseline_months = months[:3]
    baseline = []
    for m in baseline_months:
        baseline.extend(monthly[m])
    print(f"  Baseline: {baseline_months[0]}..{baseline_months[-1]} ({len(baseline)} records)")

    drift_table = []
    for m in months[3:]:
        window = monthly[m]
        results = detect_drift(baseline, window)
        # Take the MAX KS stat across the 5 lagged-rv features as the "month drift"
        max_ks = max((r.ks_statistic for r in results if r.ks_statistic == r.ks_statistic), default=float("nan"))
        min_p = min((r.ks_p_value for r in results if r.ks_p_value == r.ks_p_value), default=float("nan"))
        max_psi = max((r.psi for r in results if r.psi == r.psi), default=float("nan"))
        any_drift = any(r.psi_band == "high" for r in results) or min_p < 0.01
        drift_table.append({
            "month": m,
            "n_records": len(window),
            "max_ks_statistic": round(max_ks, 4) if max_ks == max_ks else None,
            "min_ks_p_value": round(min_p, 6) if min_p == min_p else None,
            "max_psi": round(max_psi, 4) if max_psi == max_psi else None,
            "drift_detected": bool(any_drift),
        })

    n_drift = sum(1 for r in drift_table if r["drift_detected"])
    print(f"  Drift flagged in {n_drift}/{len(drift_table)} months")
    for r in drift_table[:6]:
        marker = "DRIFT" if r["drift_detected"] else "ok"
        print(f"    {r['month']}  n={r['n_records']}  max_KS={r['max_ks_statistic']}  min_p={r['min_ks_p_value']}  [{marker}]")
    print("    ...")
    for r in drift_table[-3:]:
        marker = "DRIFT" if r["drift_detected"] else "ok"
        print(f"    {r['month']}  n={r['n_records']}  max_KS={r['max_ks_statistic']}  min_p={r['min_ks_p_value']}  [{marker}]")
    return {
        "baseline_months": baseline_months,
        "n_months_evaluated": len(drift_table),
        "n_months_with_drift": n_drift,
        "monthly_table": drift_table,
    }


def use_case_3_validation_report(records: list) -> dict:
    print("\n=== USE CASE 3: Model risk validation report ===")
    by_model: dict[str, list] = defaultdict(list)
    for r in records:
        by_model[r.model.name].append(r)

    report = []
    for model_name, recs in sorted(by_model.items()):
        preds = [r.prediction.value for r in recs if isinstance(r.prediction.value, (int, float))]
        n = len(preds)
        mean = sum(preds) / n if n else 0.0
        var = sum((p - mean) ** 2 for p in preds) / n if n else 0.0
        std = var ** 0.5
        latest = recs[-1]
        report.append({
            "model": model_name,
            "n_predictions": n,
            "first_date": recs[0].prediction.predicted_for_date,
            "last_date": latest.prediction.predicted_for_date,
            "prediction_mean": round(mean, 6),
            "prediction_std": round(std, 6),
            "code_commit_at_latest": (latest.code.code_commit_hash or "n/a")[:12],
            "library_versions_at_latest": dict(latest.code.library_versions),
        })
    print(f"  {len(report)} models in log")
    for r in report:
        print(f"    {r['model']:<10}  n={r['n_predictions']:>5}  mean={r['prediction_mean']:.4f}  std={r['prediction_std']:.4f}")
    return {"per_model_summary": report}


def use_case_4_regulator_export(records: list) -> dict:
    print("\n=== USE CASE 4: Regulator export bundle ===")
    out = export_bundle(records, BUNDLE_PATH, source_log_path=str(LOG_PATH))
    bundle_size_kb = out.stat().st_size // 1024
    print(f"  Wrote bundle: {out} ({bundle_size_kb} KB)")
    verify_result = verify_bundle(BUNDLE_PATH)
    print(f"  Bundle valid: {verify_result['overall_valid']}")
    print(f"  Files SHA-matched: {all(verify_result['files_sha_match'].values())}")
    print(f"  Chain valid: {verify_result['chain_valid']}")
    print(f"  Records in bundle: {verify_result['n_records']}")
    return {
        "bundle_path": str(out),
        "bundle_size_kb": bundle_size_kb,
        "verification": {
            "overall_valid": verify_result["overall_valid"],
            "manifest_sha_match": verify_result["manifest_sha_match"],
            "files_sha_match": verify_result["files_sha_match"],
            "chain_valid": verify_result["chain_valid"],
            "n_records": verify_result["n_records"],
            "errors": verify_result["errors"],
        },
    }


def main():
    if not LOG_PATH.exists():
        print(f"ERROR: log not found: {LOG_PATH}", file=sys.stderr)
        print("Run studies/01_instrument_paper2.py first.", file=sys.stderr)
        return 1

    RESULTS.mkdir(exist_ok=True)
    print(f"Loading audit log: {LOG_PATH}")
    store = SQLiteStore(LOG_PATH)
    records = store.read_all()
    print(f"  records: {len(records)}")

    # Pre-flight: chain integrity
    valid, errors = verify_chain(records)
    print(f"  chain valid: {valid}")
    if not valid:
        print(f"  errors: {errors[:3]}")

    case1 = use_case_1_replay(records)
    case2 = use_case_2_drift(records)
    case3 = use_case_3_validation_report(records)
    case4 = use_case_4_regulator_export(records)

    summary = {
        "n_records": len(records),
        "chain_integrity": {"valid": valid, "errors": errors},
        "use_case_1_reproducibility": case1,
        "use_case_2_drift": case2,
        "use_case_3_validation": case3,
        "use_case_4_export": case4,
    }
    out_path = RESULTS / "use_cases_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
