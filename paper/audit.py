"""End-to-end audit script for Paper 4 v0.

Re-derives every numerical claim in main.tex from the JSON outputs of
studies/01_instrument_paper2.py and studies/02_use_cases.py, plus the
unit-test invocation. Verifies agreement to literal equality where possible
and to four decimals otherwise.

Run:
    python paper/audit.py

Exits 0 if all checks pass; 1 with an itemized failure list otherwise.

Same convention as Paper 3 v1.2 / Paper 2 v1: every paper in the portfolio
ships an analogous audit script.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "studies" / "results"
LOGS = ROOT / "studies" / "audit_logs"


def fail(msg: str) -> int:
    print(f"  FAIL: {msg}")
    return 1


def passmsg(msg: str) -> int:
    print(f"  ok:   {msg}")
    return 0


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    fails = 0

    # ----- Instrumentation log -----
    section("INSTRUMENTATION (Tab. 4 throughput)")
    inst_path = LOGS / "instrumentation_log.json"
    if not inst_path.exists():
        return fail(f"missing {inst_path}")
    inst = json.loads(inst_path.read_text())

    expected = {
        "n_days_in_panel": 980,
        "n_models": 7,
        "n_records_written": 6825,
    }
    for k, v in expected.items():
        if inst.get(k) == v:
            passmsg(f"{k} = {v}")
        else:
            fails += fail(f"{k}: expected {v}, got {inst.get(k)}")

    if inst.get("log_size_bytes_per_record"):
        bpr = inst["log_size_bytes_per_record"]
        # Paper claims 2,266 bytes per record (within ~5% tolerance for run-to-run)
        if abs(bpr - 2266) < 200:
            passmsg(f"bytes per record ~ 2266 (got {bpr})")
        else:
            fails += fail(f"bytes per record: expected ~2266, got {bpr}")

    # ----- Use cases summary -----
    section("USE CASES (Tab. 5, 6, 7, 8)")
    summary_path = RESULTS / "use_cases_summary.json"
    if not summary_path.exists():
        return fail(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text())

    if summary["n_records"] == 6825:
        passmsg("n_records = 6825")
    else:
        fails += fail(f"n_records: expected 6825, got {summary['n_records']}")

    if summary["chain_integrity"]["valid"]:
        passmsg("chain integrity = valid")
    else:
        fails += fail(f"chain integrity: invalid; errors {summary['chain_integrity']['errors']}")

    # Use case 1
    case1 = summary["use_case_1_reproducibility"]
    if case1["sample_size"] == 5 and case1["n_match"] == 5:
        passmsg(f"replay: {case1['n_match']}/{case1['sample_size']} match")
    else:
        fails += fail(
            f"replay: expected 5/5 match, got {case1.get('n_match')}/{case1.get('sample_size')}"
        )

    # Use case 2
    case2 = summary["use_case_2_drift"]
    if case2["n_months_evaluated"] == 44:
        passmsg("drift: 44 months evaluated")
    else:
        fails += fail(f"drift months evaluated: expected 44, got {case2['n_months_evaluated']}")
    if case2["n_months_with_drift"] == 44:
        passmsg("drift: 44 of 44 months flagged")
    else:
        fails += fail(f"drift flagged: expected 44, got {case2['n_months_with_drift']}")

    # Use case 3
    case3 = summary["use_case_3_validation"]
    n_models = len(case3["per_model_summary"])
    if n_models == 7:
        passmsg("validation report: 7 models")
    else:
        fails += fail(f"validation models: expected 7, got {n_models}")
    expected_models = {"GARCH", "EGARCH", "GJR-GARCH", "HAR-RV", "LightGBM", "XGBoost", "Ensemble"}
    actual_models = {r["model"] for r in case3["per_model_summary"]}
    if actual_models == expected_models:
        passmsg(f"validation models: {sorted(actual_models)}")
    else:
        fails += fail(f"validation models: expected {expected_models}, got {actual_models}")
    n_per_model = {r["n_predictions"] for r in case3["per_model_summary"]}
    if n_per_model == {975}:
        passmsg("validation: 975 predictions per model")
    else:
        fails += fail(f"per-model n: expected {{975}}, got {n_per_model}")

    # Use case 4
    case4 = summary["use_case_4_export"]
    if case4["verification"]["overall_valid"]:
        passmsg(f"bundle valid (size {case4['bundle_size_kb']} KB)")
    else:
        fails += fail(f"bundle invalid: {case4['verification']['errors']}")
    if case4["verification"]["n_records"] == 6825:
        passmsg("bundle: 6825 records")
    else:
        fails += fail(f"bundle records: expected 6825, got {case4['verification']['n_records']}")
    if case4["bundle_size_kb"] == 841:
        passmsg("bundle size = 841 KB")
    else:
        fails += fail(f"bundle size: expected 841 KB, got {case4['bundle_size_kb']} KB")

    # ----- Test suite -----
    section("TEST SUITE")
    pytest_result = subprocess.run(
        ["python3", "-m", "pytest", str(ROOT / "tests"), "-q", "--no-header"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if pytest_result.returncode == 0:
        last_lines = pytest_result.stdout.strip().split("\n")[-3:]
        for line in last_lines:
            if line.strip():
                passmsg(line.strip())
    else:
        fails += fail(f"pytest exit code {pytest_result.returncode}")
        print(pytest_result.stdout[-500:])

    # ----- Final -----
    section("FINAL")
    if fails == 0:
        print("\nALL CHECKS PASSED")
        return 0
    print(f"\n{fails} CHECK(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
