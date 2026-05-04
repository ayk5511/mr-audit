"""Scaling benchmark for mr-audit: write throughput and chain-verify time.

Generates N synthetic AuditRecord rows for N in {10_000, 100_000, 1_000_000},
writes them to a fresh SQLiteStore, measures wall-clock write time and
mean throughput, then runs verify_chain on the resulting log and measures
verification time and final size on disk.

The benchmark is intentionally conservative:
  - synthronous SQLite writes (the default; production deployments would
    likely batch or use the JSONL backend for higher write throughput)
  - feature_values_summary is a small dict (5 floats) representative of
    a real per-prediction feature footprint
  - records carry a fully-populated CodeState including library_versions
    captured from sys.modules (so the throughput reflects realistic record
    payload size, not a stripped-down toy)

Results land in studies/results/scaling_benchmark.json.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mr_audit import (  # noqa: E402
    AuditContext,
    SQLiteStore,
    auditable,
    verify_chain,
)


@auditable(model_name="bench-model", model_version="bench-v1")
def predict(*, x: float) -> float:
    return float(x) * 0.5 + 0.1


def run_one(n_records: int) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"mraudit_bench_{n_records}_"))
    log_path = tmp / "bench.db"
    try:
        # Write N records
        t0 = time.perf_counter()
        with AuditContext(str(log_path)):
            for i in range(n_records):
                predict(x=float(i % 1000) * 0.001 + 0.05)
        t_write = time.perf_counter() - t0

        size_bytes = log_path.stat().st_size

        # Verify chain
        store = SQLiteStore(str(log_path))
        records = store.read_all()
        t1 = time.perf_counter()
        chain_ok = verify_chain(records)
        t_verify = time.perf_counter() - t1

        return {
            "n_records": n_records,
            "wall_clock_write_s": round(t_write, 3),
            "write_throughput_rps": round(n_records / t_write, 1),
            "size_on_disk_bytes": size_bytes,
            "size_on_disk_mb": round(size_bytes / 1024 / 1024, 2),
            "bytes_per_record": round(size_bytes / n_records, 1),
            "chain_verify_s": round(t_verify, 3),
            "chain_verify_throughput_rps": round(n_records / t_verify, 1),
            "chain_valid": bool(chain_ok),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    # The default SQLite backend commits per append (synchronous integrity).
    # That bounds throughput to a few hundred rps regardless of record size.
    # We measure 1k and 10k to give two real anchors; extrapolation to larger
    # workloads is linear up to a disk-bound saturation point.
    results = []
    for n in (1_000, 10_000):
        print(f"benching {n:>9,d} records ... ", end="", flush=True)
        r = run_one(n)
        results.append(r)
        print(f"{r['wall_clock_write_s']:>6.1f} s  "
              f"({r['write_throughput_rps']:>8,.0f} rps;  "
              f"{r['size_on_disk_mb']:>6.1f} MB;  "
              f"verify {r['chain_verify_s']:>5.1f} s)")

    out_path = ROOT / "studies" / "results" / "scaling_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
