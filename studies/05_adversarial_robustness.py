"""Adversarial failure-injection — corrected version that tampers the embedded
prev_record_hash inside the json column, where verify_chain actually reads from.
"""
import shutil
import sqlite3
import json
from pathlib import Path

from mr_audit import verify_chain
from mr_audit.core.store import SQLiteStore

SRC = Path("studies/audit_logs/paper2_demo.db")
WORK = Path("/tmp/mr_audit_adv4")
WORK.mkdir(exist_ok=True)


def check(db_path):
    store = SQLiteStore(db_path)
    records = store.read_all()
    valid, errors = verify_chain(records)
    store.close()
    return {"n": len(records), "valid": valid, "n_errors": len(errors)}


# Baseline
baseline = check(SRC)
print(f"BASELINE: {baseline}")

# Scenario A: corrupt prediction value inside json
db_a = WORK / "a.db"
shutil.copy(SRC, db_a)
con = sqlite3.connect(db_a)
cur = con.cursor()
cur.execute("SELECT id, json FROM audit_records WHERE id = 3001")
target_id, target_json = cur.fetchone()
rec = json.loads(target_json)
original_value = rec["prediction"]["value"]
rec["prediction"]["value"] = 99.99
tampered = json.dumps(rec, separators=(",", ":"), sort_keys=True)
cur.execute("UPDATE audit_records SET json = ? WHERE id = ?", (tampered, target_id))
con.commit()
con.close()
a = check(db_a)
print(f"A (single-record value corrupt): {a}  (was {original_value} -> 99.99)")

# Scenario B: tamper embedded prev_record_hash within json
db_b = WORK / "b.db"
shutil.copy(SRC, db_b)
con = sqlite3.connect(db_b)
cur = con.cursor()
cur.execute("SELECT id, json FROM audit_records WHERE id = 5001")
target_id, target_json = cur.fetchone()
rec = json.loads(target_json)
rec["prev_record_hash"] = "0" * 64
tampered = json.dumps(rec, separators=(",", ":"), sort_keys=True)
cur.execute("UPDATE audit_records SET json = ? WHERE id = ?", (tampered, target_id))
con.commit()
con.close()
b = check(db_b)
print(f"B (chain-link prev_record_hash tampered): {b}")

# Scenario C: tail truncation
db_c = WORK / "c.db"
shutil.copy(SRC, db_c)
con = sqlite3.connect(db_c)
cur = con.cursor()
cur.execute("DELETE FROM audit_records WHERE id > (SELECT MAX(id) - 100 FROM audit_records)")
con.commit()
con.close()
c = check(db_c)
print(f"C (tail truncation of 100 records): {c}")

# Scenario D: middle-record deletion (breaks chain in the middle)
db_d = WORK / "d.db"
shutil.copy(SRC, db_d)
con = sqlite3.connect(db_d)
cur = con.cursor()
cur.execute("DELETE FROM audit_records WHERE id = 3500")
con.commit()
con.close()
d = check(db_d)
print(f"D (middle-record deletion of id=3500): {d}")

summary = {
    "baseline": {"valid": baseline["valid"], "n_errors": baseline["n_errors"], "n_records": baseline["n"]},
    "scenario_a": {"detected": not a["valid"], "n_errors": a["n_errors"], "tamper": "single_record_prediction_value"},
    "scenario_b": {"detected": not b["valid"], "n_errors": b["n_errors"], "tamper": "embedded_prev_record_hash"},
    "scenario_c": {"detected": not c["valid"], "n_errors": c["n_errors"], "tamper": "tail_truncation_100_records",
                   "note": "Tail truncation produces a self-consistent sub-chain. Detection requires external anchoring of the latest record_id."},
    "scenario_d": {"detected": not d["valid"], "n_errors": d["n_errors"], "tamper": "middle_record_deletion"},
}
with open("studies/results/adversarial_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print("\n=== FINAL ===")
print(f"  A (record corruption):    detected={summary['scenario_a']['detected']}")
print(f"  B (chain-link tamper):    detected={summary['scenario_b']['detected']}")
print(f"  C (tail truncation):      detected={summary['scenario_c']['detected']}")
print(f"  D (middle-record delete): detected={summary['scenario_d']['detected']}")
