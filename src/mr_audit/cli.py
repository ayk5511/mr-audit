"""Command-line interface: `mr-audit` entry point.

Subcommands:
    show     <log>             Print a tabular summary of records in a log
    verify   <log>             Verify the hash chain of a log
    export   <log>  <bundle>   Build a regulator-ready bundle from a log
    inspect  <bundle>          Verify and summarise a bundle (zip)

The CLI is deliberately small. It exposes the four operations a model-risk
team would actually run from a terminal; programmatic users should import
the Python API.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.hash import verify_chain
from .core.store import JSONLStore, SQLiteStore
from .export.bundle import export_bundle, verify_bundle


def _load(path: str):
    p = Path(path)
    if not p.exists():
        print(f"error: log file not found: {path}", file=sys.stderr)
        sys.exit(2)
    suffix = p.suffix.lower()
    if suffix in (".db", ".sqlite", ".sqlite3"):
        return SQLiteStore(p)
    return JSONLStore(p)


def cmd_show(args: argparse.Namespace) -> int:
    store = _load(args.log)
    records = store.read_all()
    if args.json:
        print(json.dumps(
            [{"record_id": r.record_id[:16] + "...",
              "timestamp": r.timestamp_utc,
              "model": f"{r.model.name}:{r.model.version}",
              "prediction": r.prediction.value}
             for r in records],
            indent=2,
        ))
    else:
        print(f"{len(records)} record(s) in {args.log}")
        print()
        print(f"{'#':<4} {'record_id':<20} {'timestamp':<26} {'model':<25} prediction")
        print("-" * 100)
        for i, r in enumerate(records[: args.limit if args.limit > 0 else len(records)]):
            model_label = f"{r.model.name}:{r.model.version}"
            print(
                f"{i:<4} {r.record_id[:16] + '..':<20} {r.timestamp_utc:<26} "
                f"{model_label:<25} {r.prediction.value}"
            )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    store = _load(args.log)
    records = store.read_all()
    valid, errors = verify_chain(records)
    print(f"Records: {len(records)}")
    print(f"Chain valid: {valid}")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")
        return 1
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = _load(args.log)
    records = store.read_all()
    out = export_bundle(records, args.bundle, source_log_path=args.log)
    print(f"Wrote bundle: {out}")
    print(f"  records: {len(records)}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    result = verify_bundle(args.bundle)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("overall_valid") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mr-audit",
        description="An audit-trail framework for ML pipelines under SR 11-7 and the EU AI Act.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    show_p = sub.add_parser("show", help="Print a tabular summary of an audit log")
    show_p.add_argument("log", help="Path to audit log (.db, .sqlite, .jsonl)")
    show_p.add_argument("--json", action="store_true", help="Output JSON instead of a table")
    show_p.add_argument("--limit", type=int, default=20, help="Max records to print (0=all)")
    show_p.set_defaults(func=cmd_show)

    verify_p = sub.add_parser("verify", help="Verify the hash chain of an audit log")
    verify_p.add_argument("log", help="Path to audit log")
    verify_p.set_defaults(func=cmd_verify)

    export_p = sub.add_parser("export", help="Export a regulator-ready bundle from an audit log")
    export_p.add_argument("log", help="Path to audit log")
    export_p.add_argument("bundle", help="Output path for the .zip bundle")
    export_p.set_defaults(func=cmd_export)

    inspect_p = sub.add_parser("inspect", help="Verify and summarise an exported bundle")
    inspect_p.add_argument("bundle", help="Path to .zip bundle")
    inspect_p.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
