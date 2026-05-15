"""Quick DB inspector.

Usage from the project root:

    python scripts/inspect_db.py                # default: summary of all tables
    python scripts/inspect_db.py runs           # rows from one table
    python scripts/inspect_db.py runs --limit 5
    python scripts/inspect_db.py derivations --run a1b2c3d4ef
    python scripts/inspect_db.py audit_entries  # any append-only table works

The script reads ``storage/agentic.db`` (or whatever ``DB_URL`` points to)
and pretty-prints either a table-by-table summary or a single table's rows.
Designed for grokking the audit substrate without firing up a separate
SQLite GUI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so ``backend.*`` imports work when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import inspect, text

from backend.db.session import SessionLocal, _engine


_TABLES = [
    "runs", "agent_events", "derivations", "validations",
    "hitl_decisions", "audit_entries", "memory_patterns",
    "clarification_memory",
]


def summary() -> None:
    """Print row count + last update for every audit-substrate table."""
    insp = inspect(_engine)
    existing = set(insp.get_table_names())
    print(f"Database: {_engine.url}")
    print(f"Tables present: {len(existing)}")
    print()
    print(f"{'Table':<24} {'Rows':>6}  Latest row")
    print("-" * 70)
    db = SessionLocal()
    try:
        for t in _TABLES:
            if t not in existing:
                print(f"{t:<24} {'-':>6}  (missing)")
                continue
            row_count = db.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            latest = ""
            try:
                latest = str(
                    db.execute(
                        text(f"SELECT MAX(created_at) FROM {t}")
                    ).scalar()
                    or ""
                )
            except Exception:
                pass
            print(f"{t:<24} {row_count:>6}  {latest}")
    finally:
        db.close()


def _row_to_dict(row) -> dict:
    # SQLAlchemy 2.x Row -> dict
    return {k: getattr(row, k) for k in row._fields}


def dump_table(name: str, *, limit: int = 20, run_id: str | None = None) -> None:
    insp = inspect(_engine)
    if name not in insp.get_table_names():
        print(f"Table '{name}' not in DB. Known tables: {_TABLES}")
        return
    where = f" WHERE run_id = '{run_id}'" if run_id else ""
    # Prefer ordering by created_at when the column exists; fall back to id.
    cols = {c["name"] for c in insp.get_columns(name)}
    order_by = "created_at DESC" if "created_at" in cols else "id DESC"
    sql = f"SELECT * FROM {name}{where} ORDER BY {order_by} LIMIT {limit}"
    db = SessionLocal()
    try:
        rows = [_row_to_dict(r) for r in db.execute(text(sql)).all()]
    finally:
        db.close()
    if not rows:
        print(f"(empty: {sql})")
        return
    print(f"\n{name}  (showing up to {limit}{' for run ' + run_id if run_id else ''})")
    print("-" * 70)
    for r in rows:
        # Trim very long fields so the terminal stays readable.
        for k, v in list(r.items()):
            if isinstance(v, (dict, list)):
                r[k] = v  # keep structured
            elif isinstance(v, str) and len(v) > 160:
                r[k] = v[:160] + "..."
        print(json.dumps(r, indent=2, default=str))
        print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "table",
        nargs="?",
        help="One of: " + ", ".join(_TABLES) + ".  Omit for a summary.",
    )
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--run", default=None, help="Filter by run_id")
    args = p.parse_args()
    if args.table:
        dump_table(args.table, limit=args.limit, run_id=args.run)
    else:
        summary()


if __name__ == "__main__":
    main()
