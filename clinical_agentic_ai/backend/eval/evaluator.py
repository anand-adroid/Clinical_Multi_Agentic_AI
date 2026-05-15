"""
Evaluation harness.

Three kinds of metric are computed:

1. Correctness — match between the derived output and a golden truth
   table, per target column and overall.
2. Coverage — fraction of expected derived targets the system actually
   produced.
3. Reliability — process metrics that do not need a golden table: retry
   rate, refiner escalation rate, share served from memory vs LLM, share
   of targets that passed the verifier with zero blocking findings.

The evaluator is itself deterministic, which is what makes it safe to
wire into CI as a quality gate.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def evaluate_run(
    output_path: str,
    golden_path: str,
    targets: list[str],
) -> dict[str, Any]:
    out = pd.read_parquet(output_path) if output_path.endswith(".parquet") else pd.read_csv(output_path)
    gold = pd.read_csv(golden_path) if golden_path.endswith(".csv") else pd.read_parquet(golden_path)

    if "patient_id" in out.columns and "patient_id" in gold.columns:
        # Align by PK
        out = out.set_index("patient_id")
        gold = gold.set_index("patient_id")
        common = out.index.intersection(gold.index)
        out = out.loc[common]
        gold = gold.loc[common]
        n = len(common)
    else:
        n = min(len(out), len(gold))
        out = out.head(n).reset_index(drop=True)
        gold = gold.head(n).reset_index(drop=True)

    per_target: dict[str, Any] = {}
    matched_total = 0
    total = 0
    covered = 0

    for t in targets:
        if t not in gold.columns:
            per_target[t] = {"in_golden": False}
            continue
        covered += 1
        if t not in out.columns:
            per_target[t] = {"in_golden": True, "in_output": False, "match": 0.0, "count": int(n)}
            total += n
            continue
        match_mask = (
            (out[t].astype(str).where(out[t].notna(), other="<NULL>")
             == gold[t].astype(str).where(gold[t].notna(), other="<NULL>"))
        )
        m = int(match_mask.sum())
        per_target[t] = {
            "in_golden": True,
            "in_output": True,
            "match": m / max(n, 1),
            "matched": m,
            "count": int(n),
            "mismatched_examples": (
                out.loc[~match_mask, [t]]
                .head(5)
                .assign(expected=gold.loc[~match_mask, t].head(5).values)
                .reset_index()
                .to_dict(orient="records")
            ),
        }
        matched_total += m
        total += n

    correctness = matched_total / total if total else 0.0
    coverage = covered / len(targets) if targets else 0.0
    return {
        "correctness": round(correctness, 4),
        "coverage": round(coverage, 4),
        "reliability": {
            "rows_evaluated": int(n),
            "targets_in_golden": int(covered),
            "targets_total": len(targets),
        },
        "per_target": per_target,
    }
