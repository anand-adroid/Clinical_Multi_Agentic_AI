"""HITL Quality Dashboard endpoints.

Industry-grade HITL is not just "show a button" — it's a measurable
process. These endpoints surface metrics every regulated AI platform tracks:
intervention rate, false-alert rate, memory reuse, decision latency, and
confidence calibration. Together they prove the HITL process actually
catches problems instead of inducing reviewer fatigue.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.models import (
    AgentEvent,
    AuditEntry,
    ClarificationMemory,
    Derivation,
    HITLDecision,
    MemoryPattern,
    Run,
    Validation,
)
from backend.db.session import get_db


router = APIRouter(prefix="/hitl-quality", tags=["hitl-quality"])


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[k])


@router.get("/summary")
def quality_summary(
    days: int = 30,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return aggregated HITL quality metrics over the last ``days``.

    Five headline metrics:
      - intervention_rate: % of HITL gates where action != approve
      - false_alert_rate: % of HITL gates resolved purely with 'approve' (no edit/regen)
      - memory_reuse_rate: % of derivations served from LTM (vs new LLM)
      - decision_latency_p50_ms / p95_ms: time spent on each HITL gate
      - confidence_calibration: avg LLM confidence on gated derivations vs override rate
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    decisions = (
        db.query(HITLDecision)
        .filter(HITLDecision.created_at >= cutoff)
        .all()
    )
    audit_rows = (
        db.query(AuditEntry)
        .filter(AuditEntry.created_at >= cutoff)
        .all()
    )
    derivations = (
        db.query(Derivation)
        .filter(Derivation.created_at >= cutoff)
        .all()
    )

    total_decisions = len(decisions)
    edits = sum(1 for d in decisions if d.action in ("edit", "regenerate"))
    rejects = sum(1 for d in decisions if d.action == "reject")
    pure_approves = sum(
        1 for d in decisions
        if d.action == "approve" and not d.edited_code and not (d.comment or "").strip()
    )

    intervention_rate = (
        ((edits + rejects) / total_decisions * 100) if total_decisions else 0.0
    )
    false_alert_rate = (
        (pure_approves / total_decisions * 100) if total_decisions else 0.0
    )

    # Memory reuse: % of derivations whose generator is "memory"
    total_derivs = len(derivations)
    mem_derivs = sum(1 for d in derivations if d.generator == "memory")
    memory_reuse_rate = (
        (mem_derivs / total_derivs * 100) if total_derivs else 0.0
    )

    # Decision latency: time between HITL pause (hitl.requested audit row)
    # and the matching HITL decision row, per run.
    latencies_ms: list[float] = []
    pause_times: dict[tuple[str, str], datetime] = {}
    for a in audit_rows:
        if a.action == "hitl.requested":
            key = (a.run_id, (a.object_ref or "") or (a.detail or {}).get("reason", ""))
            pause_times[key] = a.created_at
    for d in decisions:
        # Match the earliest pause for that run that hasn't been consumed.
        match_key = next(
            (k for k in pause_times if k[0] == d.run_id),
            None,
        )
        if match_key and pause_times[match_key]:
            dt_ms = (d.created_at - pause_times[match_key]).total_seconds() * 1000
            if dt_ms >= 0:
                latencies_ms.append(dt_ms)
                pause_times.pop(match_key, None)

    latency_p50 = _percentile(latencies_ms, 50)
    latency_p95 = _percentile(latencies_ms, 95)

    # Confidence calibration: average confidence on derivations that had a HITL
    # decision attached. Where confidence is high but humans intervened, the
    # confidence is poorly calibrated (overconfident).
    intervened_targets = {(d.run_id, d.target) for d in decisions if d.target}
    confidence_overrides: list[tuple[float, bool]] = []
    for d in derivations:
        # Confidence is not stored on the Derivation DB row yet; the proxy
        # below infers it from the generator (memory=1.0, llm default 0.85).
        # A richer signal would
        # persist confidence in the derivation row — listed as a v2 item.
        was_overridden = (d.run_id, d.target) in intervened_targets
        proxy_conf = 1.0 if d.generator == "memory" else 0.85  # LLM default
        confidence_overrides.append((proxy_conf, was_overridden))

    overconfident_overrides = sum(
        1 for c, overridden in confidence_overrides if c >= 0.85 and overridden
    )
    high_conf_total = sum(1 for c, _ in confidence_overrides if c >= 0.85)
    overconfidence_rate = (
        (overconfident_overrides / high_conf_total * 100) if high_conf_total else 0.0
    )

    # Run-level counts
    runs_total = db.query(func.count(Run.id)).filter(Run.created_at >= cutoff).scalar() or 0
    runs_with_hitl = (
        db.query(func.count(func.distinct(HITLDecision.run_id)))
        .filter(HITLDecision.created_at >= cutoff)
        .scalar() or 0
    )

    # Memory growth
    pattern_count = db.query(func.count(MemoryPattern.id)).scalar() or 0
    clarif_count = db.query(func.count(ClarificationMemory.id)).scalar() or 0

    return {
        "window_days": days,
        "headline": {
            "intervention_rate_pct": round(intervention_rate, 1),
            "false_alert_rate_pct": round(false_alert_rate, 1),
            "memory_reuse_rate_pct": round(memory_reuse_rate, 1),
            "decision_latency_p50_ms": round(latency_p50, 1),
            "decision_latency_p95_ms": round(latency_p95, 1),
            "overconfidence_rate_pct": round(overconfidence_rate, 1),
        },
        "counts": {
            "total_runs": runs_total,
            "runs_with_hitl": runs_with_hitl,
            "total_hitl_decisions": total_decisions,
            "edits": edits,
            "rejects": rejects,
            "pure_approves": pure_approves,
            "memory_patterns": pattern_count,
            "clarification_memory": clarif_count,
            "total_derivations": total_derivs,
            "memory_served_derivations": mem_derivs,
        },
        "targets": {
            "intervention_rate_pct": "10–30",
            "false_alert_rate_pct": "below 50",
            "memory_reuse_rate_pct": "should grow over time",
            "decision_latency_p95_ms": "depends on gate SLA",
            "overconfidence_rate_pct": "below 5",
        },
    }


@router.get("/recent-decisions")
def recent_decisions(
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Most-recent HITL decisions across all runs (for the activity feed)."""
    rows = (
        db.query(HITLDecision)
        .order_by(HITLDecision.created_at.desc())
        .limit(limit)
        .all()
    )
    out = []
    for r in rows:
        # Try to parse the structured comment payload (clarification_answers etc.)
        structured = None
        if r.comment:
            try:
                parsed = json.loads(r.comment)
                if isinstance(parsed, dict) and (
                    "clarification_answers" in parsed
                    or "derivation_overrides" in parsed
                ):
                    structured = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        out.append({
            "id": r.id,
            "run_id": r.run_id,
            "target": r.target,
            "reviewer": r.reviewer,
            "action": r.action,
            "comment": r.comment if not structured else (structured.get("comment") or ""),
            "structured_payload": structured,
            "edited": bool(r.edited_code),
            "created_at": r.created_at.isoformat(),
        })
    return out
