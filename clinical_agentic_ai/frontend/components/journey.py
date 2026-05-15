"""Map internal pipeline state into the user's six-step journey.

The orchestrator runs twelve phases; that is correct for a regulator but
too much detail for a clinical reviewer. I map every phase into one of
six milestones that describe the journey from the reviewer's perspective:

  1. Submit                   inputs received
  2. Specification review     ambiguities resolved (auto or HITL)
  3. Code generation          AI produced derive(row) functions
  4. Code approval            humans approved regulatory-critical code
  5. Verification & execution code ran, tests passed, output written
  6. Audit & output           audit ready, downloads available

Each milestone carries one of seven visual states (pending, active,
awaiting_you, complete, complete_with_decisions, skipped, blocked). The
stepper component renders those states; this module just decides what
state each milestone is in given a run summary and (optionally) the
detailed state checkpoint.
"""
from __future__ import annotations

from typing import Any


# Public order — UI components rely on this for the stepper layout.
MILESTONE_KEYS = [
    "submit",
    "spec_review",
    "code_generation",
    "code_approval",
    "verification",
    "audit",
]

MILESTONE_LABEL = {
    "submit": "Submit",
    "spec_review": "Specification review",
    "code_generation": "Code generation",
    "code_approval": "Code approval",
    "verification": "Verification & execution",
    "audit": "Audit & output",
}


# ----------------------------------------------------------------- status text
_STATUS_TRANSLATIONS: dict[str, tuple[str, str]] = {
    # internal -> (human label, semantic color)
    "created": ("Created", "gray"),
    "running": ("Running", "blue"),
    "awaiting_hitl_spec": ("Awaiting your answers", "amber"),
    "awaiting_hitl_preapproval": ("Awaiting your review", "amber"),
    "awaiting_hitl_refine": ("Awaiting your code fix", "amber"),
    "awaiting_hitl_codegen": ("Code generation failed", "red"),
    "completed": ("Completed", "green"),
    "completed_with_warnings": ("Completed with warnings", "amber"),
    "failed": ("Failed", "red"),
}


def human_status(status: str | None) -> tuple[str, str]:
    """Return ``(human_label, semantic_color)`` for an internal status."""
    if not status:
        return ("Unknown", "gray")
    return _STATUS_TRANSLATIONS.get(status, (str(status), "gray"))


# Maps the orchestrator's internal phase keys to short user-facing labels
# that fit inside a status pill. Keep these terse — the pill is narrow.
_PHASE_LABEL: dict[str, str] = {
    "input_guard": "Input check",
    "review": "Spec review",
    "build_dag": "DAG build",
    "plan": "Planning",
    "generate": "Code generation",
    "preapproval": "Code preapproval",
    "static_check": "Safety check",
    "test_cases": "Test cases",
    "execute": "Executing",
    "verify": "Verification",
    "refine": "Refining",
    "audit": "Audit",
}

# Total pipeline phases — exposed for the "step X of N" UX. Kept in lock
# step with ``PHASE_ORDER`` in ``backend/core/orchestrator.py``.
TOTAL_PHASES = 12


def current_phase(state: dict | None) -> tuple[str, int] | None:
    """Inspect the run's recorded agent events and return the most recently
    *started* phase along with its 1-indexed position in the pipeline.

    Returns ``None`` when there is no event history yet. The result is
    designed for use in status pills like ``Running: Code generation (5/12)``
    so the user can see WHERE in the pipeline a run is, not just THAT it
    is running.
    """
    if not state:
        return None
    events = state.get("events") or []
    if not events:
        return None
    # ``events`` is recorded in chronological order. The "current" phase
    # for display purposes is the latest one — completed or in-progress
    # — because that is what just happened from the user's viewpoint.
    last = events[-1]
    step = last.get("step", "")
    label = _PHASE_LABEL.get(step)
    if not label:
        return None
    # Map the step back to its ordinal in PHASE_ORDER (1-indexed).
    phase_order = [
        "input_guard", "review", "build_dag", "plan", "generate",
        "preapproval", "static_check", "test_cases", "execute",
        "verify", "refine", "audit",
    ]
    try:
        ordinal = phase_order.index(step) + 1
    except ValueError:
        ordinal = 0
    return (label, ordinal)


def needs_user_action(status: str | None) -> bool:
    return bool(status and status.startswith("awaiting_hitl_"))


# ----------------------------------------------------------------- milestones
def _completed_steps(state: dict[str, Any] | None) -> set[str]:
    if not state:
        return set()
    events = state.get("events") or []
    return {
        e.get("step")
        for e in events
        if e.get("status") in ("ok", "warn")
    }


def journey_milestones(
    run: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute the six-milestone view of a run.

    Each returned item is a dict with keys ``key``, ``label``, ``state``,
    and ``detail``. ``state`` is one of: pending, active, awaiting_you,
    complete, complete_with_decisions, skipped, blocked.
    """
    status = (run or {}).get("status") or ""
    state = state or {}
    completed = _completed_steps(state)
    derivations = state.get("derivations") or {}
    validations = state.get("validations") or []
    hitl_history = state.get("hitl_history") or []
    hitl_pending = state.get("hitl_pending") or {}
    pending_reason = (hitl_pending.get("reason") or "") if isinstance(hitl_pending, dict) else ""
    pending_ctx = (hitl_pending.get("context") or {}) if isinstance(hitl_pending, dict) else {}

    items: list[dict[str, Any]] = []

    # 1. Submit ------------------------------------------------------------
    items.append({
        "key": "submit",
        "label": MILESTONE_LABEL["submit"],
        "state": "complete",
        "detail": (run.get("created_at") or "")[:19].replace("T", " "),
    })

    # 2. Specification review ---------------------------------------------
    spec_done = "review" in completed
    spec_clarifs = sum(
        len((h.get("clarification_answers") or {}))
        for h in hitl_history
    )
    if pending_reason == "spec_clarifications_required":
        n = len((pending_ctx.get("clarifications") or []))
        items.append({
            "key": "spec_review",
            "label": MILESTONE_LABEL["spec_review"],
            "state": "awaiting_you",
            "detail": f"{n} clarification(s) need your input",
        })
    elif spec_done:
        if spec_clarifs > 0:
            items.append({
                "key": "spec_review",
                "label": MILESTONE_LABEL["spec_review"],
                "state": "complete_with_decisions",
                "detail": f"{spec_clarifs} clarification(s) answered",
            })
        else:
            items.append({
                "key": "spec_review",
                "label": MILESTONE_LABEL["spec_review"],
                "state": "complete",
                "detail": "No ambiguities found",
            })
    else:
        items.append({
            "key": "spec_review",
            "label": MILESTONE_LABEL["spec_review"],
            "state": "active" if status == "running" else "pending",
            "detail": "",
        })

    # 3. Code generation ---------------------------------------------------
    gen_done = "generate" in completed
    if pending_reason == "codegen_failed":
        items.append({
            "key": "code_generation",
            "label": MILESTONE_LABEL["code_generation"],
            "state": "blocked",
            "detail": "AI could not generate code",
        })
    elif gen_done:
        n_cols = len([d for d in derivations.values() if d.get("status") in ("generated", "ok", "refined")])
        items.append({
            "key": "code_generation",
            "label": MILESTONE_LABEL["code_generation"],
            "state": "complete",
            "detail": f"{n_cols} column(s) generated",
        })
    else:
        items.append({
            "key": "code_generation",
            "label": MILESTONE_LABEL["code_generation"],
            "state": "active" if status == "running" else "pending",
            "detail": "",
        })

    # 4. Code approval -----------------------------------------------------
    approval_done = "preapproval" in completed
    decisions_with_overrides = [
        h for h in hitl_history
        if (h.get("derivation_overrides") or {})
    ]
    if pending_reason == "code_preapproval_required":
        n_targets = len((pending_ctx.get("preapproval_targets") or []))
        items.append({
            "key": "code_approval",
            "label": MILESTONE_LABEL["code_approval"],
            "state": "awaiting_you",
            "detail": f"{n_targets} derivation(s) need your review",
        })
    elif approval_done:
        n_approvals = sum(
            len((h.get("derivation_overrides") or {}))
            for h in decisions_with_overrides
        )
        if n_approvals > 0:
            items.append({
                "key": "code_approval",
                "label": MILESTONE_LABEL["code_approval"],
                "state": "complete_with_decisions",
                "detail": f"{n_approvals} approval(s) recorded",
            })
        else:
            items.append({
                "key": "code_approval",
                "label": MILESTONE_LABEL["code_approval"],
                "state": "skipped",
                "detail": "Confidence high; no review required",
            })
    else:
        items.append({
            "key": "code_approval",
            "label": MILESTONE_LABEL["code_approval"],
            "state": "active" if status == "running" else "pending",
            "detail": "",
        })

    # 5. Verification & execution -----------------------------------------
    verify_done = "verify" in completed
    if pending_reason == "refinement_exhausted":
        tgt = hitl_pending.get("target") if isinstance(hitl_pending, dict) else None
        items.append({
            "key": "verification",
            "label": MILESTONE_LABEL["verification"],
            "state": "awaiting_you",
            "detail": f"Code fix needed for {tgt}" if tgt else "Code fix needed",
        })
    elif verify_done:
        tc = [v for v in validations if (v.get("rule_id") or "").startswith("TEST_CASE_")]
        tc_pass = sum(1 for v in tc if v.get("passed"))
        if tc:
            items.append({
                "key": "verification",
                "label": MILESTONE_LABEL["verification"],
                "state": "complete",
                "detail": f"{tc_pass}/{len(tc)} tests passing",
            })
        else:
            items.append({
                "key": "verification",
                "label": MILESTONE_LABEL["verification"],
                "state": "complete",
                "detail": "Verified",
            })
    else:
        items.append({
            "key": "verification",
            "label": MILESTONE_LABEL["verification"],
            "state": "active" if status == "running" else "pending",
            "detail": "",
        })

    # 6. Audit & output ----------------------------------------------------
    audit_done = "audit" in completed or status.startswith("completed")
    if status == "failed":
        items.append({
            "key": "audit",
            "label": MILESTONE_LABEL["audit"],
            "state": "blocked",
            "detail": "Run failed",
        })
    elif audit_done:
        items.append({
            "key": "audit",
            "label": MILESTONE_LABEL["audit"],
            "state": "complete",
            "detail": "Audit ready",
        })
    else:
        items.append({
            "key": "audit",
            "label": MILESTONE_LABEL["audit"],
            "state": "pending",
            "detail": "",
        })

    return items
