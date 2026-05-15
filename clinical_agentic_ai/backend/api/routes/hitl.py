"""Human-in-the-Loop endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.core.orchestrator import Orchestrator
from backend.db.repositories import HITLRepository, RunRepository
from backend.db.session import get_db
from backend.memory.short_term import ShortTermMemory
from backend.schemas.api_schemas import HITLDecisionRequest, HITLOut, RunSummary

router = APIRouter(prefix="/runs/{run_id}/hitl", tags=["hitl"])


@router.get("/pending")
def get_pending(run_id: str, db: Session = Depends(get_db)) -> dict:
    state = ShortTermMemory.restore(db, run_id)
    if not state:
        raise HTTPException(404, f"No state for run {run_id}")
    if not state.hitl_pending:
        return {"pending": False}
    return {
        "pending": True,
        "target": state.hitl_pending.target,
        "reason": state.hitl_pending.reason,
        "context": state.hitl_pending.context,
        "raised_at": state.hitl_pending.raised_at,
    }


@router.post("/decision", response_model=RunSummary)
def post_decision(run_id: str, req: HITLDecisionRequest, db: Session = Depends(get_db)) -> RunSummary:
    if req.action not in ("approve", "edit", "reject", "regenerate"):
        raise HTTPException(400, "action must be approve | edit | reject | regenerate")
    orch = Orchestrator(db)
    # Apply the decision synchronously (fast — state mutation + DB writes),
    # then spawn a daemon thread to continue the workflow. The HTTP response
    # returns immediately with the current state; the frontend polls for
    # progress just like for a fresh run.
    state = orch.apply_hitl_decision(
        run_id=run_id, reviewer=req.reviewer, action=req.action,
        target=req.target, comment=req.comment, edited_code=req.edited_code,
        clarification_answers=req.clarification_answers,
        derivation_overrides=req.derivation_overrides,
        regenerate_hint=req.regenerate_hint,
        resume=False,
    )
    if (
        not state.status.startswith("awaiting_hitl_")
        and state.status != "failed"
    ):
        from backend.api.routes.workflows import _spawn_run
        _spawn_run(run_id)
    return RunSummary.model_validate(orch.runs.get(state.run_id), from_attributes=True)


@router.get("/history", response_model=list[HITLOut])
def history(run_id: str, db: Session = Depends(get_db)) -> list[HITLOut]:
    rows = HITLRepository(db).list_for_run(run_id)
    return [HITLOut.model_validate(r, from_attributes=True) for r in rows]
