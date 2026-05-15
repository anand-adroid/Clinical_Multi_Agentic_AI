"""Audit & traceability endpoints."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.db.repositories import AuditRepository, EventRepository, ValidationRepository
from backend.db.session import get_db
from backend.schemas.api_schemas import AuditOut, EventOut, ValidationOut

router = APIRouter(prefix="/runs/{run_id}", tags=["audit"])


@router.get("/events", response_model=list[EventOut])
def list_events(run_id: str, db: Session = Depends(get_db)) -> list[EventOut]:
    return [
        EventOut.model_validate(e, from_attributes=True)
        for e in EventRepository(db).list_for_run(run_id)
    ]


@router.get("/validations", response_model=list[ValidationOut])
def list_validations(run_id: str, db: Session = Depends(get_db)) -> list[ValidationOut]:
    return [
        ValidationOut.model_validate(v, from_attributes=True)
        for v in ValidationRepository(db).list_for_run(run_id)
    ]


@router.get("/audit", response_model=list[AuditOut])
def list_audit(run_id: str, db: Session = Depends(get_db)) -> list[AuditOut]:
    return [
        AuditOut.model_validate(a, from_attributes=True)
        for a in AuditRepository(db).list_for_run(run_id)
    ]


@router.get("/audit/report")
def audit_report(run_id: str) -> dict:
    p = Path(settings.run_artifact_dir) / run_id / "audit" / "audit.json"
    if not p.exists():
        raise HTTPException(404, "No audit report generated yet for this run")
    return json.loads(p.read_text())
