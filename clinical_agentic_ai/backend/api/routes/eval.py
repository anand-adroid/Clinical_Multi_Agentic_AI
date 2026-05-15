"""Evaluation endpoints — wired into backend.eval.evaluator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.db.session import get_db
from backend.eval.evaluator import evaluate_run
from backend.memory.short_term import ShortTermMemory
from backend.schemas.api_schemas import EvalResultOut

router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/{run_id}", response_model=EvalResultOut)
def eval_run(run_id: str, golden_path: str | None = None, db: Session = Depends(get_db)) -> EvalResultOut:
    state = ShortTermMemory.restore(db, run_id)
    if not state or not state.output_path:
        raise HTTPException(404, "Run output not available")
    gp = golden_path or str(Path(settings.run_artifact_dir).parent.parent / "data" / "golden" / "expected.csv")
    if not Path(gp).exists():
        raise HTTPException(404, f"Golden file not found: {gp}")
    result = evaluate_run(state.output_path, gp, list(state.derivations.keys()))
    # Persist the eval alongside the audit so it appears in the report.
    Path(state.output_path).parent.joinpath("eval.json").write_text(
        json.dumps(result, indent=2)
    )
    return EvalResultOut(**result)
