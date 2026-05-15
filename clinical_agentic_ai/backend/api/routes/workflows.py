"""Workflow lifecycle endpoints.

Long-running orchestrator work runs in a daemon thread per run. HTTP
handlers return immediately with the current state; the frontend polls
``GET /runs/{id}`` to learn when the run completes or pauses for HITL.
This decouples the request lifecycle from the pipeline lifecycle —
without it a single in-flight run would block every other endpoint
including ``/health``.
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.orchestrator import Orchestrator
from backend.core.workflow_state import WorkflowState
from backend.db.repositories import DerivationRepository, RunRepository
from backend.db.session import SessionLocal, get_db
from backend.memory.short_term import ShortTermMemory
from backend.schemas.api_schemas import DerivationOut, RunCreateRequest, RunSummary
from backend.utils.logging_setup import get_logger

import json
import pandas as pd
import yaml

router = APIRouter(prefix="/runs", tags=["runs"])
_log = get_logger("api.workflows")


def _run_async(run_id: str) -> None:
    """Restore a run from its checkpoint and drive the orchestrator to its
    next stable state (completion, pause, or failure). Runs in a daemon
    thread with its own SQLAlchemy session — keep the HTTP request handler
    short-lived.

    Any uncaught exception marks the run as failed so the frontend can
    surface the problem; the thread never escalates outside its own
    boundary.
    """
    db = SessionLocal()
    try:
        state = ShortTermMemory.restore(db, run_id)
        if not state:
            _log.warning("background_run.no_state", run_id=run_id)
            return
        Orchestrator(db).run_to_completion(state)
    except Exception as exc:  # noqa: BLE001 — last-ditch safety net
        _log.exception("background_run.failed", run_id=run_id, error=str(exc))
        try:
            RunRepository(db).update_status(
                run_id, "failed",
                summary={"crash": f"{type(exc).__name__}: {exc}"},
            )
            db.commit()
        except Exception:  # pragma: no cover — best-effort cleanup
            db.rollback()
    finally:
        db.close()


def _spawn_run(run_id: str) -> None:
    threading.Thread(
        target=_run_async,
        args=(run_id,),
        daemon=True,
        name=f"orchestrator-{run_id}",
    ).start()


@router.post("", response_model=RunSummary)
def create_run(req: RunCreateRequest, db: Session = Depends(get_db)) -> RunSummary:
    if not Path(req.dataset_path).exists():
        raise HTTPException(404, f"Dataset not found: {req.dataset_path}")
    orch = Orchestrator(db)
    state = orch.create_run(
        run_id=req.run_id, spec=req.spec,
        dataset_path=req.dataset_path,
        user_id=req.user_id, notes=req.notes,
    )
    run = orch.runs.get(state.run_id)
    return RunSummary.model_validate(run, from_attributes=True)


@router.post("/{run_id}/start", response_model=RunSummary)
def start_run(run_id: str, db: Session = Depends(get_db)) -> RunSummary:
    state = ShortTermMemory.restore(db, run_id)
    if not state:
        raise HTTPException(404, f"No checkpoint for run {run_id}")
    _spawn_run(run_id)
    # Return the current snapshot; the frontend polls for the live status.
    return RunSummary.model_validate(
        RunRepository(db).get(run_id), from_attributes=True
    )


@router.post("/{run_id}/resume", response_model=RunSummary)
def resume_run(run_id: str, db: Session = Depends(get_db)) -> RunSummary:
    """Resume a run that was interrupted mid-flight.

    The checkpoint mechanism writes state to disk after every phase, so this
    endpoint re-hydrates from the latest checkpoint and re-enters the loop.
    Phases that already produced their characteristic side effect are
    skipped via ``_already_done`` guards in the orchestrator.

    Use cases:
      - Backend process was killed mid-run; DB still says ``running``.
      - Network dropped between a HITL decision and the resume.
      - User wants to pick up a paused run after a long absence.
    """
    run = RunRepository(db).get(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    # Failed runs ARE resumable — the typical cause is a transient
    # Anthropic 5xx / 529 / 429, and the orchestrator checkpoints after
    # every phase, so picking up from the last good state is safe.
    # Completed runs have nothing to resume; rerunning them would just
    # waste tokens and rewrite the audit trail.
    if run.status in ("completed", "completed_with_warnings"):
        raise HTTPException(
            409,
            f"Run {run_id} is already in terminal state '{run.status}'; nothing to resume.",
        )
    state = ShortTermMemory.restore(db, run_id)
    if not state:
        raise HTTPException(404, f"No checkpoint to resume from for run {run_id}")
    if state.hitl_pending:
        raise HTTPException(
            409,
            f"Run {run_id} is awaiting a human decision; resolve via "
            "POST /runs/{run_id}/hitl/decision instead.",
        )
    # Resuming a previously-failed run: flip the persisted status back to
    # "running" so the UI status pill updates immediately and the next
    # poll picks up the orchestrator's progress instead of showing the
    # stale 'failed' label.
    if run.status == "failed":
        RunRepository(db).update_status(run_id, "running")
        db.commit()
    _spawn_run(run_id)
    return RunSummary.model_validate(
        RunRepository(db).get(run_id), from_attributes=True
    )


@router.post("/upload-and-run", response_model=RunSummary)
async def upload_and_run(
    dataset: UploadFile = File(...),
    spec: UploadFile = File(...),
    auto_start: bool = Form(True),
    run_id: str | None = Form(None),
    user_id: str = Form("anonymous"),
    db: Session = Depends(get_db),
) -> RunSummary:
    rid = run_id or uuid.uuid4().hex[:12]
    upload_dir = Path(settings.run_artifact_dir) / rid / "input"
    upload_dir.mkdir(parents=True, exist_ok=True)
    ds_path = upload_dir / dataset.filename
    sp_path = upload_dir / spec.filename
    ds_path.write_bytes(await dataset.read())
    sp_path.write_bytes(await spec.read())

    suffix = sp_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        spec_obj: dict[str, Any] = yaml.safe_load(sp_path.read_text(encoding="utf-8"))
    elif suffix == ".csv":
        # Real pharma specs are often authored as CSV (or Excel) rather than
        # YAML. The CSV adapter converts the row-per-derivation format into
        # the same internal shape downstream agents already understand.
        from backend.utils.csv_spec import CSVSpecError, parse_csv_spec
        try:
            spec_obj = parse_csv_spec(
                sp_path.read_text(encoding="utf-8"),
                default_name=f"CSV spec: {spec.filename}",
            )
        except CSVSpecError as exc:
            raise HTTPException(400, f"CSV spec is invalid: {exc}") from exc
    elif suffix == ".json":
        spec_obj = json.loads(sp_path.read_text(encoding="utf-8"))
    else:
        # Best-effort fall-through; YAML loader handles plain JSON too.
        spec_obj = yaml.safe_load(sp_path.read_text(encoding="utf-8"))

    # Auto-infer source_schema from the dataset when the spec did not
    # declare one (typical for the CSV format, which has no schema block).
    # Without this, the DAG Builder rejects every raw column reference as
    # "unknown" because there is no whitelist to validate against.
    if not spec_obj.get("source_schema"):
        spec_obj["source_schema"] = _infer_schema_from_dataset(ds_path)

    orch = Orchestrator(db)
    state = orch.create_run(
        run_id=rid, spec=spec_obj, dataset_path=str(ds_path),
        user_id=user_id, notes=f"Uploaded {dataset.filename}",
    )
    if auto_start:
        # Pipeline runs in a daemon thread; the HTTP response returns the
        # initial state and the frontend polls for progress.
        _spawn_run(state.run_id)
    return RunSummary.model_validate(orch.runs.get(state.run_id), from_attributes=True)


def _infer_schema_from_dataset(ds_path: Path) -> dict[str, str]:
    """Peek at the dataset and map each column to a coarse internal type.

    The mapping is deliberately conservative — anything that smells like a
    number becomes float, ISO-shaped strings become date, everything else
    stays string. Downstream agents only need this for the DAG Builder's
    'unknown column' check and the LLM's RAW-vs-DERIVED source annotation,
    so precision is not critical; coverage is.
    """
    import pandas as pd

    if ds_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(ds_path)
    else:
        df = pd.read_csv(ds_path, nrows=200)  # sample is enough for type inference

    schema: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            schema[col] = "int"
        elif pd.api.types.is_float_dtype(s):
            schema[col] = "float"
        elif pd.api.types.is_bool_dtype(s):
            schema[col] = "bool"
        else:
            # Object dtype. Sniff for ISO date strings; otherwise treat as
            # plain string. A coerce-to-datetime that succeeds on a healthy
            # majority of non-null cells signals a date column.
            non_null = s.dropna().astype(str)
            if not non_null.empty:
                parsed = pd.to_datetime(non_null.head(20), errors="coerce")
                hit_rate = parsed.notna().mean()
                if hit_rate >= 0.8:
                    schema[col] = "date"
                else:
                    schema[col] = "string"
            else:
                schema[col] = "string"
    return schema


@router.get("", response_model=list[RunSummary])
def list_runs(db: Session = Depends(get_db)) -> list[RunSummary]:
    runs = RunRepository(db).list()
    return [RunSummary.model_validate(r, from_attributes=True) for r in runs]


@router.post("/cleanup-stale")
def cleanup_stale(
    max_age_seconds: int = 600,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Mark runs stuck in ``running`` past ``max_age_seconds`` as failed.

    These are runs whose orchestrator process died mid-flight (e.g. backend
    crash, kill -9) and so never wrote a terminal status. Runs awaiting HITL
    are left alone — they are legitimately paused.
    """
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
    rr = RunRepository(db)
    cleaned: list[str] = []
    for run in rr.list(limit=500):
        if run.status == "running" and (run.updated_at or run.created_at) < cutoff:
            new_summary = dict(run.summary or {})
            new_summary["cleanup_reason"] = "stale_running"
            rr.update_status(run.id, "failed", summary=new_summary)
            cleaned.append(run.id)
    db.commit()
    return {"cleaned": cleaned, "count": len(cleaned), "max_age_seconds": max_age_seconds}


@router.get("/{run_id}", response_model=RunSummary)
def get_run(run_id: str, db: Session = Depends(get_db)) -> RunSummary:
    run = RunRepository(db).get(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    return RunSummary.model_validate(run, from_attributes=True)


@router.get("/{run_id}/state")
def get_state(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    state = ShortTermMemory.restore(db, run_id)
    if not state:
        raise HTTPException(404, f"No state checkpoint for {run_id}")
    return state.to_dict()


@router.get("/{run_id}/derivations", response_model=list[DerivationOut])
def list_derivations(run_id: str, db: Session = Depends(get_db)) -> list[DerivationOut]:
    rows = DerivationRepository(db).list_for_run(run_id)
    return [DerivationOut.model_validate(r, from_attributes=True) for r in rows]


@router.get("/{run_id}/download/{kind}")
def download_artefact(run_id: str, kind: str) -> FileResponse:
    """Stream a run artefact back to the caller. ``kind`` selects:

      - ``csv``       - the analysis-ready table as CSV
      - ``parquet``   - the same data, columnar
      - ``audit_json``- the machine-readable lineage report
      - ``audit_md``  - the human-readable lineage report

    The endpoint is a thin file-server over ``storage/runs/<run_id>/`` so
    the user can pull every byte of a run without leaving the UI.
    """
    base = Path(settings.run_artifact_dir) / run_id
    mapping = {
        "csv":        (base / "output.csv",            "text/csv"),
        "parquet":    (base / "output.parquet",        "application/octet-stream"),
        "audit_json": (base / "audit" / "audit.json",  "application/json"),
        "audit_md":   (base / "audit" / "audit.md",    "text/markdown"),
    }
    if kind not in mapping:
        raise HTTPException(400, f"Unknown artefact kind '{kind}'.")
    path, media_type = mapping[kind]
    if not path.exists():
        raise HTTPException(
            404,
            f"Artefact `{kind}` for run {run_id} not found (run may not "
            "have reached the corresponding phase).",
        )
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=f"{run_id}_{path.name}",
    )


@router.get("/{run_id}/output")
def get_output(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    state = ShortTermMemory.restore(db, run_id)
    if not state or not state.output_path:
        raise HTTPException(404, "No output for this run yet")
    df = pd.read_parquet(state.output_path)
    return {
        "columns": list(df.columns),
        "rows": df.head(200).to_dict(orient="records"),
        "row_count": len(df),
        "preview_size": min(200, len(df)),
        "output_path": state.output_path,
    }
