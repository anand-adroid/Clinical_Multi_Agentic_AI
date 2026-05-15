"""
Short-Term Memory (STM)
-----------------------

What it is
~~~~~~~~~~
The *working memory* of a single workflow run. It holds:

* the current `WorkflowState` (DAG, derivations-in-progress, intermediate
  outputs),
* artefact hashes,
* pending HITL request,
* the most recent agent step.

How it differs from LTM
~~~~~~~~~~~~~~~~~~~~~~~
STM is **scoped to a run**, **mutable**, and **discardable** at the end of the
run. LTM (see `long_term.py`) is **cross-run**, **append-mostly**, and
**reusable** between studies.

How it's stored
~~~~~~~~~~~~~~~
STM is materialised in two complementary places:
  1. Disk checkpoints (`backend.core.checkpoint`) — the source of truth for
     restoring after a crash or after a HITL pause.
  2. SQL audit tables (`agent_events`, `derivations`, ...) — the source of
     truth for human/regulator inspection.

The class below is a thin facade so agents don't have to know about either.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.core import checkpoint
from backend.core.workflow_state import StepRecord, WorkflowState
from backend.db.repositories import (
    AuditRepository,
    DerivationRepository,
    EventRepository,
    HITLRepository,
    RunRepository,
    ValidationRepository,
)
from backend.utils.logging_setup import get_logger
from sqlalchemy.orm import Session

log = get_logger("memory.stm")


class ShortTermMemory:
    def __init__(self, db: Session, state: WorkflowState) -> None:
        self.db = db
        self.state = state
        self.runs = RunRepository(db)
        self.events = EventRepository(db)
        self.derivs = DerivationRepository(db)
        self.vals = ValidationRepository(db)
        self.hitl = HITLRepository(db)
        self.audit = AuditRepository(db)

    # ----------------------------- events ---------------------------------
    def begin_step(self, agent: str, step: str) -> StepRecord:
        rec = StepRecord(
            agent=agent, step=step, status="started",
            started_at=datetime.utcnow().isoformat(),
        )
        self.state.events.append(rec)
        self.events.record(run_id=self.state.run_id, agent=agent, step=step, status="started")
        self.audit.record(
            run_id=self.state.run_id, actor=agent, actor_type="agent",
            action=f"begin:{step}",
        )
        return rec

    def end_step(self, rec: StepRecord, *, status: str = "ok",
                 inputs_hash: str | None = None,
                 outputs_hash: str | None = None,
                 detail: dict[str, Any] | None = None) -> None:
        rec.status = status
        rec.finished_at = datetime.utcnow().isoformat()
        rec.inputs_hash = inputs_hash
        rec.outputs_hash = outputs_hash
        if detail:
            rec.detail.update(detail)
        try:
            rec.duration_ms = int(
                (datetime.fromisoformat(rec.finished_at)
                 - datetime.fromisoformat(rec.started_at)).total_seconds() * 1000
            )
        except Exception:
            rec.duration_ms = 0
        self.events.record(
            run_id=self.state.run_id, agent=rec.agent, step=rec.step,
            status=status, duration_ms=rec.duration_ms,
            inputs_hash=inputs_hash, outputs_hash=outputs_hash, detail=detail,
        )
        self.audit.record(
            run_id=self.state.run_id, actor=rec.agent, actor_type="agent",
            action=f"end:{rec.step}", detail={"status": status, "duration_ms": rec.duration_ms},
        )

    # ----------------------------- checkpoints ----------------------------
    def checkpoint(self, step_id: str) -> None:
        checkpoint.save(self.state.run_id, step_id, self.state.to_dict())

    @staticmethod
    def restore(db: Session, run_id: str) -> WorkflowState | None:
        latest = checkpoint.latest(run_id)
        if not latest:
            return None
        _step, blob = latest
        return WorkflowState.from_dict(blob)
