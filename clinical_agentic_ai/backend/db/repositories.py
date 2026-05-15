"""
Repositories — every write to the audit substrate goes through here.

Why repositories (instead of agents writing raw SQL)?
  * Centralised invariants: e.g. `record_event` always stamps the agent name
    and step id, so no agent can forget to.
  * Mocking surface for tests: agents take a repo, not a Session.
  - One place to swap SQLite -> Postgres -> cloud-managed audit store.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

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


class RunRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, *, run_id: str, spec_hash: str, dataset_hash: str,
               user_id: str = "anonymous", notes: str | None = None) -> Run:
        run = Run(
            id=run_id, status="created",
            spec_hash=spec_hash, dataset_hash=dataset_hash,
            user_id=user_id, notes=notes,
        )
        self.db.add(run)
        self.db.flush()
        return run

    def get(self, run_id: str) -> Run | None:
        return self.db.get(Run, run_id)

    def update_status(self, run_id: str, status: str,
                      summary: dict[str, Any] | None = None) -> None:
        run = self.db.get(Run, run_id)
        if not run:
            return
        run.status = status
        run.updated_at = datetime.utcnow()
        if summary is not None:
            run.summary = summary
        self.db.flush()

    def list(self, limit: int = 50) -> list[Run]:
        return list(
            self.db.query(Run).order_by(Run.created_at.desc()).limit(limit).all()
        )


class EventRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def record(self, *, run_id: str, agent: str, step: str, status: str,
               duration_ms: int = 0, inputs_hash: str | None = None,
               outputs_hash: str | None = None,
               detail: dict[str, Any] | None = None) -> AgentEvent:
        ev = AgentEvent(
            run_id=run_id, agent=agent, step=step, status=status,
            duration_ms=duration_ms, inputs_hash=inputs_hash,
            outputs_hash=outputs_hash, detail=detail,
        )
        self.db.add(ev)
        self.db.flush()
        return ev

    def list_for_run(self, run_id: str) -> list[AgentEvent]:
        return list(
            self.db.query(AgentEvent)
            .filter(AgentEvent.run_id == run_id)
            .order_by(AgentEvent.created_at.asc(), AgentEvent.id.asc())
            .all()
        )


class DerivationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert(self, *, run_id: str, target: str, sources: list[str],
               rule_text: str, code: str, code_hash: str,
               generator: str, attempt: int, status: str,
               output_hash: str | None = None) -> Derivation:
        d = Derivation(
            run_id=run_id, target=target, sources=sources,
            rule_text=rule_text, code=code, code_hash=code_hash,
            generator=generator, attempt=attempt, status=status,
            output_hash=output_hash,
        )
        self.db.add(d)
        self.db.flush()
        return d

    def list_for_run(self, run_id: str) -> list[Derivation]:
        return list(
            self.db.query(Derivation)
            .filter(Derivation.run_id == run_id)
            .order_by(Derivation.created_at.asc(), Derivation.id.asc())
            .all()
        )


class ValidationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def record(self, *, run_id: str, target: str | None, rule_id: str,
               severity: str, passed: bool, message: str,
               detail: dict[str, Any] | None = None) -> Validation:
        v = Validation(
            run_id=run_id, target=target, rule_id=rule_id,
            severity=severity, passed=passed, message=message, detail=detail,
        )
        self.db.add(v)
        self.db.flush()
        return v

    def list_for_run(self, run_id: str) -> list[Validation]:
        return list(
            self.db.query(Validation).filter(Validation.run_id == run_id).all()
        )


class HITLRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def record(self, *, run_id: str, target: str | None, reviewer: str,
               action: str, comment: str | None = None,
               edited_code: str | None = None) -> HITLDecision:
        h = HITLDecision(
            run_id=run_id, target=target, reviewer=reviewer,
            action=action, comment=comment, edited_code=edited_code,
        )
        self.db.add(h)
        self.db.flush()
        return h

    def list_for_run(self, run_id: str) -> list[HITLDecision]:
        return list(
            self.db.query(HITLDecision)
            .filter(HITLDecision.run_id == run_id)
            .order_by(HITLDecision.created_at.asc())
            .all()
        )


class AuditRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def record(self, *, run_id: str, actor: str, actor_type: str,
               action: str, object_ref: str | None = None,
               detail: dict[str, Any] | None = None) -> AuditEntry:
        a = AuditEntry(
            run_id=run_id, actor=actor, actor_type=actor_type,
            action=action, object_ref=object_ref, detail=detail,
        )
        self.db.add(a)
        self.db.flush()
        return a

    def list_for_run(self, run_id: str) -> list[AuditEntry]:
        return list(
            self.db.query(AuditEntry)
            .filter(AuditEntry.run_id == run_id)
            .order_by(AuditEntry.created_at.asc(), AuditEntry.id.asc())
            .all()
        )


class MemoryRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add_pattern(self, *, signature: str, target: str, rule_text: str,
                    code: str, sources: list[str], created_by: str = "system",
                    reasoning: str | None = None) -> MemoryPattern:
        # If an identical signature/target/code triple exists, bump its score
        # instead of inserting a duplicate. Reasoning is updated only when
        # the existing entry has none (first capture wins; reviewer edits
        # produce fresh patterns via the human path).
        existing = (
            self.db.query(MemoryPattern)
            .filter(MemoryPattern.signature == signature,
                    MemoryPattern.target == target,
                    MemoryPattern.code == code)
            .first()
        )
        if existing:
            existing.score += 0.5
            existing.times_used += 1
            existing.last_used = datetime.utcnow()
            if reasoning and not existing.reasoning:
                existing.reasoning = reasoning
            self.db.flush()
            return existing
        p = MemoryPattern(
            signature=signature, target=target, rule_text=rule_text,
            code=code, sources=sources, created_by=created_by, score=1.0,
            reasoning=reasoning,
        )
        self.db.add(p)
        self.db.flush()
        return p

    def find_for(self, signature: str, target: str) -> MemoryPattern | None:
        return (
            self.db.query(MemoryPattern)
            .filter(MemoryPattern.signature == signature,
                    MemoryPattern.target == target)
            .order_by(MemoryPattern.score.desc(), MemoryPattern.times_used.desc())
            .first()
        )

    def mark_used(self, pattern_id: int) -> None:
        p = self.db.get(MemoryPattern, pattern_id)
        if p:
            p.times_used += 1
            p.last_used = datetime.utcnow()
            p.score += 0.25
            self.db.flush()

    def list(self, limit: int = 100) -> list[MemoryPattern]:
        return list(
            self.db.query(MemoryPattern)
            .order_by(MemoryPattern.score.desc(), MemoryPattern.id.desc())
            .limit(limit)
            .all()
        )


class ClarificationMemoryRepository:
    """Phase 2: persistence layer for resolved spec-review clarifications."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert(self, *, signature: str, target: str, issue: str, answer: str,
               reviewer: str) -> ClarificationMemory:
        existing = (
            self.db.query(ClarificationMemory)
            .filter(
                ClarificationMemory.signature == signature,
                ClarificationMemory.target == target,
                ClarificationMemory.answer == answer,
            )
            .first()
        )
        if existing:
            existing.times_used += 1
            existing.score += 0.5
            existing.last_used = datetime.utcnow()
            self.db.flush()
            return existing
        row = ClarificationMemory(
            signature=signature, target=target, issue=issue,
            answer=answer, reviewer=reviewer, score=1.0, times_used=0,
        )
        self.db.add(row)
        self.db.flush()
        return row

    def find_for(self, *, signature: str, target: str) -> ClarificationMemory | None:
        return (
            self.db.query(ClarificationMemory)
            .filter(
                ClarificationMemory.signature == signature,
                ClarificationMemory.target == target,
            )
            .order_by(ClarificationMemory.score.desc(),
                      ClarificationMemory.times_used.desc())
            .first()
        )

    def mark_used(self, row_id: int) -> None:
        row = self.db.get(ClarificationMemory, row_id)
        if row:
            row.times_used += 1
            row.score += 0.25
            row.last_used = datetime.utcnow()
            self.db.flush()

    def list(self, limit: int = 100) -> list[ClarificationMemory]:
        return list(
            self.db.query(ClarificationMemory)
            .order_by(ClarificationMemory.score.desc(),
                      ClarificationMemory.id.desc())
            .limit(limit)
            .all()
        )
