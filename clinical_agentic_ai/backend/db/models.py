"""
SQLAlchemy models.

These tables together form the audit substrate. Each row is immutable once
written; the orchestrator never UPDATEs state, it APPENDs new events. That
discipline is what lets the system claim reproducibility: every artifact
(spec, dataset, code, output, decision) points back at the inputs that
produced it via stable hashes.

Tables
------
- ``runs``              — one row per workflow execution
- ``agent_events``      — append-only log of every agent step
- ``derivations``       — one row per target per run (code + hash + status)
- ``validations``       — one row per validation rule outcome
- ``hitl_decisions``    — every human review decision
- ``audit_entries``     — cross-cutting audit ledger
- ``memory_patterns``   — long-term reusable derivation patterns
- ``clarification_memory`` — long-term reviewer answers to spec ambiguities

The schema is deliberately simple and append-only. In production this would
sit on managed Postgres with a parallel WORM bucket (S3 Object Lock or
equivalent) for tamper evidence — the table shape stays the same.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.utcnow()


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    status: Mapped[str] = mapped_column(String(32), index=True)  # created, running, awaiting_hitl, completed, failed
    spec_hash: Mapped[str] = mapped_column(String(64), index=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), default="anonymous")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    events: Mapped[list["AgentEvent"]] = relationship(back_populates="run", cascade="all,delete")
    derivations: Mapped[list["Derivation"]] = relationship(back_populates="run", cascade="all,delete")
    validations: Mapped[list["Validation"]] = relationship(back_populates="run", cascade="all,delete")
    hitl: Mapped[list["HITLDecision"]] = relationship(back_populates="run", cascade="all,delete")
    audit: Mapped[list["AuditEntry"]] = relationship(back_populates="run", cascade="all,delete")


class AgentEvent(Base):
    """Append-only event log — one row per agent action."""

    __tablename__ = "agent_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    agent: Mapped[str] = mapped_column(String(64), index=True)
    step: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32))  # started, ok, warn, failed
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    inputs_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    outputs_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    run: Mapped[Run] = relationship(back_populates="events")

    __table_args__ = (
        Index("ix_events_run_agent_step", "run_id", "agent", "step"),
    )


class Derivation(Base):
    """One row per *target variable* in a run. Records the *exact* code that
    produced its values, plus the source columns it depends on."""

    __tablename__ = "derivations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    target: Mapped[str] = mapped_column(String(128), index=True)
    sources: Mapped[list[str]] = mapped_column(JSON)
    rule_text: Mapped[str] = mapped_column(Text)
    code: Mapped[str] = mapped_column(Text)
    code_hash: Mapped[str] = mapped_column(String(64), index=True)
    output_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    generator: Mapped[str] = mapped_column(String(32))  # "llm" | "rule" | "memory" | "human"
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending, ok, failed, refined
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    run: Mapped[Run] = relationship(back_populates="derivations")


class Validation(Base):
    __tablename__ = "validations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    target: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    rule_id: Mapped[str] = mapped_column(String(128), index=True)
    severity: Mapped[str] = mapped_column(String(16))  # info | warn | block
    passed: Mapped[bool] = mapped_column()
    message: Mapped[str] = mapped_column(Text)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    run: Mapped[Run] = relationship(back_populates="validations")


class HITLDecision(Base):
    __tablename__ = "hitl_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    target: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    reviewer: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(32))  # approve | edit | reject
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    edited_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    run: Mapped[Run] = relationship(back_populates="hitl")


class AuditEntry(Base):
    """Cross-cutting audit ledger — every step an auditor or regulator
    might want to see in chronological order, even if the same fact also
    lives in ``agent_events``. Keeping it separate means the audit ledger
    can be shipped to a long-term immutable store without leaking
    implementation detail from the operational tables."""

    __tablename__ = "audit_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    actor: Mapped[str] = mapped_column(String(64))    # agent name or human id
    actor_type: Mapped[str] = mapped_column(String(16))  # "agent" | "human" | "system"
    action: Mapped[str] = mapped_column(String(64))
    object_ref: Mapped[str | None] = mapped_column(String(256), nullable=True)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    run: Mapped[Run] = relationship(back_populates="audit")


class MemoryPattern(Base):
    """Long-term memory — validated, reusable derivation patterns.

    ``reasoning`` carries the LLM's original explanation captured at the
    time the pattern was first promoted. On a subsequent memory hit the
    reviewer sees that original reasoning — not a meta-note about reuse —
    so the audit trail explains WHY the code is correct in every run.
    """

    __tablename__ = "memory_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signature: Mapped[str] = mapped_column(String(128), index=True)
    target: Mapped[str] = mapped_column(String(128), index=True)
    rule_text: Mapped[str] = mapped_column(Text)
    code: Mapped[str] = mapped_column(Text)
    sources: Mapped[list[str]] = mapped_column(JSON)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=1.0)
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    last_used: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ClarificationMemory(Base):
    """Phase 2: long-term memory of resolved spec-review clarifications.

    Stores the human-authored answer for each (target, issue) pair so future
    runs can pre-fill the HITL form. Auditors can also query this table to
    see how each kind of ambiguity has historically been resolved.
    """

    __tablename__ = "clarification_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signature: Mapped[str] = mapped_column(String(128), index=True)
    target: Mapped[str] = mapped_column(String(128), index=True)
    issue: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    reviewer: Mapped[str] = mapped_column(String(128), default="system")
    score: Mapped[float] = mapped_column(Float, default=1.0)
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    last_used: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
