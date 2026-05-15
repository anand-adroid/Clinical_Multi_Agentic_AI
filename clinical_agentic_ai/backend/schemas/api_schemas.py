"""Pydantic request/response schemas for the FastAPI surface."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- runs
class RunCreateRequest(BaseModel):
    run_id: str
    spec: dict[str, Any]
    dataset_path: str = Field(..., description="Path to the input CSV/Parquet (server-local)")
    user_id: str = "anonymous"
    notes: Optional[str] = None


class RunSummary(BaseModel):
    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    spec_hash: str
    dataset_hash: str
    user_id: str
    summary: Optional[dict[str, Any]] = None


class HITLDecisionRequest(BaseModel):
    reviewer: str
    action: str  # approve | edit | reject | regenerate
    target: Optional[str] = None
    comment: Optional[str] = None
    edited_code: Optional[str] = None
    # Phase 1: structured per-question answers for spec-review HITL.
    # Maps derivation name -> reviewer's clarifying answer.
    clarification_answers: Optional[dict[str, str]] = None
    # Phase 3: per-derivation overrides for the code-preapproval HITL.
    # Each value: {"action": "approve"|"edit"|"regenerate", "code": "...", "hint": "..."}
    derivation_overrides: Optional[dict[str, dict[str, Any]]] = None
    # Phase 3: optional hint shown to the LLM when ``action="regenerate"``.
    regenerate_hint: Optional[str] = None


class EventOut(BaseModel):
    id: int
    agent: str
    step: str
    status: str
    duration_ms: int
    inputs_hash: Optional[str]
    outputs_hash: Optional[str]
    detail: Optional[dict[str, Any]]
    created_at: datetime


class ValidationOut(BaseModel):
    rule_id: str
    target: Optional[str]
    severity: str
    passed: bool
    message: str
    detail: Optional[dict[str, Any]]
    created_at: datetime


class DerivationOut(BaseModel):
    target: str
    sources: list[str]
    rule_text: str
    code: str
    code_hash: str
    generator: str
    attempt: int
    status: str
    created_at: datetime


class HITLOut(BaseModel):
    id: int
    target: Optional[str]
    reviewer: str
    action: str
    comment: Optional[str]
    edited_code: Optional[str]
    created_at: datetime


class AuditOut(BaseModel):
    id: int
    actor: str
    actor_type: str
    action: str
    object_ref: Optional[str]
    detail: Optional[dict[str, Any]]
    created_at: datetime


class MemoryPatternOut(BaseModel):
    id: int
    target: str
    signature: str
    rule_text: str
    code: str
    sources: list[str]
    score: float
    times_used: int


class EvalResultOut(BaseModel):
    correctness: float
    coverage: float
    reliability: dict[str, Any]
    per_target: dict[str, Any]
