"""
WorkflowState — the in-memory object that flows through every agent.

It is both the short-term memory of a run and the unit that gets
checkpointed to disk. Each agent receives it, mutates a clearly demarcated
section, appends to its event log, and hands it on.

Design rules I follow throughout this codebase:
  - Every field is JSON-serialisable. State must survive ``json.dumps`` so
    checkpoints are diffable and inspectable.
  - Nothing is ever removed; sections only grow. That gives natural lineage
    and means an auditor can reconstruct the run at any phase.
  - DataFrames live outside the state. Hashes go here, bytes go on disk
    under ``storage/runs/<run_id>/``. Keeping pandas frames out of the
    checkpointable struct keeps each checkpoint small and serialisation cheap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StepRecord:
    agent: str
    step: str
    status: str  # started | ok | warn | failed
    started_at: str
    finished_at: str | None = None
    duration_ms: int = 0
    inputs_hash: str | None = None
    outputs_hash: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class DerivationRecord:
    target: str
    sources: list[str]
    rule_text: str
    code: str
    code_hash: str
    generator: str
    attempt: int = 1
    status: str = "pending"
    output_hash: str | None = None
    null_count: int = 0
    row_errors: list[dict[str, Any]] = field(default_factory=list)
    # Phase 5: LLM self-reported confidence (0.0-1.0). None means "unknown".
    confidence: float | None = None
    # Phase 5: human-readable explanation of any uncertainty.
    uncertainty_notes: str | None = None
    # Reasoning trace: LLM's explanation of WHY it produced this code. This is
    # what makes the system genuinely agentic and not just a workflow — each
    # agent decision carries its own justification into the audit trail.
    reasoning: str | None = None
    # Phase 1.3: risk class inherited from the spec; drives gate routing.
    risk_class: str = "routine"


@dataclass
class ValidationRecord:
    rule_id: str
    target: str | None
    severity: str
    passed: bool
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class HITLRequest:
    target: str | None
    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    raised_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class DerivationPolicy:
    """Per-derivation execution policy produced by the Planner Agent.

    The Planner is what makes this system *agentic*: an LLM reads the spec
    and decides — per derivation — how cautious the pipeline should be.
    Subsequent agents consult these policies instead of using global flags.
    """
    target: str
    risk_class: str = "routine"             # routine | critical | regulatory_critical | exploratory
    require_preapproval: bool = False        # force code preapproval HITL
    require_signoff: bool = False            # require G4 sign-off before release
    confidence_threshold: float = 0.7        # min LLM confidence to skip HITL
    reviewer_tier: str = "T1"                # T1 | T2 | T3 — required reviewer expertise
    rationale: str = ""                      # planner's reasoning for this policy


@dataclass
class RunPlan:
    """The hashed run plan produced by the Planner Agent. Stored on state
    and embedded in the audit report — auditors can prove that ``run X used
    plan Y to choose which gates to apply``."""
    policies: dict[str, DerivationPolicy] = field(default_factory=dict)
    strategy: str = "balanced"               # cautious | balanced | fast
    rationale: str = ""
    plan_hash: str = ""


@dataclass
class WorkflowState:
    run_id: str
    spec: dict[str, Any]
    spec_hash: str
    dataset_hash: str
    dataset_path: str
    # Phase 1: hash of the spec *after* human-mediated disambiguation. Equals
    # ``spec_hash`` if no clarifications were resolved. Auditors use this to
    # prove which spec version actually produced the output.
    resolved_spec_hash: str | None = None
    # Planner-produced run plan (Priority 1.3). Decides per-derivation policy
    # for HITL gating, reviewer tier requirements, and confidence thresholds.
    plan: RunPlan | None = None
    output_path: str | None = None
    status: str = "created"
    topo_order: list[str] = field(default_factory=list)
    dag: dict[str, list[str]] = field(default_factory=dict)  # target -> sources
    derivations: dict[str, DerivationRecord] = field(default_factory=dict)
    validations: list[ValidationRecord] = field(default_factory=list)
    hitl_pending: HITLRequest | None = None
    hitl_history: list[dict[str, Any]] = field(default_factory=list)
    events: list[StepRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    # ---- serialisation ----
    def to_dict(self) -> dict[str, Any]:
        def _conv(o: Any) -> Any:
            if hasattr(o, "__dataclass_fields__"):
                return {k: _conv(getattr(o, k)) for k in o.__dataclass_fields__}
            if isinstance(o, list):
                return [_conv(x) for x in o]
            if isinstance(o, dict):
                return {k: _conv(v) for k, v in o.items()}
            return o
        return _conv(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowState":
        ws = cls(
            run_id=d["run_id"],
            spec=d["spec"],
            spec_hash=d["spec_hash"],
            dataset_hash=d["dataset_hash"],
            dataset_path=d["dataset_path"],
            resolved_spec_hash=d.get("resolved_spec_hash"),
            output_path=d.get("output_path"),
            status=d.get("status", "created"),
            topo_order=list(d.get("topo_order", [])),
            dag=dict(d.get("dag", {})),
            summary=dict(d.get("summary", {})),
            config_snapshot=dict(d.get("config_snapshot", {})),
        )
        for tgt, dr in (d.get("derivations") or {}).items():
            ws.derivations[tgt] = DerivationRecord(**dr)
        for v in (d.get("validations") or []):
            ws.validations.append(ValidationRecord(**v))
        for e in (d.get("events") or []):
            ws.events.append(StepRecord(**e))
        h = d.get("hitl_pending")
        if h:
            ws.hitl_pending = HITLRequest(**h)
        plan_d = d.get("plan")
        if plan_d:
            policies = {
                k: DerivationPolicy(**v)
                for k, v in (plan_d.get("policies") or {}).items()
            }
            ws.plan = RunPlan(
                policies=policies,
                strategy=plan_d.get("strategy", "balanced"),
                rationale=plan_d.get("rationale", ""),
                plan_hash=plan_d.get("plan_hash", ""),
            )
        ws.hitl_history = list(d.get("hitl_history") or [])
        return ws
