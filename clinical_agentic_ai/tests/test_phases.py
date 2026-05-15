"""Phase 0-5 integration tests.

Each test exercises a single phase's distinctive behaviour. They run against
the real Orchestrator + SQLite DB but stub the LLM to keep things
deterministic.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import yaml

from backend.core.orchestrator import Orchestrator
from backend.db.session import session_scope

from tests._helpers import (
    auto_approve_spec_clarifications,
    run_to_completion_auto_approving,
)


def _load_spec(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
#  Phase 0 — test_runner module                                               #
# --------------------------------------------------------------------------- #

def test_phase0_test_runner_module_basics():
    from backend.core.test_runner import run_test_cases

    code = (
        "def derive(row):\n"
        "    age = to_int(row['age'])\n"
        "    if age is None: return None\n"
        "    if age < 18: return '<18'\n"
        "    if age < 65: return '18-64'\n"
        "    return '>=65'\n"
    )
    cases = [
        {"input": {"age": 12}, "expected": "<18"},
        {"input": {"age": 65}, "expected": ">=65"},
        {"input": {"age": None}, "expected": None},
        {"input": {"age": 30}, "expected": "WRONG"},  # deliberate fail
    ]
    r = run_test_cases(code, "AGE_GROUP", cases)
    assert r.total == 4
    assert r.passed_count == 3
    assert r.failed_cases[0].case_index == 3


def test_phase0_failed_test_marks_derivation_failed(sample_paths, stub_llm):
    """A spec with a deliberately wrong expected value should drive the
    pipeline into the refiner loop because the test gate fails. The
    regulatory_critical preapproval pause is auto-approved first so we reach
    the test-cases gate."""
    spec = _load_spec(sample_paths["spec"])
    # Patch AGE_GROUP's first test case to expect a wrong value.
    for d in spec["derivations"]:
        if d["name"] == "AGE_GROUP":
            d["test_cases"][0] = {"input": {"age": 12}, "expected": "WRONG"}

    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec, dataset_path=str(sample_paths["dataset"]),
        )
        final = run_to_completion_auto_approving(orch, state)

    # After the refiner exhausts retries (it will, because stub keeps returning
    # the same code), the run pauses for HITL. It may also pause earlier at
    # preapproval for the regulatory_critical derivation — either is a valid
    # outcome for this test (the point is just that test failures *register*).
    assert final.status in (
        "awaiting_hitl_refine",
        "awaiting_hitl_preapproval",
        "completed_with_warnings",
        "completed",
    )
    tc_validations = [v for v in final.validations if v.rule_id.startswith("TEST_CASE_")]
    assert any(not v.passed for v in tc_validations), \
        "expected at least one failing test case"


# --------------------------------------------------------------------------- #
#  Phase 1 — spec-review HITL with structured answers                         #
# --------------------------------------------------------------------------- #

def test_phase1_clarification_answers_fold_into_rule(sample_paths, monkeypatch):
    """Stub the spec reviewer to ALWAYS raise a clarification; verify the
    reviewer's answer ends up in the normalised rule and resolved_spec_hash
    is updated."""
    from backend.core.llm_client import LLMResponse, llm

    spec = _load_spec(sample_paths["spec"])

    def _fake(*, system, user, expect_json=True, purpose="completion", **kw):  # noqa: ARG001
        if "spec reviewer" in system.lower():
            normalised = []
            for d in spec["derivations"]:
                n = dict(d)
                n.setdefault("max_null_rate", 0.5)
                normalised.append(n)
            return LLMResponse(
                text="{}",
                parsed={
                    "clarifications": [
                        {"name": "AGE_GROUP", "issue": "thresholds ambiguous",
                         "suggested_question": "Use inclusive bound at 65?"},
                    ],
                    "normalised_derivations": normalised,
                },
                tokens_in=0, tokens_out=0, latency_ms=0,
                model="stub", backend="stub",
            )
        return LLMResponse(
            text="{}", parsed={"code": "", "confidence": 0.0}, tokens_in=0,
            tokens_out=0, latency_ms=0, model="stub", backend="stub",
        )

    monkeypatch.setattr(llm, "enabled", True)
    monkeypatch.setattr(llm, "complete", _fake)

    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec, dataset_path=str(sample_paths["dataset"]),
        )
        state = orch.run_to_completion(state)
        assert state.status == "awaiting_hitl_spec"
        assert state.hitl_pending is not None

        # Apply structured answer
        state = orch.apply_hitl_decision(
            run_id=run_id,
            reviewer="tester",
            action="approve",
            target=None,
            clarification_answers={"AGE_GROUP": "Use inclusive >= 65."},
        )

    # The clarification answer should be folded into AGE_GROUP's rule.
    found_rule = next(
        d["rule"] for d in state.spec["normalised_derivations"]
        if d["name"] == "AGE_GROUP"
    )
    assert "Clarified by reviewer" in found_rule
    assert "Use inclusive >= 65." in found_rule
    assert state.resolved_spec_hash is not None
    assert state.resolved_spec_hash != state.spec_hash


# --------------------------------------------------------------------------- #
#  Phase 2 — clarification memory persists and pre-fills                      #
# --------------------------------------------------------------------------- #

def test_phase2_clarification_memory_roundtrip():
    from backend.memory.long_term import LongTermMemory

    with session_scope() as db:
        ltm = LongTermMemory(db)
        ltm.remember_clarification(
            target="AGE_GROUP",
            issue="thresholds ambiguous",
            answer="Use inclusive >= 65.",
            reviewer="alice",
        )

    with session_scope() as db:
        ltm = LongTermMemory(db)
        hit = ltm.lookup_clarification(
            target="AGE_GROUP", issue="thresholds ambiguous"
        )
    assert hit is not None
    assert hit.answer == "Use inclusive >= 65."
    assert hit.target == "AGE_GROUP"


# --------------------------------------------------------------------------- #
#  Phase 3 — code preapproval gate triggers when policy flag is on            #
# --------------------------------------------------------------------------- #

def test_phase3_preapproval_pauses_when_flag_on(sample_paths, stub_llm, monkeypatch):
    from backend.core import config as cfg

    monkeypatch.setattr(cfg.settings, "require_code_preapproval", True)

    spec = _load_spec(sample_paths["spec"])
    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec, dataset_path=str(sample_paths["dataset"]),
        )
        state = orch.run_to_completion(state)
        # The comprehensive demo spec includes an ambiguous derivation that
        # triggers the spec clarifications gate first; auto-approve past it
        # so the assertion below targets the preapproval gate.
        state = auto_approve_spec_clarifications(orch, state)

    assert state.status == "awaiting_hitl_preapproval"
    assert state.hitl_pending is not None
    assert state.hitl_pending.reason == "code_preapproval_required"
    ctx = state.hitl_pending.context or {}
    assert len(ctx.get("preapproval_targets") or []) == 7  # seven derivations
    for t in ctx["preapproval_targets"]:
        assert "preview" in t


def test_phase3_approve_overrides_then_resumes_to_completion(
    sample_paths, stub_llm, monkeypatch,
):
    from backend.core import config as cfg

    monkeypatch.setattr(cfg.settings, "require_code_preapproval", True)

    spec = _load_spec(sample_paths["spec"])
    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec, dataset_path=str(sample_paths["dataset"]),
        )
        state = orch.run_to_completion(state)
        state = auto_approve_spec_clarifications(orch, state)
        assert state.status == "awaiting_hitl_preapproval"

        overrides = {
            t["target"]: {"action": "approve"}
            for t in (state.hitl_pending.context or {})["preapproval_targets"]
        }
        state = orch.apply_hitl_decision(
            run_id=run_id, reviewer="tester", action="approve", target=None,
            derivation_overrides=overrides,
        )

    assert state.status.startswith("completed"), f"status was {state.status}"


# --------------------------------------------------------------------------- #
#  Phase 4 — no LLM, no fallback => escalation to HITL                        #
# --------------------------------------------------------------------------- #

def test_phase4_offline_no_llm_escalates_to_hitl(sample_paths, monkeypatch, clean_memory):
    from backend.core.llm_client import llm

    monkeypatch.setattr(llm, "enabled", False)

    spec = _load_spec(sample_paths["spec"])
    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec, dataset_path=str(sample_paths["dataset"]),
        )
        state = orch.run_to_completion(state)
        # Heuristic review (LLM disabled) raises a clarification for the
        # ambiguous COHORT_LABEL derivation; auto-approve past that gate so
        # we reach the codegen-failed gate this test is verifying.
        state = auto_approve_spec_clarifications(orch, state)

    assert state.status == "awaiting_hitl_codegen"
    assert state.hitl_pending is not None
    assert state.hitl_pending.reason == "codegen_failed"
    blocks = [v for v in state.validations if v.rule_id == "CODEGEN_NO_OUTPUT"]
    assert len(blocks) == len(spec["derivations"])


# --------------------------------------------------------------------------- #
#  Phase 5 — low LLM confidence triggers early HITL                           #
# --------------------------------------------------------------------------- #

def test_phase5_low_confidence_triggers_preapproval(
    sample_paths, low_confidence_llm, monkeypatch, clean_memory,
):
    from backend.core import config as cfg

    monkeypatch.setattr(cfg.settings, "require_code_preapproval", False)
    monkeypatch.setattr(cfg.settings, "min_confidence_threshold", 0.7)

    spec = _load_spec(sample_paths["spec"])
    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec, dataset_path=str(sample_paths["dataset"]),
        )
        state = orch.run_to_completion(state)
        state = auto_approve_spec_clarifications(orch, state)

    assert state.status == "awaiting_hitl_preapproval"
    ctx = state.hitl_pending.context or {}
    low = ctx.get("low_confidence_targets") or []
    assert len(low) >= 1
    # Every derivation should have confidence recorded.
    for d in state.derivations.values():
        assert d.confidence is not None
        assert d.confidence < 0.7


def test_planner_deterministic_uses_declared_risk_class(sample_paths, monkeypatch, clean_memory):
    """When LLM is off, the Planner falls back to risk-class defaults from
    the spec. regulatory_critical => require_preapproval=True; routine =>
    require_preapproval=False."""
    from backend.core.llm_client import llm

    monkeypatch.setattr(llm, "enabled", False)

    spec = _load_spec(sample_paths["spec"])
    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec, dataset_path=str(sample_paths["dataset"]),
        )
        state = orch.run_to_completion(state)
        state = auto_approve_spec_clarifications(orch, state)

    # Run pauses at code_generate (Phase 4) because LLM is off; that's fine.
    # We just need to assert the planner produced a plan in the meantime.
    assert state.plan is not None, "planner should always produce a plan"
    assert state.plan.plan_hash, "plan should be hashed"

    policies = state.plan.policies
    # AGE_GROUP is declared `routine` in the sample spec
    assert policies["AGE_GROUP"].risk_class == "routine"
    assert policies["AGE_GROUP"].require_preapproval is False
    # COMPOSITE_RISK_TIER is declared `regulatory_critical`
    assert policies["COMPOSITE_RISK_TIER"].risk_class == "regulatory_critical"
    assert policies["COMPOSITE_RISK_TIER"].require_preapproval is True
    assert policies["COMPOSITE_RISK_TIER"].require_signoff is True


def test_planner_policy_forces_preapproval_even_at_high_confidence(
    sample_paths, stub_llm, monkeypatch, clean_memory,
):
    """A derivation flagged regulatory_critical by the planner must trigger
    code preapproval even though stub_llm returns confidence 0.95."""
    from backend.core import config as cfg

    monkeypatch.setattr(cfg.settings, "require_code_preapproval", False)

    spec = _load_spec(sample_paths["spec"])
    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec, dataset_path=str(sample_paths["dataset"]),
        )
        state = orch.run_to_completion(state)
        state = auto_approve_spec_clarifications(orch, state)

    # COMPOSITE_RISK_TIER is regulatory_critical => policy_forced preapproval
    assert state.status == "awaiting_hitl_preapproval"
    ctx = state.hitl_pending.context or {}
    forced = ctx.get("policy_forced_targets") or []
    assert "COMPOSITE_RISK_TIER" in forced


def test_phase5_high_confidence_proceeds_without_pause(
    sample_paths, stub_llm, monkeypatch, clean_memory,
):
    """High LLM confidence + non-regulatory_critical risk class => no
    confidence-driven pause. The spec's one regulatory_critical derivation
    (COMPOSITE_RISK_TIER) still pauses for policy review; we auto-approve."""
    from backend.core import config as cfg

    monkeypatch.setattr(cfg.settings, "require_code_preapproval", False)
    monkeypatch.setattr(cfg.settings, "min_confidence_threshold", 0.7)

    spec = _load_spec(sample_paths["spec"])
    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec, dataset_path=str(sample_paths["dataset"]),
        )
        state = orch.run_to_completion(state)
        state = auto_approve_spec_clarifications(orch, state)
        # Pause should be policy-forced only — NOT a low-confidence pause.
        assert state.status == "awaiting_hitl_preapproval"
        ctx = state.hitl_pending.context or {}
        assert ctx.get("low_confidence_targets") == []
        assert "COMPOSITE_RISK_TIER" in (ctx.get("policy_forced_targets") or [])
        state = run_to_completion_auto_approving(orch, state)

    assert state.status.startswith("completed"), f"status was {state.status}"
    for d in state.derivations.values():
        assert (d.confidence or 0) >= 0.9
