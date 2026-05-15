"""
Orchestrator — the deterministic state machine that drives the agent pipeline.

Given the same input and configuration, this orchestrator advances through
the same sequence of agent calls. Each call is bracketed by a guard check
and a checkpoint write, so any run can be paused, killed, or audited at
will and resumed from the last checkpoint.

I wrote this by hand rather than reaching for LangGraph or CrewAI for two
reasons specific to the regulated-pharma setting:

  1. A regulator reads code, not graph DSLs. The full control flow fits in
     one file. Anything an auditor needs to understand about ``when does
     this agent run, with what inputs, after what guard?`` is right here.
  2. Human-in-the-Loop is a first-class pause primitive in this codebase
     and most graph frameworks treat human input as just another node.
     The pause/resume semantics here let a run halt mid-DAG, persist its
     state to disk, and re-enter from any checkpoint after an arbitrary
     delay — which is what real reviewers need.

States
    created -> running -> (awaiting_hitl_*) -> running -> completed | failed

Transitions
    create_run                       created -> running
    run_to_completion                running -> running | awaiting_hitl_*
    apply_hitl_decision              awaiting_hitl_* -> running
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from backend.agents.auditor import AuditorAgent
from backend.agents.code_generator import CodeGeneratorAgent
from backend.agents.code_preapproval import CodePreApprovalAgent
from backend.agents.dag_builder import DAGBuilderAgent
from backend.agents.executor import ExecutorAgent
from backend.agents.planner import PlannerAgent
from backend.agents.refiner import RefinerAgent
from backend.agents.spec_reviewer import SpecReviewerAgent
from backend.agents.static_validator import StaticValidatorAgent
from backend.agents.test_runner import TestRunnerAgent
from backend.agents.verifier import VerifierAgent
from backend.core import checkpoint
from backend.core.config import settings
from backend.core.guardrails import check_input_schema, check_pii
from backend.utils import console
from backend.core.workflow_state import (
    DerivationRecord,
    HITLRequest,
    ValidationRecord,
    WorkflowState,
)  # noqa: F401  — HITLRequest used by helpers below
from backend.db.repositories import (
    AuditRepository,
    DerivationRepository,
    EventRepository,
    HITLRepository,
    RunRepository,
    ValidationRepository,
)
from backend.memory.long_term import LongTermMemory
from backend.memory.short_term import ShortTermMemory
from backend.utils.hashing import hash_dataframe, hash_obj
from backend.utils.logging_setup import get_logger


log = get_logger("orchestrator")


PHASE_ORDER = [
    "input_guard",
    "spec_review",
    "dag_build",
    "plan",
    "code_generate",
    "code_preapproval",
    "static_validate",
    "test_cases",
    "execute",
    "verify",
    "refine",
    "audit",
]


class Orchestrator:
    """Stateless façade — every method takes the DB session and run_id."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.runs = RunRepository(db)
        self.events = EventRepository(db)
        self.derivs = DerivationRepository(db)
        self.vals = ValidationRepository(db)
        self.hitl = HITLRepository(db)
        self.audit = AuditRepository(db)

    # --------------------------------------------------------------- run setup
    def create_run(
        self,
        *,
        run_id: str,
        spec: dict[str, Any],
        dataset_path: str,
        user_id: str = "anonymous",
        notes: str | None = None,
    ) -> WorkflowState:
        df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_parquet(dataset_path)
        spec_hash = hash_obj(spec)
        dataset_hash = hash_dataframe(df)
        self.runs.create(
            run_id=run_id, spec_hash=spec_hash, dataset_hash=dataset_hash,
            user_id=user_id, notes=notes,
        )
        state = WorkflowState(
            run_id=run_id,
            spec=spec,
            spec_hash=spec_hash,
            dataset_hash=dataset_hash,
            dataset_path=dataset_path,
            status="created",
            config_snapshot={
                "llm_enabled": settings.llm_enabled,
                "max_refine_retries": settings.max_refine_retries,
                "max_runtime_seconds": settings.max_runtime_seconds,
            },
        )
        checkpoint.save(run_id, "00_created", state.to_dict())
        self.audit.record(
            run_id=run_id, actor="orchestrator", actor_type="system",
            action="run.created",
            detail={"spec_hash": spec_hash, "dataset_hash": dataset_hash},
        )
        self.db.commit()
        console.run_banner(
            run_id,
            note=f"dataset={Path(dataset_path).name}, llm={'on' if settings.llm_enabled else 'off (rule-based)'}",
        )
        return state

    # --------------------------------------------------------------- run loop
    def run_to_completion(self, state: WorkflowState) -> WorkflowState:
        """Run all phases until completion or a HITL pause."""
        deadline = time.time() + settings.max_runtime_seconds
        state.status = "running"
        self.runs.update_status(state.run_id, "running")
        if state.events:
            last_step = state.events[-1].step if state.events else "start"
            console.run_resumed(state.run_id, last_step)

        stm = ShortTermMemory(self.db, state)
        ltm = LongTermMemory(self.db)

        TOTAL = 12

        # ----- 1. Input guardrails (PII + schema) -----
        if not self._already_done(state, "input_guard"):
            with console.phase(state.run_id, 1, TOTAL, "Input guardrails"):
                self._run_input_guards(stm, state)
            checkpoint.save(state.run_id, "01_input_guard", state.to_dict())
            if state.status == "failed":
                self.runs.update_status(state.run_id, "failed",
                                        summary=state.summary)
                self.db.commit()
                console.run_completed(state.run_id, "failed", state.summary)
                return state

        # ----- 2. Spec review -----
        if not self._already_done(state, "spec_review"):
            with console.phase(state.run_id, 2, TOTAL, "Spec review"):
                SpecReviewerAgent(stm).run(state)
            checkpoint.save(state.run_id, "02_spec_review", state.to_dict())
            if state.hitl_pending:
                console.run_paused(
                    state.run_id,
                    f"clarifications requested: {len(state.spec.get('clarifications') or [])}",
                )
                return self._pause_for_hitl(state, "awaiting_hitl_spec")

        # ----- 3. DAG build -----
        if not self._already_done(state, "dag_build"):
            with console.phase(state.run_id, 3, TOTAL, "DAG build"):
                DAGBuilderAgent(stm).run(state)
            console.say(f"execution order: {' -> '.join(state.topo_order)}")
            checkpoint.save(state.run_id, "03_dag_build", state.to_dict())

        # ----- 4. Plan (LLM-driven per-derivation policy) -----
        if not self._already_done(state, "plan"):
            with console.phase(state.run_id, 4, TOTAL, "Plan"):
                PlannerAgent(stm).run(state)
            if state.plan:
                console.say(
                    f"strategy={state.plan.strategy}  "
                    f"plan_hash={state.plan.plan_hash[:10]}  "
                    f"policies={len(state.plan.policies)}"
                )
            checkpoint.save(state.run_id, "04_plan", state.to_dict())

        # ----- 5. Code generation -----
        if not self._already_done(state, "code_generate"):
            with console.phase(state.run_id, 5, TOTAL, "Code generation"):
                CodeGeneratorAgent(stm, ltm).run(state)
            for tgt, d in state.derivations.items():
                conf = f" conf={d.confidence:.2f}" if d.confidence is not None else ""
                console.say(f"{tgt:<24} generator={d.generator}  status={d.status}{conf}")
            checkpoint.save(state.run_id, "05_code_generate", state.to_dict())
            # Phase 4: when code generation produced *nothing* usable across the
            # board (no LLM, no memory hit), escalate immediately to HITL.
            unfinished = [
                d for d in state.derivations.values()
                if d.status not in ("generated", "ok", "refined")
            ]
            if unfinished and len(unfinished) == len(state.derivations):
                state.hitl_pending = self._build_codegen_failure_hitl(state, unfinished)
                console.run_paused(
                    state.run_id,
                    f"code generation failed for all {len(unfinished)} derivations",
                )
                return self._pause_for_hitl(state, "awaiting_hitl_codegen")

        # ----- 6. Code preapproval (Phase 3 / G2) -----
        if not self._already_done(state, "code_preapproval"):
            with console.phase(state.run_id, 6, TOTAL, "Code preapproval"):
                CodePreApprovalAgent(stm).run(state)
            checkpoint.save(state.run_id, "06_code_preapproval", state.to_dict())
            if state.hitl_pending and state.hitl_pending.reason == "code_preapproval_required":
                n = len((state.hitl_pending.context or {}).get("preapproval_targets") or [])
                console.run_paused(
                    state.run_id,
                    f"code preapproval requested for {n} derivation(s)",
                )
                return self._pause_for_hitl(state, "awaiting_hitl_preapproval")

        # ----- 7. Static validation -----
        if not self._already_done(state, "static_validate"):
            with console.phase(state.run_id, 7, TOTAL, "Static validation"):
                StaticValidatorAgent(stm).run(state)
            checkpoint.save(state.run_id, "07_static_validate", state.to_dict())

        # ----- 8. Test cases (Phase-0 quality gate) -----
        if not self._already_done(state, "test_cases"):
            with console.phase(state.run_id, 8, TOTAL, "Test cases"):
                TestRunnerAgent(stm).run(state)
            tc_validations = [
                v for v in state.validations
                if v.rule_id.startswith("TEST_CASE_")
            ]
            tc_pass = sum(1 for v in tc_validations if v.passed)
            tc_total = len(tc_validations)
            if tc_total:
                console.say(f"test cases: {tc_pass}/{tc_total} passing")
            else:
                console.say("no test cases declared in spec — skipping")
            checkpoint.save(state.run_id, "08_test_cases", state.to_dict())

        # ----- 9. Execute -----
        if not self._already_done(state, "execute"):
            with console.phase(state.run_id, 9, TOTAL, "Execute"):
                ExecutorAgent(stm).run(state)
            if state.output_path:
                console.say(f"output written to {state.output_path}")
            checkpoint.save(state.run_id, "09_execute", state.to_dict())

        # ----- 10. Verify -----
        if not self._already_done(state, "verify"):
            with console.phase(state.run_id, 10, TOTAL, "Verify"):
                VerifierAgent(stm).run(state)
            failed = sum(1 for v in state.validations if v.severity == "block" and not v.passed)
            warned = sum(1 for v in state.validations if v.severity == "warn" and not v.passed)
            console.say(f"verification: {failed} blocking, {warned} warnings")
            checkpoint.save(state.run_id, "10_verify", state.to_dict())

        # ----- 11. Refine if needed; loop with bound -----
        loops = 0
        while loops < settings.max_refine_retries:
            failing = [t for t, d in state.derivations.items()
                       if d.status in ("failed", "unsafe")]
            blocked_validations = [v for v in state.validations
                                   if v.severity == "block" and not v.passed]
            if not failing and not blocked_validations:
                break
            loops += 1
            with console.phase(state.run_id, 11, TOTAL, f"Refine (attempt {loops}/{settings.max_refine_retries})"):
                console.say(f"failing targets: {failing or 'none'}")
                RefinerAgent(stm).run(state)
            checkpoint.save(state.run_id, f"10_refine_{loops:02d}", state.to_dict())
            if state.hitl_pending:
                console.run_paused(state.run_id, "refinement budget exhausted, human escalation")
                return self._pause_for_hitl(state, "awaiting_hitl_refine")
            # Re-run test cases, then re-execute & re-verify the refined targets.
            TestRunnerAgent(stm).run(state)
            ExecutorAgent(stm).run(state)
            VerifierAgent(stm).run(state)
            checkpoint.save(state.run_id, f"10_verify_{loops:02d}", state.to_dict())

        # ----- 11. Promote validated patterns into LTM -----
        self._promote_to_ltm(state, ltm)

        # ----- 12. Audit report -----
        with console.phase(state.run_id, 12, TOTAL, "Audit"):
            AuditorAgent(stm).run(state)
        checkpoint.save(state.run_id, "11_audit", state.to_dict())

        # Final status
        critical = [v for v in state.validations if v.severity == "block" and not v.passed]
        state.status = "completed" if not critical else "completed_with_warnings"
        state.summary["derivations_total"] = len(state.derivations)
        state.summary["derivations_ok"] = sum(
            1 for d in state.derivations.values() if d.status in ("ok", "refined")
        )
        state.summary["blocking_findings"] = len(critical)
        self.runs.update_status(state.run_id, state.status, summary=state.summary)
        self.audit.record(
            run_id=state.run_id, actor="orchestrator", actor_type="system",
            action="run.completed", detail=state.summary,
        )
        self.db.commit()
        console.run_completed(state.run_id, state.status, state.summary)
        return state

    # ------------------------------------------------------- HITL pause/resume
    def _pause_for_hitl(self, state: WorkflowState, status: str) -> WorkflowState:
        state.status = status
        self.runs.update_status(state.run_id, status, summary=state.summary)
        if state.hitl_pending:
            self.audit.record(
                run_id=state.run_id, actor="orchestrator", actor_type="system",
                action="hitl.requested",
                detail={"target": state.hitl_pending.target,
                        "reason": state.hitl_pending.reason},
            )
        self.db.commit()
        # Write a checkpoint so the restored state on the next
        # ``/runs/{id}/hitl/pending`` request includes ``hitl_pending``.
        # Without this, the latest per-phase checkpoint pre-dates the
        # assignment of ``state.hitl_pending`` in the calling phase, so
        # the API returns ``{pending: false}`` and the UI cannot render
        # the form even though the run is genuinely paused.
        checkpoint.save(state.run_id, f"hitl_paused_{status}", state.to_dict())
        return state

    def apply_hitl_decision(
        self,
        run_id: str,
        *,
        reviewer: str,
        action: str,
        target: str | None,
        comment: str | None = None,
        edited_code: str | None = None,
        clarification_answers: dict[str, str] | None = None,
        derivation_overrides: dict[str, dict[str, Any]] | None = None,
        regenerate_hint: str | None = None,
        resume: bool = True,
    ) -> WorkflowState:
        state = ShortTermMemory.restore(self.db, run_id)
        if not state:
            raise ValueError(f"No checkpoint for run {run_id}")
        console.hitl_decision(run_id, reviewer, action, target)
        pending_reason = state.hitl_pending.reason if state.hitl_pending else None
        # Persist a structured comment so the audit trail captures exactly what
        # the reviewer answered, not just "approve".
        structured_comment = comment
        if clarification_answers:
            import json as _json
            structured_comment = _json.dumps({
                "comment": comment,
                "clarification_answers": clarification_answers,
            })
        elif derivation_overrides:
            import json as _json
            structured_comment = _json.dumps({
                "comment": comment,
                "derivation_overrides": derivation_overrides,
            })
        self.hitl.record(
            run_id=run_id, target=target, reviewer=reviewer, action=action,
            comment=structured_comment, edited_code=edited_code,
        )
        self.audit.record(
            run_id=run_id, actor=reviewer, actor_type="human",
            action=f"hitl.{action}", object_ref=target,
            detail={
                "comment": comment,
                "edited": bool(edited_code),
                "clarification_count": len(clarification_answers or {}),
                "override_count": len(derivation_overrides or {}),
                "pending_reason": pending_reason,
            },
        )
        state.hitl_history.append({
            "reviewer": reviewer, "action": action, "target": target,
            "comment": comment, "edited": bool(edited_code),
            "clarification_answers": clarification_answers or {},
            "derivation_overrides": derivation_overrides or {},
            "at": datetime.utcnow().isoformat(),
        })

        # ------------------------------------------------------------------
        # Phase 1: fold clarification answers into the normalised spec rules.
        # ------------------------------------------------------------------
        if clarification_answers and action != "reject":
            self._apply_clarification_answers(
                state, clarification_answers, reviewer=reviewer
            )

        # ------------------------------------------------------------------
        # Phase 3: apply per-derivation overrides from the code-preapproval
        # gate. Each override is one of: approve | edit (with code) | regenerate
        # (with optional hint).
        # ------------------------------------------------------------------
        if derivation_overrides and action != "reject":
            self._apply_derivation_overrides(
                state,
                derivation_overrides,
                regenerate_hint=regenerate_hint,
            )

        # Apply the decision to state
        if action == "approve":
            # Clearing the pending HITL request lets the orchestrator advance.
            state.hitl_pending = None
            # Flip status back to "running" so the API spawn-check and
            # the frontend status pill both reflect the resumed state.
            if state.status.startswith("awaiting_hitl_"):
                state.status = "running"
                self.runs.update_status(run_id, "running")
        elif action == "regenerate":
            # Treat as approve at the pause level; the actual regeneration is
            # handled in ``_apply_derivation_overrides`` (which sets the failing
            # derivations back to status="pending" so CodeGenerator re-runs them).
            state.hitl_pending = None
            if state.status.startswith("awaiting_hitl_"):
                state.status = "running"
                self.runs.update_status(run_id, "running")
        elif action == "edit" and target and edited_code:
            d = state.derivations.get(target)
            if not d:
                # New derivation injected by reviewer — record minimally.
                from backend.utils.hashing import hash_text
                spec_d = next(
                    (s for s in state.spec.get("normalised_derivations", []) if s["name"] == target),
                    None,
                )
                state.derivations[target] = DerivationRecord(
                    target=target,
                    sources=list(spec_d.get("sources", [])) if spec_d else [],
                    rule_text=spec_d.get("rule", "") if spec_d else "",
                    code=edited_code,
                    code_hash=hash_text(edited_code),
                    generator="human",
                    attempt=1, status="generated",
                )
            else:
                from backend.utils.hashing import hash_text
                d.code = edited_code
                d.code_hash = hash_text(edited_code)
                d.generator = "human"
                d.attempt += 1
                d.status = "generated"
            self.derivs.upsert(
                run_id=run_id, target=target,
                sources=list(state.derivations[target].sources),
                rule_text=state.derivations[target].rule_text,
                code=edited_code,
                code_hash=state.derivations[target].code_hash,
                generator="human",
                attempt=state.derivations[target].attempt,
                status="generated",
            )
            state.hitl_pending = None
            if state.status.startswith("awaiting_hitl_"):
                state.status = "running"
                self.runs.update_status(run_id, "running")
        elif action == "reject":
            state.status = "failed"
            state.summary["rejected_by_reviewer"] = True
            self.runs.update_status(run_id, "failed", summary=state.summary)
            self.db.commit()
            checkpoint.save(run_id, "hitl_reject", state.to_dict())
            return state
        else:
            raise ValueError(f"Unknown HITL action `{action}`")

        checkpoint.save(run_id, "hitl_applied", state.to_dict())
        self.db.commit()
        if resume:
            # Re-enter the loop synchronously; ``_already_done`` skips
            # phases that completed before the pause. The HTTP route that
            # threads the orchestrator passes ``resume=False`` and runs
            # the continuation in a background worker instead.
            return self.run_to_completion(state)
        return state

    # ---------------------------------------------- HITL helpers (Phases 1-3)
    def _apply_clarification_answers(
        self,
        state: WorkflowState,
        answers: dict[str, str],
        *,
        reviewer: str,
    ) -> None:
        """Phase 1: fold structured reviewer answers into the normalised spec.

        For each (derivation_name, answer), append the answer to the
        derivation's rule text so the code generator sees the resolved
        constraint directly. Recompute ``resolved_spec_hash`` and persist
        to the clarification-memory store (Phase 2) for future runs.
        """
        from backend.memory.long_term import LongTermMemory  # local import to avoid cycle

        normalised = state.spec.get("normalised_derivations") or []
        clarifications = state.spec.get("clarifications") or []
        by_name = {n.get("name"): n for n in normalised}
        clarif_by_name = {c.get("name"): c for c in clarifications}
        applied = 0
        ltm = LongTermMemory(self.db)
        for name, answer in (answers or {}).items():
            answer = (answer or "").strip()
            if not answer:
                continue
            n = by_name.get(name)
            if not n:
                continue
            original_rule = n.get("rule", "")
            n["rule"] = (
                f"{original_rule}\n\nClarified by reviewer ({reviewer}): {answer}"
            ).strip()
            applied += 1
            # Phase 2: persist this resolution to long-term memory.
            issue = (clarif_by_name.get(name) or {}).get("issue", "")
            try:
                ltm.remember_clarification(
                    target=name, issue=issue, answer=answer, reviewer=reviewer,
                )
            except Exception as exc:  # noqa: BLE001
                self.audit.record(
                    run_id=state.run_id, actor="orchestrator", actor_type="system",
                    action="clarification.memory_write_failed",
                    detail={"error": str(exc)},
                )

        if applied:
            state.spec["normalised_derivations"] = normalised
            # Drop the resolved clarifications from the spec so a re-run of
            # SpecReviewer doesn't immediately re-raise them.
            remaining = [c for c in clarifications if c.get("name") not in answers]
            state.spec["clarifications"] = remaining
            state.resolved_spec_hash = hash_obj(normalised)
            self.audit.record(
                run_id=state.run_id, actor=reviewer, actor_type="human",
                action="spec.clarifications_resolved",
                detail={
                    "applied_count": applied,
                    "resolved_spec_hash": state.resolved_spec_hash,
                },
            )

    def _build_codegen_failure_hitl(
        self,
        state: WorkflowState,
        unfinished: list[DerivationRecord],
    ) -> HITLRequest:
        """Phase 4: when code generation could not produce any usable code,
        package the failure context for a human reviewer."""
        normalised = {
            d.get("name"): d
            for d in (state.spec.get("normalised_derivations") or [])
        }
        targets = []
        for d in unfinished:
            spec_d = normalised.get(d.target) or {}
            targets.append({
                "target": d.target,
                "sources": list(d.sources),
                "rule": spec_d.get("rule", d.rule_text),
                "status": d.status,
                "generator": d.generator,
                "type": spec_d.get("type"),
                "allowed_values": spec_d.get("allowed_values"),
            })
        return HITLRequest(
            target=None,
            reason="codegen_failed",
            context={
                "message": (
                    "Code generation could not produce safe code for any "
                    "derivation. The LLM is unavailable or its output failed "
                    "every safety check. Provide code manually or retry once "
                    "an LLM is configured."
                ),
                "preapproval_targets": targets,
            },
        )

    def _apply_derivation_overrides(
        self,
        state: WorkflowState,
        overrides: dict[str, dict[str, Any]],
        *,
        regenerate_hint: str | None = None,
    ) -> None:
        """Phase 3: apply per-derivation actions from the code-preapproval gate.

        ``overrides`` is a dict ``{target_name: {"action": ..., "code": ..., "hint": ...}}``.
        Supported per-derivation actions:
          * ``approve``    — keep generated code as-is (no-op here).
          * ``edit``       — replace code with reviewer-supplied snippet.
          * ``regenerate`` — wipe the derivation so the code generator re-runs
            with the optional ``hint`` woven into the prompt.
        """
        from backend.utils.hashing import hash_text

        regen_hints = state.spec.setdefault("regenerate_hints", {})
        for tgt, info in (overrides or {}).items():
            sub_action = (info or {}).get("action", "approve")
            d = state.derivations.get(tgt)
            if sub_action == "approve":
                continue
            if sub_action == "edit":
                new_code = (info or {}).get("code") or ""
                if not d or not new_code.strip():
                    continue
                d.code = new_code
                d.code_hash = hash_text(new_code)
                d.generator = "human"
                d.attempt += 1
                d.status = "generated"
                d.confidence = 1.0
                d.uncertainty_notes = None
                self.derivs.upsert(
                    run_id=state.run_id, target=tgt,
                    sources=list(d.sources), rule_text=d.rule_text,
                    code=new_code, code_hash=d.code_hash,
                    generator="human", attempt=d.attempt, status="generated",
                )
            elif sub_action == "regenerate":
                hint = (info or {}).get("hint") or regenerate_hint
                if hint:
                    regen_hints[tgt] = hint
                # Drop the derivation so CodeGenerator regenerates from scratch.
                state.derivations.pop(tgt, None)
                # Invalidate downstream phase events so resume re-runs them.
                state.events = [
                    e for e in state.events
                    if e.step not in ("static_check", "test_cases", "execute", "verify")
                ]
                state.output_path = None

    # ---------------------------------------------------------- helpers
    def _already_done(self, state: WorkflowState, phase: str) -> bool:
        # Resuming a checkpointed run: skip phases that have already
        # produced their characteristic side-effect.
        if phase == "input_guard":
            return any(v.rule_id.startswith("INPUT_GUARD_DONE") for v in state.validations)
        if phase == "spec_review":
            return "normalised_derivations" in state.spec
        if phase == "dag_build":
            return bool(state.topo_order)
        if phase == "plan":
            return state.plan is not None
        if phase == "code_generate":
            # If every declared derivation already has a code record, skip;
            # otherwise re-run (e.g. after a Phase-3 regenerate override).
            declared = {d["name"] for d in (state.spec.get("normalised_derivations") or [])}
            return bool(declared) and declared.issubset(state.derivations.keys())
        if phase == "code_preapproval":
            return any(e.step == "preapproval" and e.status in ("ok", "warn") for e in state.events)
        if phase == "static_validate":
            return any(e.step == "static_check" and e.status in ("ok", "warn") for e in state.events)
        if phase == "test_cases":
            return any(e.step == "test_cases" and e.status in ("ok", "warn") for e in state.events)
        if phase == "execute":
            return bool(state.output_path)
        if phase == "verify":
            return any(e.step == "verify" and e.status in ("ok", "warn") for e in state.events)
        return False

    def _run_input_guards(self, stm: ShortTermMemory, state: WorkflowState) -> None:
        df = pd.read_csv(state.dataset_path) if state.dataset_path.endswith(".csv") else pd.read_parquet(state.dataset_path)
        pii = check_pii(df)
        schema = check_input_schema(df, state.spec.get("source_schema") or {})
        for r in (pii, schema):
            for f in r.findings:
                state.validations.append(ValidationRecord(
                    rule_id=f.code, target=None, severity=f.severity,
                    passed=False, message=f.message, detail=f.details,
                ))
                stm.vals.record(
                    run_id=state.run_id, target=None, rule_id=f.code,
                    severity=f.severity, passed=False, message=f.message, detail=f.details,
                )
        if not pii.ok or not schema.ok:
            state.status = "failed"
            state.summary["input_guard"] = "failed"
            stm.audit.record(
                run_id=state.run_id, actor="orchestrator", actor_type="system",
                action="input_guard.blocked",
            )
            return
        # Sentinel record so the resume check above can detect this phase ran.
        state.validations.append(ValidationRecord(
            rule_id="INPUT_GUARD_DONE", target=None, severity="info",
            passed=True, message="Input guardrails passed.",
        ))
        stm.vals.record(
            run_id=state.run_id, target=None, rule_id="INPUT_GUARD_DONE",
            severity="info", passed=True, message="Input guardrails passed.",
        )

    def _promote_to_ltm(self, state: WorkflowState, ltm: LongTermMemory) -> None:
        """Promote every successful derivation into long-term memory so future
        runs can reuse it. Patterns from a human-edited derivation get a small
        score boost via the repository."""
        from backend.agents.code_generator import PROMPT_VERSION as _CG_VERSION
        for d in state.derivations.values():
            if d.status in ("ok", "refined") and d.generator != "memory":
                ltm.remember(
                    target=d.target,
                    sources=d.sources,
                    rule_text=d.rule_text,
                    code=d.code,
                    created_by=d.generator,
                    reasoning=d.reasoning,
                    generator_version=_CG_VERSION,
                )
