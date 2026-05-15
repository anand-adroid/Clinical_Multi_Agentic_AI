"""Shared test utilities (importable; ``conftest.py`` is not)."""
from __future__ import annotations


def auto_approve_spec_clarifications(orch, state):
    """Resume past ``awaiting_hitl_spec`` by accepting the LLM's normalised
    version with no clarification answers folded in. Used by tests that
    want to assert on subsequent gates (preapproval, refine, codegen)
    without first dealing with the spec gate that the comprehensive demo
    spec deliberately triggers."""
    while state.status == "awaiting_hitl_spec":
        state = orch.apply_hitl_decision(
            run_id=state.run_id,
            reviewer="test-auto-approver",
            action="approve",
            target=None,
            clarification_answers={},
        )
    return state


def run_to_completion_auto_approving(orch, state):
    """Run an orchestrator to completion, auto-approving any human-input
    gate the pipeline raises. Used by tests that want to exercise the full
    pipeline end-to-end when the spec contains regulatory_critical
    derivations (which force preapproval) or ambiguous rules (which trigger
    spec clarifications).

    Auto-approval semantics:
      - Spec clarifications: accept the LLM/heuristic's normalised version
        with no answers folded in.
      - Code preapproval: approve every queued derivation as-is.
    """
    state = orch.run_to_completion(state)
    while state.status.startswith("awaiting_hitl_"):
        reason = (state.hitl_pending.reason
                  if state.hitl_pending else "") or ""
        if reason == "spec_clarifications_required":
            state = orch.apply_hitl_decision(
                run_id=state.run_id,
                reviewer="test-auto-approver",
                action="approve",
                target=None,
                clarification_answers={},
            )
            continue
        if reason == "code_preapproval_required":
            targets = (state.hitl_pending.context or {}).get(
                "preapproval_targets"
            ) or []
            overrides = {t["target"]: {"action": "approve"} for t in targets}
            state = orch.apply_hitl_decision(
                run_id=state.run_id,
                reviewer="test-auto-approver",
                action="approve",
                target=None,
                derivation_overrides=overrides,
            )
            continue
        # Any other HITL gate (refine, codegen_failed) cannot be auto-resolved
        # without domain input; bail and let the test assert on the state.
        break
    return state
