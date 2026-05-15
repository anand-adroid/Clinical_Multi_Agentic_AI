"""
StaticValidator agent
=====================

A focused agent that re-runs the code-safety guard after code generation
has settled. Keeping this as its own step (instead of inlining it in
CodeGenerator) earns its place because:

  * It is auditable as its own row in `agent_events` — regulators want
    to see "the code was safety-checked, by this agent, at this time".
  * It can be re-run after a human edit during HITL review without rerunning
    code generation.
"""
from __future__ import annotations

from backend.agents.base import BaseAgent
from backend.core.guardrails import check_generated_code
from backend.core.workflow_state import ValidationRecord, WorkflowState


class StaticValidatorAgent(BaseAgent):
    name = "static_validator"
    step = "static_check"

    def run(self, state: WorkflowState) -> WorkflowState:
        with self.step_ctx() as rec:
            schema_cols = list(state.spec.get("source_schema", {}).keys())
            available = schema_cols + list(state.derivations.keys())
            unsafe: list[str] = []
            for tgt, d in state.derivations.items():
                g = check_generated_code(d.code, available_columns=available)
                for f in g.findings:
                    state.validations.append(ValidationRecord(
                        rule_id=f.code, target=tgt, severity=f.severity,
                        passed=False, message=f.message, detail=f.details,
                    ))
                    self.stm.vals.record(
                        run_id=state.run_id, target=tgt, rule_id=f.code,
                        severity=f.severity, passed=False, message=f.message, detail=f.details,
                    )
                if not g.ok:
                    d.status = "unsafe"
                    unsafe.append(tgt)
            rec.detail["unsafe"] = unsafe
            if unsafe:
                # Soft-fail: the refiner is given a chance, this is not yet a workflow halt.
                self.log.warning("static_validator.unsafe", targets=unsafe)
        return state
