"""TestRunner agent — Phase-0 quality gate.

Runs spec-declared test cases against every generated ``derive(row)`` function
**before** committing compute to full-dataset execution. Failed cases mark the
derivation as failed and feed concrete evidence into the Refiner.
"""
from __future__ import annotations

from typing import Any

from backend.agents.base import BaseAgent
from backend.core.test_runner import run_test_cases
from backend.core.workflow_state import ValidationRecord, WorkflowState


class TestRunnerAgent(BaseAgent):
    name = "test_runner"
    step = "test_cases"

    def run(self, state: WorkflowState) -> WorkflowState:
        with self.step_ctx() as rec:
            normalised = state.spec.get("normalised_derivations", []) or []
            total_cases = 0
            total_failed = 0
            targets_with_failures: list[str] = []

            for spec_d in normalised:
                target = spec_d["name"]
                cases: list[dict[str, Any]] = spec_d.get("test_cases") or []
                if not cases:
                    continue
                deriv = state.derivations.get(target)
                if not deriv or deriv.status not in ("generated", "refined"):
                    continue

                result = run_test_cases(deriv.code, target, cases)
                total_cases += result.total

                for case in result.cases:
                    rule_id = f"TEST_CASE_{case.case_index:02d}"
                    if case.passed:
                        msg = f"Pass: derive({case.input!r}) -> {case.actual!r}"
                        severity = "info"
                    else:
                        msg = (
                            f"Fail: derive({case.input!r}) returned {case.actual!r}, "
                            f"expected {case.expected!r}"
                        )
                        if case.error:
                            msg += f" ({case.error})"
                        severity = "block"
                    state.validations.append(ValidationRecord(
                        rule_id=rule_id,
                        target=target,
                        severity=severity,
                        passed=case.passed,
                        message=msg,
                        detail={
                            "input": case.input,
                            "expected": case.expected,
                            "actual": case.actual,
                            "error": case.error,
                        },
                    ))
                    self.stm.vals.record(
                        run_id=state.run_id,
                        target=target,
                        rule_id=rule_id,
                        severity=severity,
                        passed=case.passed,
                        message=msg,
                        detail={
                            "input": case.input,
                            "expected": case.expected,
                            "actual": case.actual,
                            "error": case.error,
                        },
                    )

                if not result.all_passed:
                    total_failed += len(result.failed_cases)
                    targets_with_failures.append(target)
                    deriv.status = "failed"
                    self.stm.derivs.upsert(
                        run_id=state.run_id,
                        target=target,
                        sources=list(deriv.sources),
                        rule_text=deriv.rule_text,
                        code=deriv.code,
                        code_hash=deriv.code_hash,
                        generator=deriv.generator,
                        attempt=deriv.attempt,
                        status="failed",
                    )

            rec.detail.update({
                "total_cases": total_cases,
                "failed_cases": total_failed,
                "targets_with_failures": targets_with_failures,
            })
        return state
