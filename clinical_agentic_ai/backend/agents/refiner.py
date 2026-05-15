"""
Refiner agent — bounded self-repair for derivations the verifier blocked.

When the Verifier flags a derivation as block-level broken, the Refiner
gets up to ``MAX_REFINE_RETRIES`` (default 3) attempts to fix the snippet
by re-prompting the LLM with the failing rule plus the validator findings.

Stopping criteria, evaluated each attempt:
  - the new code passes static-check AND the verifier rules for that
    target — accept it and move on,
  - ``MAX_REFINE_RETRIES`` is reached — escalate to HITL with the full
    failure context for a human to inspect,
  - the same code hash is produced twice — no progress, escalate.

The loop is deliberately bounded. A free-running self-repair agent in a
regulated environment is a liability; bounded + observable + escalating
to a human is the right pattern for any system that has to face an auditor.
"""
from __future__ import annotations

from typing import Any

from backend.agents.base import BaseAgent
from backend.core.config import settings
from backend.core.guardrails import check_generated_code, check_output_column
from backend.core.llm_client import LLMError, llm
from backend.core.sandbox import run_per_row
from backend.core.workflow_state import HITLRequest, ValidationRecord, WorkflowState
from backend.utils.hashing import hash_text

import pandas as pd


PROMPT_VERSION = "refiner/v1"


_REFINE_SYSTEM = """You are a code repair assistant for clinical derivations.
You receive: the original rule, the previously generated code, and a list of
findings explaining why it failed. Produce a corrected `derive(row)` function.
Same constraints as before. Reply with JSON {"code": "..."} only.
"""


class RefinerAgent(BaseAgent):
    name = "refiner"
    step = "refine"

    def run(self, state: WorkflowState) -> WorkflowState:
        with self.step_ctx() as rec:
            failing = [t for t, d in state.derivations.items() if d.status in ("failed", "unsafe")]
            df_full = pd.read_csv(state.dataset_path) if state.dataset_path.endswith(".csv") else pd.read_parquet(state.dataset_path)

            schema_cols = list(state.spec.get("source_schema", {}).keys())
            available = schema_cols + list(state.derivations.keys())

            refined = 0
            escalated = 0
            for target in failing:
                deriv = state.derivations[target]
                spec_d = next((d for d in state.spec["normalised_derivations"] if d["name"] == target), None)
                if not spec_d:
                    continue
                last_hash = deriv.code_hash
                for attempt in range(2, settings.max_refine_retries + 2):
                    findings = [v for v in state.validations if v.target == target and not v.passed]
                    new_code = self._propose_fix(deriv.code, spec_d, findings)
                    if not new_code or hash_text(new_code) == last_hash:
                        break
                    last_hash = hash_text(new_code)
                    guard = check_generated_code(new_code, available_columns=available)
                    if not guard.ok:
                        continue
                    # Try to execute, then re-verify the target column.
                    try:
                        result = run_per_row(new_code, df_full, max_seconds=settings.max_runtime_seconds)
                    except Exception as exc:  # noqa: BLE001
                        self.log.warning("refiner.exec_failed", target=target, attempt=attempt, error=str(exc))
                        continue

                    output_guard = check_output_column(
                        target, result.values,
                        expected_type=spec_d.get("type"),
                        allowed_values=spec_d.get("allowed_values") or None,
                        max_null_rate=float(spec_d.get("max_null_rate", 0.5)),
                    )
                    if output_guard.ok and not any(f.severity == "block" for f in output_guard.findings):
                        deriv.code = new_code
                        deriv.code_hash = last_hash
                        deriv.attempt = attempt
                        deriv.status = "refined"
                        deriv.generator = "refiner"
                        deriv.row_errors = [
                            {"row_index": e.row_index, "error": e.error}
                            for e in result.row_errors[:50]
                        ]
                        deriv.null_count = sum(1 for v in result.values if v is None)
                        self.stm.derivs.upsert(
                            run_id=state.run_id, target=target,
                            sources=list(spec_d.get("sources", [])),
                            rule_text=spec_d.get("rule", ""),
                            code=new_code, code_hash=last_hash,
                            generator="refiner", attempt=attempt, status="refined",
                        )
                        refined += 1
                        break

                if state.derivations[target].status not in ("ok", "refined"):
                    # Escalate to HITL — bounded retry budget exhausted.
                    state.hitl_pending = HITLRequest(
                        target=target,
                        reason="refinement_exhausted",
                        context={
                            "current_code": state.derivations[target].code,
                            "rule": spec_d.get("rule", ""),
                            "recent_findings": [
                                {"rule_id": v.rule_id, "message": v.message, "severity": v.severity}
                                for v in state.validations if v.target == target and not v.passed
                            ][-5:],
                        },
                    )
                    escalated += 1
                    break  # one HITL pause at a time

            rec.detail["refined"] = refined
            rec.detail["escalated"] = escalated
        return state

    # -------------------- helpers --------------------

    def _propose_fix(
        self,
        previous_code: str,
        spec_d: dict[str, Any],
        findings,
    ) -> str | None:
        if llm.enabled:
            tests = spec_d.get("test_cases") or []
            tests_block = ""
            if tests:
                tests_block = (
                    "\n\ntest_cases (your fix MUST satisfy each one):\n"
                    + "\n".join(
                        f"  - derive({tc.get('input')!r}) should return {tc.get('expected')!r}"
                        for tc in tests
                    )
                )
            try:
                resp = llm.complete(
                    purpose=f"{PROMPT_VERSION}:{spec_d.get('name', '?')}",
                    system=_REFINE_SYSTEM,
                    user=(
                        f"target: {spec_d['name']}\n"
                        f"sources: {spec_d.get('sources')}\n"
                        f"type: {spec_d.get('type')}\n"
                        f"allowed_values: {spec_d.get('allowed_values')}\n"
                        f"rule: {spec_d.get('rule')}\n\n"
                        f"previous_code:\n{previous_code}\n\n"
                        f"findings:\n" + "\n".join(
                            f"- [{f.severity}] {f.rule_id}: {f.message}" for f in findings[-10:]
                        )
                        + tests_block
                    ),
                    expect_json=True,
                )
                parsed = resp.parsed or {}
                return (parsed.get("code") or "").strip() or None
            except LLMError as exc:
                self.log.warning("refiner.llm_error", target=spec_d.get("name"), error=str(exc))
                return None
        # Phase 4: no hardcoded fallback. Without an LLM the Refiner cannot
        # repair the code; it returns ``None`` so the orchestrator escalates
        # to HITL after the retry budget is exhausted.
        return None
