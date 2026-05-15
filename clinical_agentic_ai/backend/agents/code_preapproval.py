"""Code Pre-Approval gate.

Sits between Code Generation and Static Validation. For each generated
``derive(row)`` function the agent produces a small preview by running the
code on the first N rows of the dataset, then raises a single HITL request
bundling all derivations that need review into one structured pause.

Pause triggers (any of):
  - the Planner Agent's policy for a derivation has ``require_preapproval``
    set (typically for ``regulatory_critical`` columns),
  - the LLM's self-reported confidence for a derivation is below the
    policy's threshold (or the global threshold when no plan is present),
  - ``settings.require_code_preapproval`` is True globally (cautious mode).

If none of those fire, the agent is a no-op and the pipeline moves on.
This is what makes the pipeline risk-adaptive: routine derivations skip
review entirely, regulatory-critical ones always pause, and ambiguous
ones pause only when the LLM itself flags low confidence.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from backend.agents.base import BaseAgent
from backend.core.config import settings
from backend.core.sandbox import SandboxViolation, compile_function
from backend.core.workflow_state import HITLRequest, WorkflowState


def _dry_run(code: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run ``code`` against a small list of row-dicts. Each entry in the
    returned list is ``{"input": row, "output": value, "error": str|None}``.
    """
    try:
        fn = compile_function(code)
    except SandboxViolation as exc:
        return [
            {"input": r, "output": None, "error": f"compile failed: {exc}"}
            for r in rows
        ]
    results: list[dict[str, Any]] = []
    for r in rows:
        try:
            v = fn(dict(r))
            results.append({"input": r, "output": v, "error": None})
        except Exception as exc:  # noqa: BLE001
            results.append({
                "input": r,
                "output": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return results


class CodePreApprovalAgent(BaseAgent):
    name = "code_preapproval"
    step = "preapproval"

    def run(self, state: WorkflowState) -> WorkflowState:
        with self.step_ctx() as rec:
            if not state.derivations:
                rec.detail["skipped"] = "no derivations"
                return state

            # Build the small preview frame and then walk derivations in
            # topological order, accumulating each derivation's output into
            # the preview rows. That way a Level-2 derivation that reads a
            # Level-1 derived column sees it during dry-run, just like it
            # will during full execution.
            preview_rows: list[dict[str, Any]] = []
            previews_per_target: dict[str, list[dict[str, Any]]] = {}
            try:
                df = (
                    pd.read_csv(state.dataset_path)
                    if state.dataset_path.endswith(".csv")
                    else pd.read_parquet(state.dataset_path)
                )
                preview_rows = df.head(
                    max(1, int(settings.preapproval_preview_rows))
                ).to_dict(orient="records")
            except Exception as exc:  # noqa: BLE001
                rec.detail["preview_load_error"] = str(exc)

            if preview_rows:
                for tgt in state.topo_order:
                    d = state.derivations.get(tgt)
                    if not d or not d.code:
                        continue
                    target_preview = _dry_run(d.code, preview_rows)
                    previews_per_target[tgt] = target_preview
                    # Carry each dry-run's output into the next derivation's
                    # row dicts so downstream derivations can read it.
                    for row, p in zip(preview_rows, target_preview):
                        row[tgt] = p.get("output")

            preapproval_targets: list[dict[str, Any]] = []
            triggered_low_confidence: list[str] = []
            triggered_policy: list[str] = []

            policies = state.plan.policies if state.plan else {}

            for tgt, d in state.derivations.items():
                policy = policies.get(tgt)
                threshold = (
                    policy.confidence_threshold
                    if policy
                    else float(settings.min_confidence_threshold)
                )
                low_conf = (
                    d.confidence is not None
                    and d.confidence < threshold
                )
                policy_forced = bool(policy and policy.require_preapproval)
                if low_conf:
                    triggered_low_confidence.append(tgt)
                if policy_forced:
                    triggered_policy.append(tgt)
                if (
                    not settings.require_code_preapproval
                    and not low_conf
                    and not policy_forced
                ):
                    continue
                spec_d = next(
                    (s for s in state.spec.get("normalised_derivations") or []
                     if s.get("name") == tgt),
                    None,
                )
                preview = previews_per_target.get(tgt) or []
                trigger = (
                    "policy" if policy_forced
                    else ("low_confidence" if low_conf else "global_flag")
                )
                preapproval_targets.append({
                    "target": tgt,
                    "generator": d.generator,
                    "rule": (spec_d or {}).get("rule", d.rule_text),
                    "sources": list(d.sources),
                    "code": d.code,
                    "code_hash": d.code_hash,
                    "confidence": d.confidence,
                    "uncertainty_notes": d.uncertainty_notes,
                    "reasoning": d.reasoning,
                    "risk_class": d.risk_class,
                    "preview": preview,
                    "trigger": trigger,
                    "policy_threshold": threshold,
                    "reviewer_tier": policy.reviewer_tier if policy else "T1",
                    "policy_rationale": policy.rationale if policy else "",
                })

            if not preapproval_targets:
                rec.detail["skipped"] = "nothing to preapprove"
                return state

            state.hitl_pending = HITLRequest(
                target=None,
                reason="code_preapproval_required",
                context={
                    "preapproval_targets": preapproval_targets,
                    "policy_require_all": bool(settings.require_code_preapproval),
                    "min_confidence_threshold": float(settings.min_confidence_threshold),
                    "low_confidence_targets": triggered_low_confidence,
                    "policy_forced_targets": triggered_policy,
                    "plan_strategy": state.plan.strategy if state.plan else None,
                    "plan_hash": state.plan.plan_hash if state.plan else None,
                },
            )
            rec.detail["preapproval_count"] = len(preapproval_targets)
            rec.detail["low_confidence_count"] = len(triggered_low_confidence)
            rec.detail["policy_forced_count"] = len(triggered_policy)
        return state
