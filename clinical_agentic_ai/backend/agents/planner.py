"""Planner Agent — the LLM-driven run-plan generator.
When the LLM is unavailable the planner falls back to a rule-based policy
keyed on the spec-declared ``risk_class`` field. That means the system
still has *something* opinionated to say about each derivation, even
offline.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from backend.agents.base import BaseAgent
from backend.core.config import settings
from backend.core.llm_client import LLMError, llm
from backend.core.workflow_state import (
    DerivationPolicy,
    RunPlan,
    WorkflowState,
)
from backend.utils.hashing import hash_obj


PROMPT_VERSION = "planner/v1"


_PLANNER_SYSTEM_PROMPT = """You are a senior clinical data platform planner.
You receive a normalised derivation specification and must produce a per-
derivation execution plan that downstream agents will follow.

For each derivation decide:
  - risk_class: one of [routine, critical, regulatory_critical, exploratory].
    Use the spec's declared risk_class if present, but you may upgrade it
    (never downgrade) if the derivation depends on other risky derivations
    or has complex multi-step logic.
  - require_preapproval: true if the generated code MUST be human-reviewed
    before execution, regardless of LLM confidence. Set this for
    regulatory_critical derivations and for derivations whose rule is
    inherently judgement-laden.
  - require_signoff: true if the FINAL output for this column must be
    human-signed-off before release. Set this for regulatory endpoints.
  - confidence_threshold: a float in [0.5, 0.95]. Below this, the
    derivation triggers HITL. Lower for routine work, higher for critical.
  - reviewer_tier: T1 (analyst), T2 (domain expert), or T3 (senior).
  - rationale: 1-2 sentences clearly explaining your choices for THIS derivation.

Also produce an overall ``strategy``:
  - "cautious": many gates, high review intensity
  - "balanced": default
  - "fast": few gates, only required where risk is high

Reply with strict JSON:
{
  "strategy": "balanced",
  "rationale": "overall reasoning",
  "policies": {
    "DERIVATION_NAME": {
      "risk_class": "critical",
      "require_preapproval": true,
      "require_signoff": false,
      "confidence_threshold": 0.85,
      "reviewer_tier": "T2",
      "rationale": "..."
    },
    ...
  }
}
"""


_RISK_DEFAULTS: dict[str, dict[str, Any]] = {
    "exploratory": {
        "require_preapproval": False,
        "require_signoff": False,
        "confidence_threshold": 0.5,
        "reviewer_tier": "T1",
    },
    "routine": {
        "require_preapproval": False,
        "require_signoff": False,
        "confidence_threshold": 0.7,
        "reviewer_tier": "T1",
    },
    "critical": {
        "require_preapproval": False,
        "require_signoff": False,
        "confidence_threshold": 0.85,
        "reviewer_tier": "T2",
    },
    "regulatory_critical": {
        "require_preapproval": True,
        "require_signoff": True,
        "confidence_threshold": 0.95,
        "reviewer_tier": "T3",
    },
}


def _deterministic_plan(spec: dict[str, Any]) -> RunPlan:
    """Build a plan purely from spec-declared ``risk_class`` fields.

    Used when the LLM is unavailable (offline mode, retired key, etc.)
    so the planner contract still holds.
    """
    policies: dict[str, DerivationPolicy] = {}
    for d in spec.get("normalised_derivations") or spec.get("derivations") or []:
        target = d["name"]
        risk = str(d.get("risk_class") or "routine")
        if risk not in _RISK_DEFAULTS:
            risk = "routine"
        defaults = _RISK_DEFAULTS[risk]
        policies[target] = DerivationPolicy(
            target=target,
            risk_class=risk,
            require_preapproval=bool(defaults["require_preapproval"]),
            require_signoff=bool(defaults["require_signoff"]),
            confidence_threshold=float(defaults["confidence_threshold"]),
            reviewer_tier=str(defaults["reviewer_tier"]),
            rationale=(
                f"Deterministic policy from declared risk_class={risk!r}. "
                f"LLM planner unavailable."
            ),
        )
    plan = RunPlan(
        policies=policies,
        strategy="balanced",
        rationale=(
            "Deterministic fallback: applied risk-class defaults from spec. "
            "No LLM-driven re-classification."
        ),
    )
    plan.plan_hash = hash_obj({k: asdict(v) for k, v in policies.items()})
    return plan


class PlannerAgent(BaseAgent):
    name = "planner"
    step = "plan"

    def run(self, state: WorkflowState) -> WorkflowState:
        with self.step_ctx() as rec:
            normalised = state.spec.get("normalised_derivations") or []
            if not normalised:
                rec.detail["skipped"] = "no derivations"
                return state

            plan = self._llm_plan(state) if llm.enabled else None
            if plan is None:
                plan = _deterministic_plan(state.spec)
                rec.detail["source"] = "deterministic"
            else:
                rec.detail["source"] = "llm"

            state.plan = plan
            rec.detail["strategy"] = plan.strategy
            rec.detail["plan_hash"] = plan.plan_hash
            rec.detail["policy_count"] = len(plan.policies)
            rec.detail["regulatory_critical_count"] = sum(
                1 for p in plan.policies.values()
                if p.risk_class == "regulatory_critical"
            )
            self.stm.audit.record(
                run_id=state.run_id, actor="planner", actor_type="agent",
                action="plan.created",
                detail={
                    "strategy": plan.strategy,
                    "plan_hash": plan.plan_hash,
                    "source": rec.detail["source"],
                    "rationale": plan.rationale,
                },
            )
        return state

    # ----------------------------- helpers -----------------------------

    def _llm_plan(self, state: WorkflowState) -> RunPlan | None:
        """Ask the LLM to produce a per-derivation policy plan. Returns
        ``None`` on any failure; the caller falls back to the deterministic
        plan so the contract always holds."""
        normalised = state.spec.get("normalised_derivations") or []
        try:
            payload = {
                "strategy_hint": "balanced",
                "derivations": [
                    {
                        "name": d.get("name"),
                        "sources": d.get("sources", []),
                        "type": d.get("type"),
                        "allowed_values": d.get("allowed_values"),
                        "rule": d.get("rule", ""),
                        "risk_class": d.get("risk_class", "routine"),
                    }
                    for d in normalised
                ],
            }
            resp = llm.complete(
                system=_PLANNER_SYSTEM_PROMPT,
                user=json.dumps(payload, indent=2),
                expect_json=True,
                purpose=PROMPT_VERSION,
            )
        except LLMError as exc:
            self.log.warning("planner.llm_failed", error=str(exc))
            return None

        parsed = resp.parsed or {}
        raw_policies = parsed.get("policies") or {}
        if not raw_policies:
            self.log.warning("planner.empty_policies")
            return None

        policies: dict[str, DerivationPolicy] = {}
        declared_names = {d["name"] for d in normalised}
        declared_risk = {
            d["name"]: str(d.get("risk_class") or "routine") for d in normalised
        }
        for name in declared_names:
            raw = raw_policies.get(name) or {}
            # Risk-class can only be upgraded by the LLM, never downgraded.
            llm_risk = str(raw.get("risk_class") or declared_risk.get(name, "routine"))
            risk = _pick_higher_risk(declared_risk.get(name, "routine"), llm_risk)
            defaults = _RISK_DEFAULTS.get(risk, _RISK_DEFAULTS["routine"])
            try:
                threshold = float(raw.get("confidence_threshold", defaults["confidence_threshold"]))
            except (TypeError, ValueError):
                threshold = float(defaults["confidence_threshold"])
            threshold = max(0.5, min(0.95, threshold))
            policies[name] = DerivationPolicy(
                target=name,
                risk_class=risk,
                require_preapproval=bool(raw.get("require_preapproval", defaults["require_preapproval"])),
                require_signoff=bool(raw.get("require_signoff", defaults["require_signoff"])),
                confidence_threshold=threshold,
                reviewer_tier=str(raw.get("reviewer_tier") or defaults["reviewer_tier"]),
                rationale=(raw.get("rationale") or "").strip()[:500],
            )
        strategy = str(parsed.get("strategy") or "balanced")
        if strategy not in ("cautious", "balanced", "fast"):
            strategy = "balanced"
        plan = RunPlan(
            policies=policies,
            strategy=strategy,
            rationale=(parsed.get("rationale") or "").strip()[:1000],
        )
        plan.plan_hash = hash_obj({k: asdict(v) for k, v in policies.items()})
        return plan


_RISK_ORDER = ["exploratory", "routine", "critical", "regulatory_critical"]


def _pick_higher_risk(a: str, b: str) -> str:
    """Return the higher of two risk classes (LLM cannot downgrade declared)."""
    ia = _RISK_ORDER.index(a) if a in _RISK_ORDER else 1
    ib = _RISK_ORDER.index(b) if b in _RISK_ORDER else 1
    return _RISK_ORDER[max(ia, ib)]
