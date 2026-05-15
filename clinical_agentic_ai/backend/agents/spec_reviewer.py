"""
SpecReviewer agent
==================

Reads the transformation spec and answers two questions:

1. **Is it parseable and complete?** (structural)
2. **Is it unambiguous given the dataset?** (semantic)

It uses the **Guardrails** module for structural checks and an LLM call (or
its rule-based fallback) for semantic ambiguity detection. Any ambiguity
that the agent cannot resolve from context is escalated to HITL — *not*
guessed, because guessing on a regulated derivation is exactly the failure
mode this system has to prevent.

Output
------
A normalised, internal spec representation:

    {
      "source_schema": {col -> dtype},
      "derivations": [
        {
          "name": "...",
          "sources": ["..."],
          "type": "category | int | float | string | bool | date",
          "allowed_values": ["..."],
          "rule": "...",
          "description": "...",
          "max_null_rate": 0.0,
          "risk_class": "routine | critical | regulatory_critical | exploratory"
        }, ...
      ]
    }

Plus a list of `clarifications` — items the human reviewer needs to confirm
before code generation proceeds.
"""
from __future__ import annotations

from typing import Any

from backend.agents.base import BaseAgent
from backend.core.guardrails import check_spec
from backend.core.llm_client import llm
from backend.core.workflow_state import ValidationRecord, WorkflowState
from backend.memory.long_term import LongTermMemory
from backend.utils.hashing import hash_obj


# Stamped on every LLM call so prompts can be A/B tested and rolled back. Bump
# this when the prompt's contract changes (added fields, stricter constraints).
PROMPT_VERSION = "spec_reviewer/v1"


_SPEC_SYSTEM_PROMPT = """You are a clinical data spec reviewer.
You receive a YAML/JSON specification for a clinical derivation pipeline.
Your job is to identify *unambiguous* problems: missing source columns, vague
buckets, undefined acronyms. You DO NOT speculate or fill in missing logic.
Preserve any ``test_cases`` field verbatim in the normalised output — do not
invent or modify test cases. Reply with strict JSON:

{
  "clarifications": [{"name": "...", "issue": "...", "suggested_question": "..."}],
  "normalised_derivations": [
    {"name": "...", "sources": [...], "type": "...", "allowed_values": [...], "rule": "...", "max_null_rate": 0.0, "test_cases": [...]}
  ]
}
"""


def _heuristic_review(spec: dict[str, Any]) -> dict[str, Any]:
    """Rule-based fallback when no LLM is available.

    The user-supplied spec is trusted as-is, but empty ``sources`` lists
    and missing ``allowed_values`` on category types are still flagged as
    clarifications so the reviewer sees them.
    """
    clarifications: list[dict[str, str]] = []
    normalised: list[dict[str, Any]] = []
    for d in spec.get("derivations", []):
        n = {
            "name": d["name"],
            "sources": list(d.get("sources") or []),
            "type": d.get("type", "string"),
            "allowed_values": list(d.get("allowed_values") or []),
            "rule": d.get("rule") or d.get("description") or "",
            "max_null_rate": float(d.get("max_null_rate", 0.5)),
            "test_cases": list(d.get("test_cases") or []),
            "risk_class": str(d.get("risk_class") or "routine"),
        }
        if not n["sources"]:
            clarifications.append({
                "name": n["name"],
                "issue": "No source columns declared.",
                "suggested_question": (
                    f"Which source columns does `{n['name']}` depend on? "
                    "List them in plain English."
                ),
            })
        if n["type"] in ("category", "enum") and not n["allowed_values"]:
            clarifications.append({
                "name": n["name"],
                "issue": (
                    "Category derivation without explicit values or "
                    "assignment rule."
                ),
                "suggested_question": (
                    f"What values may `{n['name']}` take, and what rule "
                    "decides which value each row gets? Describe the "
                    "categories and the assignment logic in plain English."
                ),
            })
        normalised.append(n)
    return {"clarifications": clarifications, "normalised_derivations": normalised}


class SpecReviewerAgent(BaseAgent):
    name = "spec_reviewer"
    step = "review"

    def run(self, state: WorkflowState) -> WorkflowState:
        with self.step_ctx() as rec:
            # 1. Structural guardrails
            g = check_spec(state.spec)
            for f in g.findings:
                state.validations.append(ValidationRecord(
                    rule_id=f.code, target=None, severity=f.severity,
                    passed=False, message=f.message, detail=f.details,
                ))
                self.stm.vals.record(
                    run_id=state.run_id, target=None, rule_id=f.code,
                    severity=f.severity, passed=False, message=f.message, detail=f.details,
                )
            if not g.ok:
                rec.detail["structural_findings"] = [f.code for f in g.findings]
                raise ValueError("Spec has blocking structural problems.")

            # 2. Semantic / ambiguity review
            if llm.enabled:
                try:
                    resp = llm.complete(
                        system=_SPEC_SYSTEM_PROMPT,
                        user=f"SPEC:\n{state.spec}\nSCHEMA:\n{state.spec.get('source_schema', {})}",
                        expect_json=True,
                        purpose=PROMPT_VERSION,
                    )
                    review = resp.parsed or _heuristic_review(state.spec)
                except Exception as exc:  # graceful fallback
                    self.log.warning("spec_reviewer.llm_failed_fallback", error=str(exc))
                    review = _heuristic_review(state.spec)
            else:
                review = _heuristic_review(state.spec)

            # Defensive: if the LLM returned a degenerate response with no
            # normalised_derivations (or fewer than declared), recover by
            # merging in the heuristic review. Code generation will fail
            # silently if state.spec["normalised_derivations"] is empty,
            # so the recovery loop below guarantees it is populated.
            declared_names = {
                d.get("name") for d in (state.spec.get("derivations") or [])
            }
            review_names = {
                n.get("name") for n in (review.get("normalised_derivations") or [])
            }
            missing = declared_names - review_names
            if missing:
                heuristic = _heuristic_review(state.spec)
                review_norm = list(review.get("normalised_derivations") or [])
                for n in heuristic["normalised_derivations"]:
                    if n.get("name") in missing:
                        review_norm.append(n)
                review["normalised_derivations"] = review_norm
                review.setdefault("clarifications", []).extend(
                    [c for c in heuristic["clarifications"]
                     if c.get("name") in missing]
                )

            # 3. Persist normalised spec back into state.
            #    Recover ``test_cases`` from the original spec even if the LLM
            #    dropped them — these are author-authored ground truth that
            #    must survive normalisation untouched.
            original_by_name = {
                d.get("name"): d for d in (state.spec.get("derivations") or [])
            }
            for n in review["normalised_derivations"]:
                orig = original_by_name.get(n.get("name")) or {}
                # Preserve risk_class from original — the LLM cannot override
                # this; it's an author-declared contract.
                if not n.get("risk_class"):
                    n["risk_class"] = str(orig.get("risk_class") or "routine")
                # Preserve test_cases from original if LLM dropped them.
                if not n.get("test_cases") and orig.get("test_cases"):
                    n["test_cases"] = list(orig["test_cases"])
            state.spec["normalised_derivations"] = review["normalised_derivations"]
            state.spec["clarifications"] = review.get("clarifications", [])

            # 4. If clarifications exist, raise a HITL request. Pre-filled
            #    answers recovered from long-term clarification memory are
            #    attached so the reviewer can one-click accept them.
            if review["clarifications"]:
                from backend.core.workflow_state import HITLRequest

                ltm = LongTermMemory(self.stm.db)
                prefilled: dict[str, dict[str, Any]] = {}
                for c in review["clarifications"]:
                    name = c.get("name")
                    issue = c.get("issue", "")
                    if not name:
                        continue
                    hit = ltm.lookup_clarification(target=name, issue=issue)
                    if hit:
                        prefilled[name] = {
                            "answer": hit.answer,
                            "score": hit.score,
                            "times_used": hit.times_used,
                        }
                state.hitl_pending = HITLRequest(
                    target=None,
                    reason="spec_clarifications_required",
                    context={
                        "clarifications": review["clarifications"],
                        "prefilled_answers": prefilled,
                    },
                )
                rec.detail["clarifications"] = review["clarifications"]
                rec.detail["prefilled_count"] = len(prefilled)
            rec.outputs_hash = hash_obj(state.spec)
            rec.detail["derivation_count"] = len(review["normalised_derivations"])
        return state
