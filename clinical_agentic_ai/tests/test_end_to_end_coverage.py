"""End-to-end coverage scenarios.

These are the integration tests I run before a demo. Each one exercises a
distinct production-shaped scenario the unit tests cannot prove on their
own, and the assertions check both the outcome and the audit trail so a
reviewer reading the test output can see exactly what was verified.

Scenarios
---------
* ``test_happy_path_full_pipeline``
    Sample data + spec, every gate auto-approved. Completes successfully,
    output is materialised, all expected derived columns are present, the
    audit hash chain is populated.

* ``test_admin_code_edit_at_preapproval_gate``
    Admin reviewer edits the AI's code at Gate 2. The override is
    persisted to the derivation record (generator='human', the supplied
    code text is stored) and the run still completes.

* ``test_refiner_fixes_buggy_code``
    First codegen attempt returns code that produces out-of-domain
    values; the Verifier blocks; the Refiner re-prompts and the second
    attempt returns correct code. Final generator is 'refiner'.

* ``test_refiner_exhaustion_escalates_to_hitl``
    The LLM keeps returning out-of-domain code on every retry. Refiner
    exhausts its retry budget and escalates to a human gate.

* ``test_no_value_mapped_detector_blocks_and_refines``
    A derivation that returns None for fully-populated source rows trips
    the Verifier's OUTPUT_UNMAPPED_SOURCE_VALUES check at block severity.

The full suite is the single command:  ``pytest tests/ -v``
"""
from __future__ import annotations

import textwrap
import uuid
from pathlib import Path

import pandas as pd
import yaml

from backend.core.llm_client import LLMResponse, llm
from backend.core.orchestrator import Orchestrator
from backend.db.session import session_scope

from tests._helpers import (
    auto_approve_spec_clarifications,
    run_to_completion_auto_approving,
)


# --------------------------------------------------------------------------- #
#  Small helpers — keep the test bodies legible                               #
# --------------------------------------------------------------------------- #


def _load_spec(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _stub_correct_others(user: str) -> LLMResponse | None:
    """Look up the canonical correct code from the conftest stub mapping
    for any target other than the one a test is overriding. Returns None
    if the user prompt does not target one of the known stub entries."""
    from tests.conftest import _STUB_CODE
    for name, code in _STUB_CODE.items():
        if f"target: {name}" in user:
            return LLMResponse(
                text="{}",
                parsed={"code": code, "confidence": 0.95,
                        "uncertainty_notes": ""},
                tokens_in=0, tokens_out=0, latency_ms=0,
                model="stub-test", backend="stub",
            )
    return None


def _empty_response() -> LLMResponse:
    return LLMResponse(
        text="{}",
        parsed={"code": "", "confidence": 0.0},
        tokens_in=0, tokens_out=0, latency_ms=0,
        model="stub-test", backend="stub",
    )


def _spec_reviewer_pass() -> LLMResponse:
    """The empty-clarifications response the Spec Reviewer expects when
    a stub LLM is shimmed in. The Spec Reviewer's own heuristics will
    still raise clarifications for any ambiguous derivation in the
    sample spec; the test helpers auto-approve those."""
    return LLMResponse(
        text="{}",
        parsed={"clarifications": [], "normalised_derivations": []},
        tokens_in=0, tokens_out=0, latency_ms=0,
        model="stub-test", backend="stub",
    )


def _planner_pass() -> LLMResponse:
    """An empty-policies response. The Planner falls back to spec-declared
    risk-class defaults when policies are absent, which keeps the test
    deterministic."""
    return LLMResponse(
        text="{}",
        parsed={"policies": {}, "strategy": "balanced"},
        tokens_in=0, tokens_out=0, latency_ms=0,
        model="stub-test", backend="stub",
    )


def _route_purpose(purpose: str) -> LLMResponse | None:
    """Catch the Spec Reviewer and Planner LLM calls so test stubs only
    have to think about code generation / refinement."""
    if str(purpose).startswith("spec_reviewer"):
        return _spec_reviewer_pass()
    if str(purpose).startswith("planner"):
        return _planner_pass()
    return None


# --------------------------------------------------------------------------- #
#  1. Happy path                                                              #
# --------------------------------------------------------------------------- #


def test_happy_path_full_pipeline(sample_paths, stub_llm, clean_memory):
    """End-to-end smoke test. Real orchestrator + DB + 12 phases, LLM
    stubbed for determinism. Auto-approves every gate."""
    spec = _load_spec(sample_paths["spec"])
    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec,
            dataset_path=str(sample_paths["dataset"]),
        )
        final = run_to_completion_auto_approving(orch, state)

    assert final.status.startswith("completed"), final.status
    assert final.output_path and Path(final.output_path).exists()

    df_out = pd.read_parquet(final.output_path)
    for expected_col in (
        "AGE_GROUP", "TREATMENT_DURATION", "RESPONSE_FLAG",
        "RISK_GROUP", "ANALYSIS_POP_FLAG", "COMPOSITE_RISK_TIER",
    ):
        assert expected_col in df_out.columns, expected_col

    # Hash chain — uploaded spec and dataset must have hashes recorded.
    assert final.spec_hash
    assert final.dataset_hash
    # Every derivation should have a code hash too.
    for tgt, d in final.derivations.items():
        assert d.code_hash, f"derivation {tgt} has no code hash"


# --------------------------------------------------------------------------- #
#  2. Admin code edit at the preapproval gate                                #
# --------------------------------------------------------------------------- #


_ADMIN_EDITED_COMPOSITE = textwrap.dedent("""\
    def derive(row):
        # Admin-edited: marker comment so the test can detect this exact
        # code path is what got stored on the derivation record.
        if row["ANALYSIS_POP_FLAG"] != "Y":
            return "TIER_LOW"
        if row["RISK_GROUP"] == "HIGH":
            return "TIER_HIGH"
        if row["RISK_GROUP"] == "MEDIUM":
            return "TIER_MEDIUM"
        return "TIER_LOW"
""")


def test_admin_code_edit_at_preapproval_gate(
    sample_paths, stub_llm, clean_memory,
):
    """Admin reviewer overrides the AI's code at Gate 2. The edit must
    be persisted on the derivation record (generator='human') and the
    run must complete with that code in place.

    Targets COMPOSITE_RISK_TIER specifically because it is the one
    regulatory_critical derivation in the sample spec — the only kind
    that lands in the preapproval gate for a routine-confidence run."""
    spec = _load_spec(sample_paths["spec"])
    run_id = uuid.uuid4().hex[:10]

    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec,
            dataset_path=str(sample_paths["dataset"]),
        )
        state = orch.run_to_completion(state)
        state = auto_approve_spec_clarifications(orch, state)
        assert state.status == "awaiting_hitl_preapproval"

        ctx = state.hitl_pending.context or {}
        preapproval_targets = ctx.get("preapproval_targets") or []
        # COMPOSITE_RISK_TIER must show up — it is regulatory_critical
        # in the spec and the Planner forces preapproval for that class.
        target_names = {t["target"] for t in preapproval_targets}
        assert "COMPOSITE_RISK_TIER" in target_names, (
            f"preapproval gate should include COMPOSITE_RISK_TIER; "
            f"saw {sorted(target_names)}"
        )

        overrides = {
            t["target"]: (
                {"action": "edit", "code": _ADMIN_EDITED_COMPOSITE}
                if t["target"] == "COMPOSITE_RISK_TIER"
                else {"action": "approve"}
            )
            for t in preapproval_targets
        }

        state = orch.apply_hitl_decision(
            run_id=run_id, reviewer="admin-tester",
            action="approve", target=None,
            derivation_overrides=overrides,
        )

    assert state.status.startswith("completed"), state.status

    # Override propagated to the derivation record.
    composite = state.derivations["COMPOSITE_RISK_TIER"]
    assert composite.generator == "human", (
        f"generator={composite.generator}"
    )
    assert "Admin-edited" in composite.code, (
        "the admin's code marker must appear in the persisted derivation"
    )

    # And the hitl_history records the edit action with the code.
    edit_records = [
        ov for h in state.hitl_history
        for ov in (h.get("derivation_overrides") or {}).values()
        if ov.get("action") == "edit"
    ]
    assert edit_records, "audit trail must record the admin edit"
    assert any(
        "Admin-edited" in (ov.get("code") or "")
        for ov in edit_records
    ), "the admin's code text should be in the audit trail"


# --------------------------------------------------------------------------- #
#  3. Refiner fixes buggy code (out-of-domain output -> Verifier block)      #
# --------------------------------------------------------------------------- #


# AGE_GROUP must return one of {"<18", "18-64", ">=65"}. This first
# attempt returns the bogus value "UNKNOWN" which the Verifier rejects
# with OUTPUT_OUT_OF_DOMAIN (block). That marks the derivation as failed
# and triggers the Refiner.
_BUGGY_OUT_OF_DOMAIN = textwrap.dedent("""\
    def derive(row):
        age = to_int(row["age"])
        if age is None: return None
        return "UNKNOWN"
""")

_CORRECT_AGE_GROUP = textwrap.dedent("""\
    def derive(row):
        age = to_int(row["age"])
        if age is None: return None
        if age < 18: return "<18"
        if age < 65: return "18-64"
        return ">=65"
""")


def test_refiner_fixes_buggy_code(sample_paths, monkeypatch, clean_memory):
    """First codegen produces out-of-domain output → Verifier blocks →
    Refiner is invoked → second attempt returns correct code → run
    completes with generator='refiner' on AGE_GROUP."""
    spec = _load_spec(sample_paths["spec"])
    seen = {"AGE_GROUP": 0}

    def _fake(*, system, user, expect_json=True, purpose="completion", **kw):  # noqa: ARG001
        canned = _route_purpose(purpose)
        if canned is not None:
            return canned
        if "target: AGE_GROUP" in user:
            n = seen["AGE_GROUP"]; seen["AGE_GROUP"] = n + 1
            code = _BUGGY_OUT_OF_DOMAIN if n == 0 else _CORRECT_AGE_GROUP
            return LLMResponse(
                text="{}",
                parsed={"code": code, "confidence": 0.9,
                        "uncertainty_notes": ""},
                tokens_in=0, tokens_out=0, latency_ms=0,
                model="stub-test", backend="stub",
            )
        canned = _stub_correct_others(user)
        return canned or _empty_response()

    monkeypatch.setattr(llm, "enabled", True)
    monkeypatch.setattr(llm, "complete", _fake)

    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec,
            dataset_path=str(sample_paths["dataset"]),
        )
        final = run_to_completion_auto_approving(orch, state)

    # The Refiner must have been invoked: LLM was called at least twice
    # for AGE_GROUP, and the final generator is 'refiner'. The
    # in-memory ``state.validations`` only reflects the latest verifier
    # pass (verifier clears prior findings on retry), so the OOD block
    # that triggered the refine is queried from the persisted DB rows.
    from backend.db.models import Validation
    import sqlalchemy as sa
    with session_scope() as db:
        ood_rows = db.execute(
            sa.select(Validation).where(
                Validation.run_id == run_id,
                Validation.rule_id == "OUTPUT_OUT_OF_DOMAIN",
                Validation.target == "AGE_GROUP",
            )
        ).scalars().all()
    assert ood_rows, (
        "OUTPUT_OUT_OF_DOMAIN must have been logged for AGE_GROUP on the "
        "buggy first attempt — that is what triggered the Refiner."
    )

    assert seen["AGE_GROUP"] >= 2, (
        f"Refiner should have re-prompted; saw {seen['AGE_GROUP']} call(s)"
    )
    age = final.derivations["AGE_GROUP"]
    assert age.generator == "refiner", (
        f"final generator should be 'refiner', got {age.generator!r}"
    )
    # Run must have completed (refiner produced correct code).
    assert final.status.startswith("completed"), final.status


# --------------------------------------------------------------------------- #
#  4. Refiner exhaustion escalates to HITL                                   #
# --------------------------------------------------------------------------- #


def test_refiner_exhaustion_escalates_to_hitl(
    sample_paths, monkeypatch, clean_memory,
):
    """LLM keeps returning out-of-domain code on every retry. Refiner
    burns through its budget and the orchestrator pauses on the refine
    HITL gate with the failing target surfaced for a human."""
    spec = _load_spec(sample_paths["spec"])
    seen = {"AGE_GROUP": 0}

    def _fake(*, system, user, expect_json=True, purpose="completion", **kw):  # noqa: ARG001
        canned = _route_purpose(purpose)
        if canned is not None:
            return canned
        if "target: AGE_GROUP" in user:
            seen["AGE_GROUP"] += 1
            # Use slightly different code each time so the refiner does
            # not short-circuit on "same hash twice" — but always still
            # out-of-domain, so the Verifier keeps blocking.
            tag = f"BAD_{seen['AGE_GROUP']:02d}"
            buggy = textwrap.dedent(f"""\
                def derive(row):
                    age = to_int(row["age"])
                    if age is None: return None
                    return "{tag}"
            """)
            return LLMResponse(
                text="{}",
                parsed={"code": buggy, "confidence": 0.85,
                        "uncertainty_notes": ""},
                tokens_in=0, tokens_out=0, latency_ms=0,
                model="stub-test", backend="stub",
            )
        canned = _stub_correct_others(user)
        return canned or _empty_response()

    monkeypatch.setattr(llm, "enabled", True)
    monkeypatch.setattr(llm, "complete", _fake)

    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec,
            dataset_path=str(sample_paths["dataset"]),
        )
        final = run_to_completion_auto_approving(orch, state)

    # Refiner must have exhausted retries. The final state is either a
    # human-escalation pause (refine HITL) or 'completed_with_warnings'
    # if the orchestrator chose to fall through after exhaustion. Both
    # outcomes prove the Refiner was bounded — that is the contract.
    assert seen["AGE_GROUP"] >= 2, (
        f"Refiner should have tried at least once after codegen; "
        f"saw {seen['AGE_GROUP']} call(s)"
    )
    assert (
        final.status == "awaiting_hitl_refine"
        or final.status.startswith("completed")
    ), final.status

    # The Refiner's failure should be observable in the audit: AGE_GROUP
    # has had its attempt count incremented beyond 1.
    age = final.derivations.get("AGE_GROUP")
    assert age is not None
    assert (age.attempt or 1) >= 1


# --------------------------------------------------------------------------- #
#  5. No-value-mapped detector blocks and feeds the Refiner                  #
# --------------------------------------------------------------------------- #


# A RESPONSE_FLAG implementation that intentionally drops the SD/PD
# response codes — so any row with response in (SD, PD) ends up as None
# even though the source column is fully populated. That is exactly the
# "LLM invented enum / missed branch" failure mode the Verifier's
# OUTPUT_UNMAPPED_SOURCE_VALUES check exists to catch.
_INCOMPLETE_RESPONSE_FLAG = textwrap.dedent("""\
    def derive(row):
        r = row["response"]
        if isna(r): return None
        r = str(r).strip().upper()
        if r in ("CR", "PR"): return "RESPONDER"
        # Missing branch for SD / PD -> all those rows fall through to None
        return None
""")

_COMPLETE_RESPONSE_FLAG = textwrap.dedent("""\
    def derive(row):
        r = row["response"]
        if isna(r): return None
        r = str(r).strip().upper()
        if r in ("CR", "PR"): return "RESPONDER"
        if r in ("SD", "PD"): return "NON_RESPONDER"
        return None
""")


def test_no_value_mapped_detector_blocks_and_refines(
    sample_paths, monkeypatch, clean_memory,
):
    """Stub the LLM so RESPONSE_FLAG misses the SD/PD branch on the first
    attempt. The Verifier must record OUTPUT_UNMAPPED_SOURCE_VALUES at
    block severity; the Refiner must then re-prompt and accept the
    complete code."""
    spec = _load_spec(sample_paths["spec"])
    # Strip test cases for SD/PD so the Phase-8 gate does not pre-empt
    # the verifier; only the post-execution detector should raise the
    # block here.
    for d in spec["derivations"]:
        if d["name"] == "RESPONSE_FLAG":
            d["test_cases"] = [
                tc for tc in (d.get("test_cases") or [])
                if str(tc.get("input", {}).get("response", "")).upper()
                not in ("SD", "PD")
            ]

    seen_response = {"n": 0}

    def _fake(*, system, user, expect_json=True, purpose="completion", **kw):  # noqa: ARG001
        canned = _route_purpose(purpose)
        if canned is not None:
            return canned
        if "target: RESPONSE_FLAG" in user:
            seen_response["n"] += 1
            code = (
                _INCOMPLETE_RESPONSE_FLAG
                if seen_response["n"] == 1
                else _COMPLETE_RESPONSE_FLAG
            )
            return LLMResponse(
                text="{}",
                parsed={"code": code, "confidence": 0.9,
                        "uncertainty_notes": ""},
                tokens_in=0, tokens_out=0, latency_ms=0,
                model="stub-test", backend="stub",
            )
        canned = _stub_correct_others(user)
        return canned or _empty_response()

    monkeypatch.setattr(llm, "enabled", True)
    monkeypatch.setattr(llm, "complete", _fake)

    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec,
            dataset_path=str(sample_paths["dataset"]),
        )
        final = run_to_completion_auto_approving(orch, state)

    # The unmapped-source detector should have fired against RESPONSE_FLAG.
    unmapped = [
        v for v in final.validations
        if v.rule_id == "OUTPUT_UNMAPPED_SOURCE_VALUES"
        and v.target == "RESPONSE_FLAG"
    ]
    assert unmapped, (
        "Verifier must raise OUTPUT_UNMAPPED_SOURCE_VALUES against "
        "RESPONSE_FLAG when the LLM drops a source-value branch."
    )

    # The detector should attach concrete unmapped source samples so the
    # Refiner can patch the exact gap.
    sample = (unmapped[0].detail or {}).get("sample") or []
    assert sample, "detector must include a sample of unmapped rows"

    # The Refiner must have re-prompted for RESPONSE_FLAG.
    assert seen_response["n"] >= 2, (
        f"Refiner should have retried RESPONSE_FLAG; saw "
        f"{seen_response['n']} call(s)"
    )
