"""Shared pytest fixtures.

We set storage env vars at *module load* (before any ``backend.*`` import) so
the Settings singleton is built with the test paths from the start. Reloading
config later would leave already-imported agent modules holding a stale
reference, which manifests as monkeypatches not taking effect.
"""
from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any

# --- env wiring must happen before any backend import ---
_TMP = Path(tempfile.mkdtemp(prefix="agentic-test-"))
os.environ["DB_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["CHECKPOINT_DIR"] = str(_TMP / "ckpt")
os.environ["RUN_ARTIFACT_DIR"] = str(_TMP / "runs")
os.environ["MEMORY_DIR"] = str(_TMP / "memory")
os.environ["LOG_DIR"] = str(_TMP / "logs")

import pytest  # noqa: E402

from backend.db.session import init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def isolated_storage():
    """Initialise the test SQLite database once per session."""
    init_db()
    yield _TMP


@pytest.fixture
def clean_memory():
    """Wipe long-term + clarification memory before the test runs.

    Tests that exercise offline behaviour (Phase 4) or low-confidence
    triggering (Phase 5) need an empty memory store; otherwise patterns
    promoted by earlier tests in the session create memory hits that
    bypass the gates under test.
    """
    from backend.db.models import ClarificationMemory, MemoryPattern
    from backend.db.session import session_scope

    with session_scope() as db:
        db.query(MemoryPattern).delete()
        db.query(ClarificationMemory).delete()
    yield


@pytest.fixture
def sample_paths():
    root = Path(__file__).resolve().parents[1]
    return {
        "dataset": root / "data" / "samples" / "clinical_sample.csv",
        "spec": root / "data" / "specs" / "sample_spec.yaml",
        "golden": root / "data" / "golden" / "expected.csv",
    }


# Canonical correct ``derive`` functions for the sample spec. The stub LLM
# returns these when the test fixture is active so the end-to-end pipeline
# runs deterministically without a real API call.
_STUB_CODE: dict[str, str] = {
    "AGE_GROUP": textwrap.dedent("""\
        def derive(row):
            age = to_int(row["age"])
            if age is None: return None
            if age < 18: return "<18"
            if age < 65: return "18-64"
            return ">=65"
        """),
    "TREATMENT_DURATION": textwrap.dedent("""\
        def derive(row):
            d = days_between(row["visit_date"], row["treatment_start_date"])
            if d is None or d < 0: return None
            return int(d)
        """),
    "RESPONSE_FLAG": textwrap.dedent("""\
        def derive(row):
            r = row["response"]
            if isna(r): return None
            r = str(r).strip().upper()
            if r in ("CR", "PR"): return "RESPONDER"
            if r in ("SD", "PD"): return "NON_RESPONDER"
            return None
        """),
    "ANALYSIS_POP_FLAG": textwrap.dedent("""\
        def derive(row):
            age = to_int(row["age"])
            if age is None or age < 18: return "N"
            d = to_int(row["TREATMENT_DURATION"])
            if d is None or d < 1: return "N"
            return "Y"
        """),
    "RISK_GROUP": textwrap.dedent("""\
        def derive(row):
            lab = to_float(row["lab_value"])
            age = to_int(row["age"])
            if lab is None or age is None: return None
            score = 0
            if lab > 7.0: score += 2
            elif lab > 5.0: score += 1
            if age >= 65: score += 2
            elif age >= 50: score += 1
            if score >= 3: return "HIGH"
            if score >= 1: return "MEDIUM"
            return "LOW"
        """),
    "COMPOSITE_RISK_TIER": textwrap.dedent("""\
        def derive(row):
            if row["ANALYSIS_POP_FLAG"] != "Y":
                return "TIER_LOW"
            if row["RISK_GROUP"] == "HIGH":
                return "TIER_HIGH"
            if row["RISK_GROUP"] == "MEDIUM":
                return "TIER_MEDIUM"
            return "TIER_LOW"
        """),
    "COHORT_LABEL": textwrap.dedent("""\
        def derive(row):
            age = to_int(row["age"])
            sex = row["sex"]
            if age is None or isna(sex):
                return None
            sx = "M" if str(sex).strip().upper() == "M" else "F"
            if age < 30: return f"YOUNG_{sx}"
            if age < 60: return f"ADULT_{sx}"
            return f"SENIOR_{sx}"
        """),
}


@pytest.fixture
def stub_llm(monkeypatch):
    """Force the LLM client to return canned code for the 5 sample targets.

    Replaces ``backend.core.llm_client.llm.complete`` and flips
    ``llm.enabled`` to True for the duration of the test. The stub also
    handles spec-review and refinement system prompts.
    """
    from backend.core.llm_client import LLMResponse, llm

    def _fake_complete(*, system: str, user: str, expect_json: bool = True, purpose: str = "completion", **_kw):  # noqa: ARG001
        # Spec-review: no clarifications, no normalisation (spec_reviewer's
        # own defensive logic will fill in normalised_derivations from the
        # original spec).
        if "spec reviewer" in system.lower():
            return LLMResponse(
                text="{}", parsed={"clarifications": [], "normalised_derivations": []},
                tokens_in=0, tokens_out=0, latency_ms=0,
                model="stub-test", backend="stub",
            )
        # Code generation / refinement: look up target name in user message
        for name, code in _STUB_CODE.items():
            if f"target: {name}" in user:
                return LLMResponse(
                    text="{}",
                    parsed={"code": code, "confidence": 0.95, "uncertainty_notes": ""},
                    tokens_in=0, tokens_out=0, latency_ms=0,
                    model="stub-test", backend="stub",
                )
        return LLMResponse(
            text="{}",
            parsed={"code": "", "confidence": 0.0,
                    "uncertainty_notes": "stub had no canned answer"},
            tokens_in=0, tokens_out=0, latency_ms=0,
            model="stub-test", backend="stub",
        )

    monkeypatch.setattr(llm, "enabled", True)
    monkeypatch.setattr(llm, "complete", _fake_complete)
    yield llm


@pytest.fixture
def low_confidence_llm(monkeypatch):
    """LLM stub that returns CORRECT code but with a low confidence score —
    used to verify the Phase-5 early-HITL behaviour."""
    from backend.core.llm_client import LLMResponse, llm

    def _fake_complete(*, system: str, user: str, expect_json: bool = True, purpose: str = "completion", **_kw):  # noqa: ARG001
        if "spec reviewer" in system.lower():
            return LLMResponse(
                text="{}", parsed={"clarifications": [], "normalised_derivations": []},
                tokens_in=0, tokens_out=0, latency_ms=0,
                model="stub-test", backend="stub",
            )
        for name, code in _STUB_CODE.items():
            if f"target: {name}" in user:
                return LLMResponse(
                    text="{}",
                    parsed={"code": code, "confidence": 0.4,
                            "uncertainty_notes": "deliberately low for test"},
                    tokens_in=0, tokens_out=0, latency_ms=0,
                    model="stub-test", backend="stub",
                )
        return LLMResponse(text="{}", parsed={}, tokens_in=0, tokens_out=0,
                            latency_ms=0, model="stub-test", backend="stub")

    monkeypatch.setattr(llm, "enabled", True)
    monkeypatch.setattr(llm, "complete", _fake_complete)
    yield llm
