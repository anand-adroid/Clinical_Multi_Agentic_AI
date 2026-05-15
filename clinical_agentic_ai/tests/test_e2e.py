"""End-to-end workflow test — no LLM, fully deterministic via fallbacks."""
from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import yaml

from backend.core.orchestrator import Orchestrator
from backend.db.session import session_scope
from backend.eval.evaluator import evaluate_run

from tests._helpers import run_to_completion_auto_approving


def test_full_pipeline_runs_clean_on_sample(sample_paths, stub_llm):
    spec = yaml.safe_load(Path(sample_paths["spec"]).read_text())
    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id,
            spec=spec,
            dataset_path=str(sample_paths["dataset"]),
        )
        final = run_to_completion_auto_approving(orch, state)

    assert final.status.startswith("completed"), f"status was {final.status}"
    assert final.output_path
    df_out = pd.read_parquet(final.output_path)
    expected_cols = ["AGE_GROUP", "TREATMENT_DURATION", "RESPONSE_FLAG",
                     "RISK_GROUP", "ANALYSIS_POP_FLAG", "COMPOSITE_RISK_TIER"]
    for c in expected_cols:
        assert c in df_out.columns
    # Sanity: the topo order respects derived dependencies — ANALYSIS_POP_FLAG
    # must come AFTER TREATMENT_DURATION; COMPOSITE_RISK_TIER must come AFTER
    # all three of its derived inputs.
    topo = final.topo_order
    assert topo.index("TREATMENT_DURATION") < topo.index("ANALYSIS_POP_FLAG")
    for src in ("RISK_GROUP", "RESPONSE_FLAG", "ANALYSIS_POP_FLAG"):
        assert topo.index(src) < topo.index("COMPOSITE_RISK_TIER")

    result = evaluate_run(final.output_path, str(sample_paths["golden"]),
                          list(final.derivations.keys()))
    # COHORT_LABEL is deliberately ambiguous in the spec and is not part of
    # the hand-authored golden table — coverage is therefore 6/7 by design.
    assert result["coverage"] >= 0.85
    assert result["correctness"] >= 0.95


def test_audit_trail_is_complete(sample_paths, stub_llm):
    from backend.db.repositories import (
        AuditRepository,
        DerivationRepository,
        EventRepository,
        ValidationRepository,
    )

    spec = yaml.safe_load(Path(sample_paths["spec"]).read_text())
    run_id = uuid.uuid4().hex[:10]
    with session_scope() as db:
        orch = Orchestrator(db)
        state = orch.create_run(
            run_id=run_id, spec=spec,
            dataset_path=str(sample_paths["dataset"]),
        )
        state = run_to_completion_auto_approving(orch, state)

    agents_seen = []
    rule_ids = []
    target_names = []
    actor_types = []
    with session_scope() as db:
        for ev in EventRepository(db).list_for_run(run_id):
            agents_seen.append(str(ev.agent))
        for v in ValidationRepository(db).list_for_run(run_id):
            rule_ids.append(str(v.rule_id))
        for d in DerivationRepository(db).list_for_run(run_id):
            target_names.append(str(d.target))
        for a in AuditRepository(db).list_for_run(run_id):
            actor_types.append(str(a.actor_type))

    required = ["spec_reviewer", "dag_builder", "code_generator",
                "code_preapproval", "static_validator", "test_runner",
                "executor", "verifier", "auditor"]
    for r in required:
        assert r in agents_seen, "missing agent in events: " + r
    assert len(target_names) >= 5
    assert "INPUT_GUARD_DONE" in rule_ids
    assert "system" in actor_types
