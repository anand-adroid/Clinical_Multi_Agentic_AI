"""
DAGBuilder agent — builds the dependency graph and the deterministic
topological order of derived variables.

Reasons this is its own agent rather than a utility function:

  - It is the gate between "spec" and "execution". A cycle here means the
    entire run is doomed; failing loudly with a precise message is cheaper
    than crashing mid-code-generation with a confusing stack trace.
  - It carries explicit lineage information (source columns per derived
    target) into ``state.events`` for the audit trail.
  - The DAG primitive is reused by the Executor (topo order),
    CodePreApproval (level grouping), and the UI (visualisation).

Cycle detection uses Kahn's algorithm; topological order is stable with an
alphabetic tiebreaker so the same spec always produces the same execution
order across runs. That stability is what lets two reviewers compare two
runs of the same spec side-by-side and reason about differences.
"""
from __future__ import annotations

from collections import defaultdict, deque

from backend.agents.base import BaseAgent
from backend.core.workflow_state import WorkflowState


class DAGBuildError(ValueError):
    pass


def _build_topo(dag: dict[str, list[str]], targets: set[str]) -> list[str]:
    indeg: dict[str, int] = {t: 0 for t in targets}
    rev: dict[str, list[str]] = defaultdict(list)
    for t, srcs in dag.items():
        for s in srcs:
            if s in targets:  # dep on another *derived* var
                indeg[t] += 1
                rev[s].append(t)

    ready: deque[str] = deque(sorted(t for t, d in indeg.items() if d == 0))
    order: list[str] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for nxt in sorted(rev[node]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)

    if len(order) != len(targets):
        remaining = [t for t in targets if t not in set(order)]
        raise DAGBuildError(f"Cycle detected involving derivations: {remaining}")
    return order


class DAGBuilderAgent(BaseAgent):
    name = "dag_builder"
    step = "build_dag"

    def run(self, state: WorkflowState) -> WorkflowState:
        with self.step_ctx() as rec:
            derivations = state.spec.get("normalised_derivations") or state.spec.get("derivations", [])
            targets = {d["name"] for d in derivations}
            dag: dict[str, list[str]] = {d["name"]: list(d.get("sources", [])) for d in derivations}

            # Validate all non-derived sources exist in the dataset schema.
            schema = state.spec.get("source_schema", {})
            schema_cols = set(schema.keys())
            unknown_refs: dict[str, list[str]] = {}
            for tgt, srcs in dag.items():
                missing = [s for s in srcs if s not in schema_cols and s not in targets]
                if missing:
                    unknown_refs[tgt] = missing
            if unknown_refs:
                raise DAGBuildError(
                    f"Derivations reference unknown columns: {unknown_refs}"
                )

            order = _build_topo(dag, targets)
            state.dag = dag
            state.topo_order = order
            rec.detail["topo_order"] = order
            rec.detail["edges"] = sum(len(v) for v in dag.values())
        return state
