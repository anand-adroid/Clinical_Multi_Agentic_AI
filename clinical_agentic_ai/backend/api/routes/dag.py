"""DAG visualisation endpoint — emits Graphviz DOT for a run's dependency graph."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.memory.short_term import ShortTermMemory

router = APIRouter(prefix="/runs/{run_id}", tags=["dag"])


# Status -> fill colour. Stays in lock-step with the badges in the UI.
_COLOR = {
    "ok": "#bff7c1",        # green
    "refined": "#fde68a",   # amber
    "generated": "#bae6fd", # blue (pre-execute)
    "pending": "#e5e7eb",   # gray
    "failed": "#fecaca",    # red
    "unsafe": "#fecaca",
    "source": "#f3f4f6",    # neutral for raw source columns
}

# Generator -> border style. Visual cue at a glance.
_BORDER = {
    "llm": "solid",
    "rule": "dashed",
    "memory": "dotted",
    "human": "bold",
    "refiner": "bold",
}


@router.get("/dag.dot")
def dag_dot(run_id: str, db: Session = Depends(get_db)) -> dict:
    """Return DOT text for the dependency graph of `run_id`."""
    state = ShortTermMemory.restore(db, run_id)
    if not state:
        raise HTTPException(404, f"No state for run {run_id}")

    source_cols = set(state.spec.get("source_schema", {}).keys())
    derived = set(state.derivations.keys()) or set(state.spec.get("derivations") and
                  [d["name"] for d in state.spec.get("derivations") or []]) or set()

    lines = [
        "digraph Derivations {",
        '  rankdir=LR;',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11];',
        '  edge [color="#6b7280", arrowsize=0.7];',
    ]

    # Source columns (small, neutral)
    for s in sorted(source_cols):
        lines.append(
            f'  "{s}" [label="{s}\\n(source)", fillcolor="{_COLOR["source"]}", '
            f'color="#9ca3af", fontsize=10];'
        )

    # Derived targets
    for tgt in sorted(derived):
        d = state.derivations.get(tgt)
        status = d.status if d else "pending"
        gen = d.generator if d else "rule"
        fill = _COLOR.get(status, _COLOR["pending"])
        border = _BORDER.get(gen, "solid")
        label = f"{tgt}\\n[{gen}] · {status}"
        if d and d.attempt > 1:
            label += f" · try {d.attempt}"
        lines.append(
            f'  "{tgt}" [label="{label}", fillcolor="{fill}", '
            f'color="#111827", style="filled,rounded,{border}"];'
        )

    # Edges (source -> target)
    for tgt, srcs in (state.dag or {}).items():
        for s in srcs:
            lines.append(f'  "{s}" -> "{tgt}";')

    # If state.dag is empty (early phase), fall back to the spec.
    if not state.dag:
        for d in state.spec.get("derivations") or []:
            for s in d.get("sources") or []:
                lines.append(f'  "{s}" -> "{d["name"]}";')

    # Legend (cluster, drawn last so it sits in a corner)
    lines += [
        '  subgraph cluster_legend {',
        '    label="legend"; fontsize=10; color="#d1d5db"; style="dashed";',
        '    node [shape=box, style="filled,rounded", fontsize=9];',
        f'    L_llm   [label="LLM",     fillcolor="{_COLOR["ok"]}", style="filled,rounded,solid"];',
        f'    L_rule  [label="rule",    fillcolor="{_COLOR["ok"]}", style="filled,rounded,dashed"];',
        f'    L_mem   [label="memory",  fillcolor="{_COLOR["ok"]}", style="filled,rounded,dotted"];',
        f'    L_human [label="human",   fillcolor="{_COLOR["refined"]}", style="filled,rounded,bold"];',
        '    L_llm -> L_rule -> L_mem -> L_human [style=invis];',
        '  }',
        "}",
    ]
    return {"dot": "\n".join(lines)}
