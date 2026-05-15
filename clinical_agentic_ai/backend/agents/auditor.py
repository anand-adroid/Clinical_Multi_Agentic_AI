"""
Auditor agent
=============

Produces the human-facing audit/traceability report at the end of a run.

For every derived column it emits:
  * source columns (lineage)
  * generator (memory | llm | rule | refiner | human)
  * code hash + verbatim code
  * verification verdicts (rule_id -> pass/fail)
  * any HITL decisions tied to it
  * output_hash

The report is saved as JSON (machine) and Markdown (human) under
`storage/runs/<run_id>/audit/`. Both are deterministic — given the same DB
state the same files come out.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.agents.base import BaseAgent
from backend.core.config import settings
from backend.core.workflow_state import WorkflowState
from backend.utils.hashing import hash_obj


class AuditorAgent(BaseAgent):
    name = "auditor"
    step = "audit"

    def run(self, state: WorkflowState) -> WorkflowState:
        with self.step_ctx() as rec:
            report = self._build_report(state)
            out_dir = Path(settings.run_artifact_dir) / state.run_id / "audit"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            (out_dir / "audit.md").write_text(self._to_markdown(report), encoding="utf-8")
            rec.detail["report_path"] = str(out_dir / "audit.json")
            rec.outputs_hash = hash_obj(report)
            state.summary["audit_report"] = str(out_dir / "audit.json")
        return state

    # ---------------------------------------------------------------------

    def _build_report(self, state: WorkflowState) -> dict[str, Any]:
        # HITL decisions, keyed by target
        hitl_for_target: dict[str, list[dict[str, Any]]] = {}
        for h in self.stm.hitl.list_for_run(state.run_id):
            hitl_for_target.setdefault(h.target or "_global_", []).append({
                "reviewer": h.reviewer, "action": h.action,
                "comment": h.comment, "edited": bool(h.edited_code),
                "created_at": h.created_at.isoformat(),
            })

        # Validations, keyed by target
        v_for_target: dict[str, list[dict[str, Any]]] = {}
        for v in self.stm.vals.list_for_run(state.run_id):
            v_for_target.setdefault(v.target or "_global_", []).append({
                "rule_id": v.rule_id, "severity": v.severity,
                "passed": v.passed, "message": v.message, "detail": v.detail,
            })

        lineage: list[dict[str, Any]] = []
        for tgt in state.topo_order:
            d = state.derivations[tgt]
            lineage.append({
                "target": tgt,
                "sources": d.sources,
                "generator": d.generator,
                "attempts": d.attempt,
                "status": d.status,
                "code_hash": d.code_hash,
                "code": d.code,
                "confidence": d.confidence,
                "uncertainty_notes": d.uncertainty_notes,
                "reasoning": d.reasoning,
                "risk_class": d.risk_class,
                "null_count": d.null_count,
                "row_errors_sample": d.row_errors[:5],
                "validations": v_for_target.get(tgt, []),
                "hitl_decisions": hitl_for_target.get(tgt, []),
            })

        events_summary = [
            {"agent": e.agent, "step": e.step, "status": e.status,
             "started_at": e.started_at, "finished_at": e.finished_at,
             "duration_ms": e.duration_ms,
             "inputs_hash": e.inputs_hash, "outputs_hash": e.outputs_hash}
            for e in state.events
        ]

        return {
            "run_id": state.run_id,
            "status": state.status,
            "spec_hash": state.spec_hash,
            "dataset_hash": state.dataset_hash,
            "output_path": state.output_path,
            "topo_order": state.topo_order,
            "events": events_summary,
            "lineage": lineage,
            "global_validations": v_for_target.get("_global_", []),
            "global_hitl": hitl_for_target.get("_global_", []),
            "summary": state.summary,
            "config": {
                "llm_enabled": settings.llm_enabled,
                "model": settings.anthropic_model if settings.llm_enabled else "rule-based",
                "max_refine_retries": settings.max_refine_retries,
            },
        }

    def _to_markdown(self, r: dict[str, Any]) -> str:
        lines = [
            f"# Audit report — run `{r['run_id']}`",
            "",
            f"- **Status:** {r['status']}",
            f"- **Spec hash:** `{r['spec_hash']}`",
            f"- **Dataset hash:** `{r['dataset_hash']}`",
            f"- **Output:** `{r['output_path']}`",
            f"- **LLM enabled:** {r['config']['llm_enabled']}  (model: {r['config']['model']})",
            "",
            "## Topological execution order",
            "",
            " -> ".join(r["topo_order"]) or "_empty_",
            "",
            "## Agent timeline",
            "",
            "| # | Agent | Step | Status | Duration (ms) | Outputs hash |",
            "|---|-------|------|--------|---------------|---------------|",
        ]
        for i, e in enumerate(r["events"], start=1):
            lines.append(
                f"| {i} | `{e['agent']}` | `{e['step']}` | {e['status']} | "
                f"{e['duration_ms']} | `{(e['outputs_hash'] or '')[:10]}` |"
            )
        lines += ["", "## Lineage per derived variable", ""]
        for item in r["lineage"]:
            conf_str = (
                f"{item.get('confidence'):.2f}"
                if item.get("confidence") is not None else "—"
            )
            lines += [
                f"### `{item['target']}`",
                f"- **Sources:** {item['sources']}",
                f"- **Risk class:** {item.get('risk_class', 'routine')}",
                f"- **Generator:** {item['generator']} (attempts: {item['attempts']})",
                f"- **Confidence:** {conf_str}",
                f"- **Status:** {item['status']}",
                f"- **Code hash:** `{item['code_hash'][:12]}`",
                f"- **Null count:** {item['null_count']}",
                "",
            ]
            if item.get("reasoning"):
                lines += [f"**Reasoning (LLM):** {item['reasoning']}", ""]
            if item.get("uncertainty_notes"):
                lines += [f"**Uncertainty:** {item['uncertainty_notes']}", ""]
            lines += [
                "**Code:**",
                "```python",
                item["code"],
                "```",
                "",
            ]
            if item["validations"]:
                lines.append("**Validations:**")
                lines.append("")
                for v in item["validations"]:
                    flag = "PASS" if v["passed"] else "FAIL"
                    lines.append(f"- {flag} `{v['rule_id']}` ({v['severity']}): {v['message']}")
                lines.append("")
            if item["hitl_decisions"]:
                lines.append("**HITL decisions:**")
                lines.append("")
                for h in item["hitl_decisions"]:
                    lines.append(
                        f"- {h['reviewer']} -> **{h['action']}** at {h['created_at']}"
                        + (f": _{h['comment']}_" if h.get("comment") else "")
                    )
                lines.append("")
        return "\n".join(lines)
