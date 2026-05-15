"""HITL Quality — measure the review process itself.

Admin-only. Industry-grade AI systems do not just put humans in the loop;
they measure whether the loop is working. This page surfaces the five
metrics every regulated AI platform reports on:

  - Intervention rate: are reviewers actually changing things?
  - False-alert rate: are reviewers rubber-stamping?
  - Memory reuse: is the system getting smarter?
  - Decision latency: are SLAs met?
  - Overconfidence rate: when LLM says high confidence, was it actually right?
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import (
    api_get,
    page_header,
)
from components.auth import require_admin


require_admin()


page_header(
    "HITL Quality",
    "Process metrics for the human-in-the-loop review pipeline. Tracks "
    "whether reviewers are catching real problems, the system is learning "
    "from past decisions, and SLA targets are met.",
)

col_a, col_b = st.columns([1, 4])
with col_a:
    window = st.selectbox(
        "Window",
        options=[7, 30, 90, 365],
        index=1,
        format_func=lambda d: f"Last {d} days",
    )

data = api_get(f"/hitl-quality/summary", days=window) or {}
headline = data.get("headline") or {}
counts = data.get("counts") or {}
targets = data.get("targets") or {}


# ----- Headline metrics -----
def _metric_card(col, label, value, target_text, color):
    col.markdown(
        f"<div style='border:1px solid #E2E8F0;border-radius:8px;padding:14px;height:130px;'>"
        f"<div style='color:#64748B;font-size:0.85em;'>{label}</div>"
        f"<div style='color:{color};font-size:1.8em;font-weight:600;margin:6px 0;'>{value}</div>"
        f"<div style='color:#94A3B8;font-size:0.75em;'>target: {target_text}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _color_for_rate(value, ok_max=None, ok_min=None):
    if value is None:
        return "#64748B"
    if ok_max is not None and value > ok_max:
        return "#B91C1C"
    if ok_min is not None and value < ok_min:
        return "#B45309"
    return "#15803D"


c1, c2, c3, c4 = st.columns(4)
ir = headline.get("intervention_rate_pct", 0.0)
_metric_card(
    c1, "Intervention rate", f"{ir}%",
    targets.get("intervention_rate_pct", "10–30%"),
    _color_for_rate(ir, ok_max=70, ok_min=5),
)
far = headline.get("false_alert_rate_pct", 0.0)
_metric_card(
    c2, "False-alert rate", f"{far}%",
    targets.get("false_alert_rate_pct", "below 50"),
    _color_for_rate(far, ok_max=50),
)
mrr = headline.get("memory_reuse_rate_pct", 0.0)
_metric_card(
    c3, "Memory reuse rate", f"{mrr}%",
    targets.get("memory_reuse_rate_pct", "grows over time"),
    "#15803D" if mrr > 0 else "#64748B",
)
ocr = headline.get("overconfidence_rate_pct", 0.0)
_metric_card(
    c4, "Overconfidence rate", f"{ocr}%",
    targets.get("overconfidence_rate_pct", "below 5"),
    _color_for_rate(ocr, ok_max=20),
)

st.write("")
c5, c6, c7, c8 = st.columns(4)
p50 = headline.get("decision_latency_p50_ms", 0.0)
p95 = headline.get("decision_latency_p95_ms", 0.0)
_metric_card(
    c5, "Decision latency P50",
    f"{p50/1000:.1f}s" if p50 < 60_000 else f"{p50/60_000:.1f}min",
    "gate SLA", "#0F172A",
)
_metric_card(
    c6, "Decision latency P95",
    f"{p95/1000:.1f}s" if p95 < 60_000 else f"{p95/60_000:.1f}min",
    "gate SLA", "#0F172A",
)
_metric_card(
    c7, "Runs (window)",
    str(counts.get("total_runs", 0)),
    "—", "#0F172A",
)
_metric_card(
    c8, "Runs with HITL",
    str(counts.get("runs_with_hitl", 0)),
    "—", "#0F172A",
)

st.divider()

# ----- Counts breakdown -----
st.subheader("Breakdown")
b1, b2 = st.columns(2)
with b1:
    st.markdown("**HITL actions taken**")
    st.dataframe(
        pd.DataFrame([
            {"action": "approve (clean)", "count": counts.get("pure_approves", 0)},
            {"action": "edit / regenerate", "count": counts.get("edits", 0)},
            {"action": "reject", "count": counts.get("rejects", 0)},
            {"action": "total decisions", "count": counts.get("total_hitl_decisions", 0)},
        ]),
        use_container_width=True,
        hide_index=True,
    )
with b2:
    st.markdown("**Memory state**")
    st.dataframe(
        pd.DataFrame([
            {"store": "Pattern memory (code)", "rows": counts.get("memory_patterns", 0)},
            {"store": "Clarification memory", "rows": counts.get("clarification_memory", 0)},
            {"store": "Derivations served from memory", "rows": counts.get("memory_served_derivations", 0)},
            {"store": "Total derivations in window", "rows": counts.get("total_derivations", 0)},
        ]),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ----- Recent decisions feed -----
st.subheader("Recent HITL decisions")
recent = api_get("/hitl-quality/recent-decisions", limit=50) or []
if not recent:
    st.caption("No HITL decisions recorded in this database yet.")
else:
    rows = []
    for r in recent:
        structured = r.get("structured_payload") or {}
        clarifications = structured.get("clarification_answers")
        overrides = structured.get("derivation_overrides")
        n_clarifs = len(clarifications or {})
        n_overrides = len(overrides or {})
        rows.append({
            "when": r.get("created_at", "")[:19].replace("T", " "),
            "run_id": (r.get("run_id") or "")[:10],
            "reviewer": r.get("reviewer"),
            "action": r.get("action"),
            "target": r.get("target") or "—",
            "clarifications": n_clarifs,
            "overrides": n_overrides,
            "edited_code": "yes" if r.get("edited") else "—",
            "comment": (r.get("comment") or "")[:80],
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ----- Methodology -----
with st.expander("Methodology — how each metric is computed"):
    st.markdown(
        """
**Intervention rate**
`(edits + regenerates + rejects) / total HITL decisions`
A pure rubber-stamp culture shows ~0%. A broken AI shows ~90%. A healthy
system sits between 10–30%: reviewers add value but the AI is competent.

**False-alert rate**
`% of decisions resolved with 'approve' AND no edited code AND no comment`
High values indicate the gate is firing too eagerly — confidence thresholds
should be lowered or risk routing recalibrated.

**Memory reuse rate**
`% of derivations served from LTM (vs new LLM call)`
Monotonically grows over time as the system matures. Each successful run
promotes its derivations to LTM; future runs with similar rules get
memory hits and skip the LLM entirely.

**Decision latency (P50, P95)**
Time between a HITL pause being raised (audit row `hitl.requested`) and
the matching reviewer decision being applied. Compared against per-gate SLAs.

**Overconfidence rate**
`% of high-confidence (≥0.85) derivations that were nonetheless overridden`
When the LLM is overconfident and humans disagree, the confidence signal is
poorly calibrated. Used to tune the per-gate thresholds and to feed back
into the Planner Agent for risk-class adjustments.
        """
    )
