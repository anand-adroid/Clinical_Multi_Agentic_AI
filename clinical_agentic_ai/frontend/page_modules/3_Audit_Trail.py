"""Run Workspace — the single-page surface for a run.

Production-grade flow: every action a reviewer takes on a run lives on this
page. When the orchestrator pauses for a decision, the form is rendered at
the top — no page switch required. When the run is processing, a live
status panel polls and self-updates. When the run completes, the tabs hold
the output, lineage, decisions, and audit detail.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

import httpx

from components import (
    API,
    api_get,
    api_post,
    page_header,
    remember_run,
    run_picker,
)
from components.auth import current_email, is_admin
from components.hitl_forms import render_active_hitl_form
from components.journey import (
    human_status,
    journey_milestones,
    needs_user_action,
)
from components.stepper import render_stepper, status_badge


# -----------------------------------------------------------------------------#
# Page chrome                                                                 #
# -----------------------------------------------------------------------------#

page_header(
    "Run Workspace",
    "Every action for a single run — decisions, output, lineage, audit.",
)

selected = run_picker(label="Run", key="detail_run_picker")
if not selected:
    st.stop()

remember_run(selected)


@st.fragment(run_every=2)
def _poll_status() -> None:
    """Rerun the page whenever this run's status changes — so the inline
    HITL form, processing spinner, or completion view appears within ~2s
    of the backend transition."""
    info_now = api_get(f"/runs/{selected}", timeout=5.0)
    if not info_now:
        return
    key = f"_run_detail_last_status_{selected}"
    cur = info_now.get("status", "")
    prev = st.session_state.get(key)
    st.session_state[key] = cur
    if prev is not None and prev != cur:
        st.rerun()


_poll_status()

info = api_get(f"/runs/{selected}", timeout=5.0) or {}
state_blob = api_get(f"/runs/{selected}/state", timeout=5.0, default={}) or {}
status = info.get("status", "")
pill_text, pill_color = human_status(status)

label = (info.get("notes") or f"Run {selected[:10]}").replace("Uploaded ", "")
spec = state_blob.get("spec") or {}
spec_name = spec.get("name") or label


# -----------------------------------------------------------------------------#
# Sticky run header — always visible regardless of scroll position           #
# -----------------------------------------------------------------------------#

st.markdown(
    """
    <style>
    /* Sticky run header. Streamlit's toolbar is hidden globally (see
       app.py) so top:0 is safe; without that hide, raise this to 3rem. */
    .run-header {
        position: sticky; top: 0; z-index: 50;
        background: #FFFFFF; border-bottom: 1px solid #E2E8F0;
        padding: 6px 0 6px 0; margin-bottom: 6px;
    }
    .run-header .title  { font-size:1.05em; font-weight:600; color:#0F172A;
                          line-height:1.2; }
    .run-header .meta   { color:#64748B; font-size:0.8em; margin-top:2px;
                          line-height:1.2; }
    </style>
    """,
    unsafe_allow_html=True,
)
hdr = st.container()
with hdr:
    st.markdown('<div class="run-header">', unsafe_allow_html=True)
    head_l, head_r = st.columns([5, 2])
    with head_l:
        st.markdown(
            f"<div>"
            f"<span class='title'>{spec_name}</span><br>"
            f"<span class='meta'>id "
            f"<span style='font-family:monospace;'>{selected[:10]}</span> · "
            f"submitted by {info.get('user_id','anonymous')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with head_r:
        st.markdown(
            f"<div style='text-align:right;'>"
            + status_badge(pill_text, pill_color)
            + "</div>",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

milestones = journey_milestones(info, state_blob)
render_stepper(milestones, compact=True, show_details=False)


# -----------------------------------------------------------------------------#
# Active section — exactly ONE of: pending HITL form, live progress, idle    #
# -----------------------------------------------------------------------------#
#
# Everything below renders into a single ``st.empty()`` slot that is cleared
# every render cycle. That guarantees stale content from a previous state
# (e.g. a G2 form after the gate is resolved) cannot linger on screen — the
# slot is overwritten exactly once per cycle based on the freshest backend
# read.

_active_slot = st.empty()
_active = _active_slot.container()

# Fetch the live pending payload alongside status. The form is gated on
# BOTH ``needs_user_action(status)`` and ``pending.get("pending")`` — either
# missing means "nothing to act on right now."
_pending_now = api_get(f"/runs/{selected}/hitl/pending", timeout=5.0, default={}) or {}
_has_active_form = (
    needs_user_action(status) and bool(_pending_now.get("pending"))
)

# ── Stale-form suppression ───────────────────────────────────────────────────
# After a HITL submit, st.rerun() fires before the backend has had time to
# transition its status. The flag + timestamp written in hitl_forms.py enforce
# a guaranteed 5-second suppression window so the resolved form can never
# flicker back regardless of backend speed. The flag is cleared once the
# status has actually moved away from any awaiting_hitl state.
_submitted_flag = f"_hitl_submitted_{selected}"
_submitted_at_key = f"_hitl_submitted_at_{selected}"
_just_submitted = bool(st.session_state.get(_submitted_flag))

if _just_submitted:
    _submitted_at = st.session_state.get(_submitted_at_key, 0.0)
    _window_elapsed = (time.monotonic() - _submitted_at) > 5.0
    # Clear the flag once status has moved on OR the 5-second window expires.
    if not needs_user_action(status) or _window_elapsed:
        st.session_state.pop(_submitted_flag, None)
        st.session_state.pop(_submitted_at_key, None)
        _just_submitted = False
    else:
        # Still inside the suppression window — hide the form.
        _has_active_form = False


def _running_indicator(blob: dict) -> str:
    """Build the live-progress line shown while status == 'running'.

    Reads ``summary.current_phase`` / ``summary.current_target`` /
    ``summary.derivations_done`` / ``summary.derivations_total`` from
    the workflow state. The backend updates these mid-phase so the
    indicator changes as each derivation flows through codegen, not
    only at phase boundaries.
    """
    summary = (blob or {}).get("summary") or {}
    phase = summary.get("current_phase")
    target = summary.get("current_target")
    done = summary.get("derivations_done")
    total = summary.get("derivations_total")
    if phase == "code_generation" and target:
        counter = f" [{done or 0}/{total}]" if total else ""
        return f"Generating code for <b>{target}</b>{counter}"
    if phase:
        return f"Running: <b>{phase.replace('_', ' ')}</b>"
    return "Pipeline is processing"


with _active:
    if _just_submitted:
        # Decision was just submitted — show a clear "applying" indicator
        # for the full suppression window so the resolved form never flickers
        # back before the backend has transitioned its state.
        st.markdown(
            "<div style='background:#EFF6FF;border-left:4px solid #1D4ED8;"
            "padding:8px 14px;border-radius:4px;margin:8px 0;display:flex;"
            "align-items:center;gap:10px;'>"
            "<div style='font-size:1.3em;animation:spin 1.4s linear infinite;"
            "color:#1D4ED8;'>○</div>"
            "<div><div style='font-weight:600;color:#1E40AF;font-size:0.92em;'>"
            "Decision applied — resuming pipeline</div>"
            "<div style='color:#1E40AF;font-size:0.8em;'>Your decision has "
            "been recorded; the pipeline will continue shortly.</div>"
            "</div></div>"
            "<style>@keyframes spin {0%{transform:rotate(0)}"
            "100%{transform:rotate(360deg)}}</style>",
            unsafe_allow_html=True,
        )
    elif _has_active_form:
        # Active HITL pause — render the form, with a small banner above it.
        st.markdown(
            "<div style='background:#FEF3C7;border-left:4px solid #B45309;"
            "padding:10px 14px;border-radius:4px;margin:8px 0;'>"
            "<div style='font-weight:600;color:#92400E;font-size:0.95em;'>"
            f"Action required — {pill_text}</div>"
            "<div style='color:#92400E;font-size:0.82em;'>Resolve the form "
            "below to resume the pipeline.</div></div>",
            unsafe_allow_html=True,
        )
        render_active_hitl_form(
            selected,
            reviewer_default=current_email(),
            key_prefix="workspace",
        )
    elif needs_user_action(status):
        # Status says paused but the pending payload is empty — the
        # background worker is still spinning up the next phase. Shown
        # for a beat, then the poll fragment flips status to "running".
        st.markdown(
            "<div style='background:#EFF6FF;border-left:4px solid #1D4ED8;"
            "padding:8px 14px;border-radius:4px;margin:8px 0;display:flex;"
            "align-items:center;gap:10px;'>"
            "<div style='font-size:1.3em;animation:spin 1.4s linear infinite;"
            "color:#1D4ED8;'>○</div>"
            "<div><div style='font-weight:600;color:#1E40AF;font-size:0.92em;'>"
            "Resuming the pipeline</div>"
            "<div style='color:#1E40AF;font-size:0.8em;'>Your decision has "
            "been applied; the background worker is starting the next phase."
            "</div></div></div>"
            "<style>@keyframes spin {0%{transform:rotate(0)}"
            "100%{transform:rotate(360deg)}}</style>",
            unsafe_allow_html=True,
        )
    elif status == "running":
        # Live progress line. Auto-refreshes via a tight fragment so the
        # text changes per-derivation as the orchestrator advances.
        @st.fragment(run_every=1.5)
        def _live_progress() -> None:
            fresh = api_get(f"/runs/{selected}/state", timeout=5.0, default={}) or {}
            line = _running_indicator(fresh)
            st.markdown(
                "<div style='background:#EFF6FF;border-left:4px solid #1D4ED8;"
                "padding:8px 14px;border-radius:4px;margin:8px 0;"
                "display:flex;align-items:center;gap:10px;'>"
                "<div style='font-size:1.2em;animation:spin 1.4s linear "
                "infinite;color:#1D4ED8;'>○</div>"
                f"<div style='color:#1E40AF;font-size:0.92em;'>{line}</div>"
                "</div>"
                "<style>@keyframes spin {0%{transform:rotate(0)}"
                "100%{transform:rotate(360deg)}}</style>",
                unsafe_allow_html=True,
            )
        _live_progress()


# -----------------------------------------------------------------------------#
# Tabs — only rendered once the pipeline reaches a terminal state            #
# -----------------------------------------------------------------------------#
# Output, Lineage, Decisions and Advanced are output artefacts — there is
# nothing meaningful to show while the pipeline is still in flight or waiting
# on a human gate. Hiding them keeps the workspace uncluttered during
# processing and makes the completion state feel distinct.

_pipeline_done = status.startswith("completed") or status == "failed"

if not _pipeline_done:
    st.stop()

decision_count = len(state_blob.get("hitl_history") or [])
decisions_label = (
    f"Decisions ({decision_count})" if decision_count else "Decisions"
)
tab_overview, tab_output, tab_decisions, tab_lineage, tab_advanced = st.tabs(
    ["Overview", "Output", decisions_label, "Lineage", "Advanced"]
)


# -------------------------------------------------------------------- Overview
with tab_overview:
    summary = info.get("summary") or {}
    if str(status).startswith("completed"):
        st.markdown(
            "<div style='font-size:0.95em;color:#15803D;font-weight:600;'>"
            "Run completed successfully.</div>",
            unsafe_allow_html=True,
        )
    elif status == "failed":
        crash = (info.get("summary") or {}).get("crash")
        if crash:
            st.error(f"**Run failed.** {crash}")
        else:
            st.error(
                "Run failed. Inspect the Decisions or Advanced tabs for "
                "the cause."
            )

        # Most failures here are transient LLM overload / rate-limit
        # errors. The orchestrator checkpoints after every phase, so we
        # can pick up exactly where it stopped — no need to re-upload
        # the dataset or re-answer earlier HITL gates.
        is_llm_overload = bool(crash) and (
            "Overloaded" in crash
            or "overloaded_error" in crash
            or "InternalServerError" in crash
            or "RateLimit" in crash
            or "529" in crash
            or "429" in crash
        )
        if is_llm_overload:
            st.info(
                "This looks like a transient Anthropic API hiccup "
                "(rate-limit or 'Overloaded'). Resume picks up from the "
                "last checkpoint without re-running prior phases."
            )
        else:
            st.caption(
                "The error above is what caused the orchestrator to abort. "
                "Resume re-runs from the last checkpoint; the prior phases "
                "are not re-executed."
            )

        rb_l, rb_r = st.columns([1, 4])
        with rb_l:
            if st.button(
                "Resume from checkpoint",
                type="primary",
                key="resume_failed",
                use_container_width=True,
            ):
                with st.spinner("Resuming the run..."):
                    rs, _ = api_post(f"/runs/{selected}/resume", timeout=30)
                if rs == 200:
                    st.toast(
                        "Resume queued. Status will update shortly.",
                        icon=":material/play_circle:",
                    )
                    st.rerun()
                else:
                    st.error(
                        f"Resume failed (HTTP {rs}). Check the backend logs."
                    )
    elif not needs_user_action(status) and status != "running":
        st.info("Run is in progress.")

    derivations = state_blob.get("derivations") or {}
    validations = state_blob.get("validations") or []
    tc = [
        v for v in validations
        if (v.get("rule_id") or "").startswith("TEST_CASE_")
    ]
    tc_pass = sum(1 for v in tc if v.get("passed"))
    block_failures = sum(
        1 for v in validations
        if v.get("severity") == "block" and not v.get("passed")
    )
    warnings = sum(
        1 for v in validations
        if v.get("severity") == "warn" and not v.get("passed")
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "Columns derived",
            f"{summary.get('derivations_ok', len(derivations))} / "
            f"{summary.get('derivations_total', len(derivations))}",
        )
    with m2:
        st.metric(
            "Tests passing",
            f"{tc_pass} / {len(tc)}" if tc else "—",
        )
    with m3:
        st.metric("Your decisions", decision_count)

    m4, m5, m6 = st.columns(3)
    with m4:
        st.metric("Blocking findings", block_failures)
    with m5:
        st.metric("Warnings", warnings)
    with m6:
        h = state_blob.get("resolved_spec_hash") or info.get("spec_hash") or ""
        st.metric("Spec fingerprint", h[:10] if h else "—")

    if state_blob.get("output_path"):
        st.caption(
            f"Output saved to `{state_blob['output_path']}`. "
            "Use the **Output** tab to preview / download, or **Lineage** "
            "to inspect per-derivation reasoning."
        )
        if str(status).startswith("completed") and is_admin():
            if st.button(
                "Score vs golden (admin)",
                key="cd_view_eval",
            ):
                st.switch_page("page_modules/5_Evaluation.py")
    out_resp = api_get(f"/runs/{selected}/output", timeout=10.0)
    if not out_resp:
        st.info(
            "No output yet — the run has not reached the execution phase."
        )
    else:
        rows = out_resp.get("rows") or []
        cols = out_resp.get("columns") or []
        st.caption(
            f"{out_resp.get('preview_size','?')} of "
            f"{out_resp.get('row_count','?')} rows shown · "
            f"{len(cols)} columns"
        )
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True, hide_index=True,
            )
        st.markdown("**Download artefacts**")
        dl_l, dl_m, dl_r, dl_x = st.columns(4)

        def _fetch_artefact(kind: str) -> bytes | None:
            try:
                resp = httpx.get(
                    f"{API}/runs/{selected}/download/{kind}",
                    timeout=30,
                )
                if resp.status_code == 200:
                    return resp.content
            except httpx.HTTPError:
                pass
            return None

        with dl_l:
            csv_bytes = _fetch_artefact("csv")
            if csv_bytes:
                st.download_button(
                    "CSV", data=csv_bytes,
                    file_name=f"{selected[:10]}_output.csv",
                    mime="text/csv", use_container_width=True,
                )
        with dl_m:
            pq_bytes = _fetch_artefact("parquet")
            if pq_bytes:
                st.download_button(
                    "Parquet", data=pq_bytes,
                    file_name=f"{selected[:10]}_output.parquet",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        with dl_r:
            audit_json = _fetch_artefact("audit_json")
            if audit_json:
                st.download_button(
                    "audit.json", data=audit_json,
                    file_name=f"{selected[:10]}_audit.json",
                    mime="application/json", use_container_width=True,
                )
        with dl_x:
            audit_md = _fetch_artefact("audit_md")
            if audit_md:
                st.download_button(
                    "audit.md", data=audit_md,
                    file_name=f"{selected[:10]}_audit.md",
                    mime="text/markdown", use_container_width=True,
                )


# ------------------------------------------------------------------- Decisions
with tab_decisions:
    hitl_history = state_blob.get("hitl_history") or []
    if not hitl_history:
        st.caption("No human decisions recorded for this run.")
    else:
        for entry in hitl_history:
            with st.container(border=True):
                left, right = st.columns([3, 2])
                with left:
                    st.markdown(
                        f"<b>{entry.get('reviewer','?')}</b> "
                        f"<span style='color:#64748B;'>·</span> "
                        f"<span style='font-weight:600;'>"
                        f"{entry.get('action','?')}</span>"
                        + (
                            f" <span style='color:#64748B;'>· "
                            f"{entry.get('target')}</span>"
                            if entry.get("target") else ""
                        ),
                        unsafe_allow_html=True,
                    )
                    if entry.get("comment"):
                        st.caption(entry["comment"])
                with right:
                    st.caption(entry.get("at", ""))
                clarifs = entry.get("clarification_answers") or {}
                if clarifs:
                    st.markdown("**Clarification answers**")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"derivation": k, "answer": v}
                                for k, v in clarifs.items()
                            ]
                        ),
                        use_container_width=True, hide_index=True,
                    )
                overrides = entry.get("derivation_overrides") or {}
                if overrides:
                    st.markdown("**Per-derivation overrides**")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "derivation": k,
                                    "action": v.get("action"),
                                    "hint": v.get("hint", ""),
                                }
                                for k, v in overrides.items()
                            ]
                        ),
                        use_container_width=True, hide_index=True,
                    )


# ---------------------------------------------------------------------- Lineage
with tab_lineage:
    report = api_get(f"/runs/{selected}/audit/report", timeout=10.0)
    lineage_items: list[dict] = []
    if report and report.get("lineage"):
        lineage_items = list(report.get("lineage") or [])
        st.caption(
            "Final audited lineage — generated by the Auditor agent at the "
            "end of the run."
        )
    else:
        derivs = state_blob.get("derivations") or {}
        all_validations = state_blob.get("validations") or []
        for tgt, d in derivs.items():
            lineage_items.append({
                "target": tgt,
                "sources": d.get("sources") or [],
                "status": d.get("status"),
                "code": d.get("code", ""),
                "code_hash": d.get("code_hash", ""),
                "generator": d.get("generator"),
                "attempts": d.get("attempt", 1),
                "confidence": d.get("confidence"),
                "uncertainty_notes": d.get("uncertainty_notes"),
                "reasoning": d.get("reasoning"),
                "risk_class": d.get("risk_class", "routine"),
                "null_count": d.get("null_count", 0),
                "row_errors_sample": (d.get("row_errors") or [])[:5],
                "validations": [
                    v for v in all_validations if v.get("target") == tgt
                ],
                "hitl_decisions": [],
            })
        if lineage_items:
            st.caption(
                "Live lineage — code generation has completed but the final "
                "audit report has not yet been written."
            )

    if not lineage_items:
        st.info(
            "Lineage will appear here once the Code Generation phase "
            "completes."
        )
    else:
        for item in lineage_items:
            with st.expander(
                f"{item.get('target')}  ·  {item.get('status','')}",
                expanded=False,
            ):
                tab_summary, tab_code = st.tabs(["Summary", "Generated code"])
                with tab_summary:
                    cA, cB, cC = st.columns(3)
                    cA.markdown(
                        f"**Depends on**  \n"
                        f"{', '.join(item.get('sources') or []) or '—'}"
                    )
                    cB.markdown(
                        f"**Risk class**  \n"
                        f"{item.get('risk_class', 'routine')}"
                    )
                    conf = item.get("confidence")
                    conf_str = f"{conf:.2f}" if conf is not None else "—"
                    conf_color = (
                        "#15803D" if (conf or 0) >= 0.85 else
                        ("#B45309" if (conf or 0) >= 0.7 else "#B91C1C")
                    ) if conf is not None else "#64748B"
                    cC.markdown(
                        f"**Confidence**  \n"
                        f"<span style='color:{conf_color};"
                        f"font-weight:600;'>{conf_str}</span>",
                        unsafe_allow_html=True,
                    )
                    if item.get("reasoning"):
                        st.markdown(
                            "<div style='background:#F1F5F9;"
                            "padding:8px 10px;border-radius:6px;"
                            "border-left:3px solid #475569;margin:6px 0;'>"
                            "<div style='font-size:0.78em;color:#475569;"
                            "font-weight:600;'>"
                            "Generation reasoning</div>"
                            f"<div style='color:#0F172A;font-size:0.88em;'>"
                            f"{item['reasoning']}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    if item.get("uncertainty_notes"):
                        st.markdown(
                            "<div style='background:#FEF3C7;"
                            "padding:8px 10px;border-radius:6px;"
                            "border-left:3px solid #B45309;margin:6px 0;'>"
                            "<div style='font-size:0.78em;color:#B45309;"
                            "font-weight:600;'>Uncertainty noted</div>"
                            f"<div style='color:#0F172A;font-size:0.88em;'>"
                            f"{item['uncertainty_notes']}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    vals = item.get("validations") or []
                    test_cases = [
                        v for v in vals
                        if v.get("rule_id", "").startswith("TEST_CASE_")
                    ]
                    other_vals = [
                        v for v in vals
                        if not v.get("rule_id", "").startswith("TEST_CASE_")
                    ]
                    if test_cases:
                        passed = sum(
                            1 for t in test_cases if t.get("passed")
                        )
                        total = len(test_cases)
                        color = "#15803D" if passed == total else "#B91C1C"
                        st.markdown(
                            f"**Test cases** &nbsp; "
                            f"<span style='color:{color};font-weight:600;'>"
                            f"{passed}/{total} passing</span>",
                            unsafe_allow_html=True,
                        )
                        rows = []
                        for t in test_cases:
                            d = t.get("detail") or {}
                            rows.append({
                                "case": t["rule_id"].replace(
                                    "TEST_CASE_", "#"
                                ),
                                "result": "Pass" if t.get("passed") else "Fail",
                                "input": d.get("input"),
                                "expected": d.get("expected"),
                                "actual": d.get("actual"),
                                "error": d.get("error") or "",
                            })
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True, hide_index=True,
                        )
                    if other_vals:
                        st.markdown("**Validations**")
                        vdf = pd.DataFrame(other_vals)
                        if "passed" in vdf.columns:
                            vdf["result"] = vdf["passed"].map(
                                lambda x: "Pass" if x else "Fail"
                            )
                            keep = [
                                c for c in
                                ["rule_id", "severity", "result", "message"]
                                if c in vdf.columns
                            ]
                            st.dataframe(
                                vdf[keep], use_container_width=True,
                                hide_index=True,
                            )
                with tab_code:
                    st.caption(
                        "Sandboxed Python function executed once per row."
                    )
                    st.code(item.get("code", ""), language="python")
                    code_hash = item.get("code_hash", "")
                    if code_hash:
                        st.caption(f"Code fingerprint: `{code_hash[:16]}`")


# --------------------------------------------------------------------- Advanced
with tab_advanced:
    st.caption(
        "Engineering detail: dependency graph, agent timeline, hash chain."
    )
    if is_admin():
        dot = api_get(f"/runs/{selected}/dag.dot", timeout=10.0)
        if dot and dot.get("dot"):
            with st.expander("Dependency graph (admin)", expanded=False):
                st.graphviz_chart(dot["dot"], use_container_width=True)
    with st.expander("Hash chain", expanded=False):
        rows = [
            {"label": "Spec (uploaded)", "hash": info.get("spec_hash", "")},
            {
                "label": "Spec (after HITL)",
                "hash": state_blob.get("resolved_spec_hash") or "—",
            },
            {"label": "Dataset", "hash": info.get("dataset_hash", "")},
        ]
        plan = state_blob.get("plan") or {}
        if plan:
            rows.append({
                "label": "Run plan",
                "hash": plan.get("plan_hash") or "—",
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True, hide_index=True,
        )
    events = api_get(f"/runs/{selected}/events", timeout=10.0, default=[]) or []
    with st.expander("Agent event timeline", expanded=False):
        if events:
            ev = pd.DataFrame(events)
            keep = [
                c for c in
                ["agent", "step", "status", "duration_ms", "created_at"]
                if c in ev.columns
            ]
            st.dataframe(
                ev[keep], use_container_width=True, hide_index=True,
                height=320,
            )
    audit_rows = api_get(f"/runs/{selected}/audit", timeout=10.0, default=[]) or []
    with st.expander("Audit ledger", expanded=False):
        if audit_rows:
            st.dataframe(
                pd.DataFrame(audit_rows),
                use_container_width=True, hide_index=True, height=320,
            )
