"""Reviews — the inbox of runs needing human input.

Production-grade flow: this page is the *queue*. Clicking a run opens the
Run Workspace where the actual decision form is rendered inline alongside
the output, lineage, and audit context. That eliminates the multi-page
context switch the previous version forced on the reviewer.

The legacy single-run view is still available for direct deep links — when
arriving via a notification or banner button, the page shows the active
form in place so an immediate action does not require a tab switch.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import (
    api_get,
    needs_review_runs,
    page_header,
    remember_run,
    run_picker,
)
from components.auth import current_email
from components.hitl_forms import render_active_hitl_form
from components.journey import human_status, journey_milestones
from components.stepper import render_stepper, status_badge


page_header(
    "Reviews",
    "Runs paused on a human decision. Pick one to act on it; or open the "
    "full Run Workspace to act in context.",
)


# ---------------------------------------------------------------------------#
# Inbox view — every paused run, newest first, with a one-click resolve     #
# ---------------------------------------------------------------------------#

queue = needs_review_runs()
if queue:
    st.markdown(
        f"<div style='color:#475569;font-size:0.85em;margin:2px 0 8px 0;'>"
        f"{len(queue)} run(s) waiting on a decision."
        "</div>",
        unsafe_allow_html=True,
    )
    for q in queue:
        rid = q.get("id", "")
        label, color = human_status(q.get("status", ""))
        with st.container(border=True):
            cl, cm, cr = st.columns([4, 2, 1.4])
            with cl:
                st.markdown(
                    f"<div style='font-family:monospace;font-weight:600;'>"
                    f"{rid[:10]}</div>"
                    f"<div style='color:#64748B;font-size:0.8em;'>"
                    f"{q.get('notes') or 'submitted run'} &middot; "
                    f"{q.get('updated_at', '')[:19]}</div>",
                    unsafe_allow_html=True,
                )
            with cm:
                st.markdown(status_badge(label, color), unsafe_allow_html=True)
            with cr:
                if st.button(
                    "Open",
                    type="primary",
                    use_container_width=True,
                    key=f"open_{rid}",
                ):
                    remember_run(rid)
                    st.switch_page("pages/3_Audit_Trail.py")
else:
    st.success("No runs are waiting on a decision.")

st.divider()


# ---------------------------------------------------------------------------#
# Inline single-run mode — when a user follows a deep link or expands a row #
# ---------------------------------------------------------------------------#

st.subheader("Resolve a specific run")
st.caption(
    "If you prefer to act here, pick the run below — the same form that "
    "appears in the Run Workspace will render in place."
)

selected = run_picker(label="Run", key="hitl_run_picker")
if not selected:
    st.stop()

remember_run(selected)


@st.fragment(run_every=2)
def _poll_status() -> None:
    info_now = api_get(f"/runs/{selected}")
    if not info_now:
        return
    cur = info_now.get("status", "")
    key = f"_last_seen_status_{selected}"
    prev = st.session_state.get(key)
    st.session_state[key] = cur
    if prev is not None and prev != cur:
        st.rerun()


_poll_status()

info = api_get(f"/runs/{selected}") or {}
state_blob = api_get(f"/runs/{selected}/state", default={}) or {}
history = api_get(f"/runs/{selected}/hitl/history", default=[]) or []

label, color = human_status(info.get("status", ""))
st.markdown(status_badge(label, color), unsafe_allow_html=True)
st.write("")
milestones = journey_milestones(info, state_blob)
render_stepper(milestones, compact=False, show_details=True)
st.write("")

rendered = render_active_hitl_form(
    selected,
    reviewer_default=current_email(),
    key_prefix="reviews",
)

if not rendered:
    current_status = info.get("status", "")
    if current_status == "running":
        with st.container(border=True):
            st.markdown(
                "<div style='display:flex;align-items:center;gap:10px;'>"
                "<div style='font-size:1.4em;animation:spin 1.4s linear infinite;'>○</div>"
                "<div><div style='font-weight:600;font-size:1.0em;'>"
                "Pipeline is processing</div>"
                "<div style='color:#64748B;font-size:0.85em;margin-top:2px;'>"
                "Another decision may appear here automatically; if the run "
                "completes you can open Run Workspace for the output."
                "</div></div></div>"
                "<style>@keyframes spin {0%{transform:rotate(0)}"
                "100%{transform:rotate(360deg)}}</style>",
                unsafe_allow_html=True,
            )
        if st.button(
            "Open in Run Workspace",
            type="primary",
            key="watch_run_detail",
        ):
            st.switch_page("pages/3_Audit_Trail.py")
    elif current_status.startswith("completed"):
        st.success(
            "Run completed. Open the **Run Workspace** to inspect output "
            "and audit trail."
        )
        if st.button(
            "Open Run Workspace",
            type="primary",
            key="goto_completed_detail",
        ):
            st.switch_page("pages/3_Audit_Trail.py")
    elif current_status == "failed":
        st.error("Run failed. Open the Run Workspace for the cause.")
        if st.button(
            "Open Run Workspace",
            key="goto_failed_detail",
        ):
            st.switch_page("pages/3_Audit_Trail.py")
    else:
        st.info("This run is not waiting on a reviewer.")

st.divider()
st.subheader("Decision history (this run)")
if history:
    st.dataframe(
        pd.DataFrame(history),
        use_container_width=True, hide_index=True,
    )
else:
    st.caption("No human decisions recorded yet.")
