"""Runs — the primary landing page.

A list of every run the user can see, presented as cards rather than a
raw table. Each card surfaces the spec name (not the run ID), the
human-readable status, and the journey stepper so the user can see at a
glance which step a run is on and whether it needs their attention. The
primary CTA — Submit a new run — sits in the top-right corner, where
people look first.

This page replaces the previous Dashboard, which mixed system health,
stale-run cleanup, and a generic recent-runs table.
"""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from components import (
    api_get,
    api_post,
    health_summary,
    page_header,
    remember_run,
)
from components.auth import current_email, is_admin
from components.journey import (
    TOTAL_PHASES,
    current_phase,
    human_status,
    journey_milestones,
    needs_user_action,
)
from components.stepper import render_stepper, status_badge


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#

def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _relative_time(ts: str | None) -> str:
    dt = _parse_ts(ts)
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60} min ago"
    if secs < 86400:
        return f"{secs // 3600} hr ago"
    return f"{secs // 86400} days ago"


def _spec_label(run: dict) -> str:
    """Display label for a run. Prefer the spec's ``name`` field; fall back
    to a short version of the run ID."""
    # The summary may carry the spec name if the orchestrator surfaced it;
    # otherwise we use the user_id + short id as a sensible default.
    summary = run.get("summary") or {}
    if isinstance(summary, dict) and summary.get("spec_name"):
        return str(summary["spec_name"])
    notes = run.get("notes") or ""
    if notes and "Uploaded" not in notes:
        return notes
    return f"Run {(run.get('id') or '')[:10]}"


def _is_resumable(run: dict) -> bool:
    if run.get("status") != "running":
        return False
    dt = _parse_ts(run.get("updated_at") or run.get("created_at"))
    if not dt:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() > 30


# -----------------------------------------------------------------------------#
# Page
# -----------------------------------------------------------------------------#

page_header(
    "Runs",
    "Pipeline submissions, in reverse chronological order.",
)

# Backend health is diagnostic noise when everything is fine; only surface
# the banner when there is a real problem. The /health endpoint returns
# {"status":"ok","version":"...","llm_enabled":bool} on success or None
# when unreachable.
# Use a fragment for health check so a slow/unavailable backend doesn't
# block the page from showing the runs list and CTA.
@st.fragment(run_every=10)
def _show_health_banner() -> None:
    health = health_summary()
    if not health or health.get("status") != "ok":
        st.error(
            "Backend unreachable at the configured API URL. "
            "Confirm the backend is running on the expected port "
            "(default `http://localhost:8000`), or set `API_BASE_URL` in `.env`."
        )
    elif not health.get("llm_enabled"):
        st.warning(
            "LLM is not configured. Runs will pause at code generation and "
            "require manual code entry through the Reviews page. Add "
            "`ANTHROPIC_API_KEY` to `.env` to enable AI-driven code generation."
        )

_show_health_banner()

# Primary CTA in the top-right corner where the eye lands first.
strip_l, strip_r = st.columns([5, 1])
with strip_r:
    if st.button("New run", type="primary", use_container_width=True):
        st.switch_page("page_modules/1_Run_Workflow.py")

# Use short 5-second timeout so page doesn't hang if backend is slow.
runs = api_get("/runs", timeout=5.0, default=[]) or []
my_email = current_email()


# -----------------------------------------------------------------------------#
# Empty state — first-time user gets a hero CTA, not a thin info banner.
# -----------------------------------------------------------------------------#

if not runs:
    st.write("")
    st.write("")
    hero_l, hero_c, hero_r = st.columns([1, 3, 1])
    with hero_c:
        st.markdown(
            "<div style='text-align:center;padding:32px 8px;'>"
            "<div style='font-size:1.4em;font-weight:600;color:#0F172A;'>"
            "Welcome to Clinical Agentic AI"
            "</div>"
            "<div style='color:#475569;margin-top:12px;font-size:0.95em;"
            "line-height:1.5;max-width:520px;margin-left:auto;margin-right:auto;'>"
            "Upload a clinical dataset together with a derivation "
            "specification. The pipeline reviews the spec, generates "
            "transformation code, validates it against your test cases, "
            "and produces an analysis-ready table with a full audit trail."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        bL, bM, bR = st.columns([1, 2, 1])
        with bM:
            if st.button(
                "Submit your first run",
                type="primary",
                use_container_width=True,
                key="hero_new_run",
            ):
                st.switch_page("page_modules/1_Run_Workflow.py")
        st.markdown(
            "<div style='text-align:center;margin-top:20px;color:#64748B;"
            "font-size:0.85em;'>or pick a sample dataset from the "
            "<b>New run</b> page</div>",
            unsafe_allow_html=True,
        )
    st.stop()


# -----------------------------------------------------------------------------#
# Filters + counts
# -----------------------------------------------------------------------------#

awaiting_count = sum(1 for r in runs if needs_user_action(r.get("status")))
running_count = sum(1 for r in runs if r.get("status") == "running")
completed_count = sum(1 for r in runs if str(r.get("status", "")).startswith("completed"))

f1, f2, f3, f4 = st.columns(4)
with f1:
    st.metric("Total runs", len(runs))
with f2:
    st.metric("Awaiting review", awaiting_count)
with f3:
    st.metric("Running", running_count)
with f4:
    st.metric("Completed", completed_count)

st.write("")

filter_col, _ = st.columns([2, 6])
with filter_col:
    filter_choice = st.selectbox(
        "Show",
        options=["All", "Needing my action", "Running", "Completed", "Failed"],
        index=0,
        label_visibility="collapsed",
    )


def _matches_filter(run: dict) -> bool:
    s = run.get("status", "")
    if filter_choice == "All":
        return True
    if filter_choice == "Needing my action":
        return needs_user_action(s)
    if filter_choice == "Running":
        return s == "running"
    if filter_choice == "Completed":
        return str(s).startswith("completed")
    if filter_choice == "Failed":
        return s == "failed"
    return True


visible = [r for r in runs if _matches_filter(r)]

if not visible:
    st.write("")
    st.caption(
        f"No runs match the filter '{filter_choice}'. "
        "Switch the filter back to 'All' to see every run."
    )
    st.stop()


# -----------------------------------------------------------------------------#
# Run cards
#
# Wrapped in a polling fragment so cards refresh every 3 seconds while at
# least one run is "running" or awaiting input. The cards animate live —
# the stepper advances, the status pill updates, the phase indicator
# shifts — without the user having to refresh.
# -----------------------------------------------------------------------------#

# Tracks whether any visible run is in an active state; used to control
# refresh cadence. If everything is terminal there's no need to poll.
_active_states = ("running",) + tuple(
    s for s in (
        "awaiting_hitl_spec", "awaiting_hitl_preapproval",
        "awaiting_hitl_refine", "awaiting_hitl_codegen",
    )
)
_has_active = any(r.get("status") in _active_states for r in visible)


@st.fragment(run_every=3 if _has_active else 30)
def _runs_list() -> None:
    """Render the run cards. Refreshes every 3 seconds while any run is
    active so the stepper and phase indicator advance live; drops to
    30-second polling when everything is terminal so the page stays
    responsive but doesn't hammer the backend."""
    # Re-fetch on every tick — keeps the list aligned with the backend.
    # Use short 5-second timeout so polling doesn't block the page.
    fresh = api_get("/runs", timeout=5.0, default=[]) or []
    rendered = [r for r in fresh if _matches_filter(r)]
    for run in rendered:
        rid = run.get("id", "")
        status = run.get("status", "")
        label = _spec_label(run)
        pill_text, pill_color = human_status(status)
        relative = _relative_time(run.get("updated_at") or run.get("created_at"))
        submitter = run.get("user_id") or "anonymous"

        # Use short 5-second timeout for state fetch so individual run cards
        # don't block if backend is slow. Missing state just means stepper
        # won't show detailed phase info, but the run card still renders.
        state_blob = api_get(f"/runs/{rid}/state", timeout=5.0, default={}) or {}

        # Augment the status pill with the current phase when a run is
        # actively running. Reviewers want to know "where in the pipeline"
        # at a glance.
        phase_info = current_phase(state_blob) if status == "running" else None
        if phase_info:
            phase_label, phase_idx = phase_info
            pill_text = f"{pill_text}: {phase_label} ({phase_idx}/{TOTAL_PHASES})"

        with st.container(border=True):
            head_l, head_r = st.columns([5, 2])
            with head_l:
                st.markdown(
                    f"<div style='font-size:1.05em;font-weight:600;color:#0F172A;'>{label}</div>"
                    f"<div style='font-size:0.85em;color:#64748B;margin-top:2px;'>"
                    f"id <span style='font-family:monospace;'>{rid[:10]}</span> · "
                    f"submitted by {submitter} · {relative}</div>",
                    unsafe_allow_html=True,
                )
            with head_r:
                st.markdown(
                    f"<div style='text-align:right;padding-top:4px;'>"
                    + status_badge(pill_text, pill_color)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            milestones = journey_milestones(run, state_blob)
            render_stepper(milestones, compact=False, show_details=True)

            _action_buttons(rid, status, run)


def _action_buttons(rid: str, status: str, run: dict) -> None:
    action_l, action_m, _ = st.columns([1, 1, 1])
    if needs_user_action(status):
        with action_l:
            if st.button("Resolve", key=f"resolve_{rid}", type="primary",
                         use_container_width=True):
                remember_run(rid)
                # Workspace renders the active HITL form inline, so the
                # reviewer can act in the same place they inspect output.
                st.switch_page("page_modules/3_Audit_Trail.py")
        with action_m:
            if st.button("Open", key=f"open_{rid}", use_container_width=True):
                remember_run(rid)
                st.switch_page("page_modules/3_Audit_Trail.py")
    elif _is_resumable(run):
        with action_l:
            if st.button("Resume", key=f"resume_{rid}", type="primary",
                         use_container_width=True):
                with st.spinner("Resuming..."):
                    s, _body = api_post(f"/runs/{rid}/resume", json={})
                if s == 200:
                    st.rerun()
        with action_m:
            if st.button("Open", key=f"open_{rid}", use_container_width=True):
                remember_run(rid)
                st.switch_page("page_modules/3_Audit_Trail.py")
    else:
        with action_l:
            if st.button("Open", key=f"open_{rid}", type="primary",
                         use_container_width=True):
                remember_run(rid)
                st.switch_page("page_modules/3_Audit_Trail.py")


_runs_list()


# -----------------------------------------------------------------------------#
# Admin-only utility: clean up stale runs
# -----------------------------------------------------------------------------#

if is_admin():
    st.write("")
    with st.expander("Admin utilities", expanded=False):
        st.caption(
            "These actions are gated by the admin role. In production they "
            "are also enforced server-side."
        )
        if st.button("Mark stale running runs as failed"):
            s, body = api_post("/runs/cleanup-stale", data=None)
            if s == 200:
                n = (body or {}).get("count", 0)
                st.success(f"Cleared {n} stale run(s).")
                st.rerun()
            else:
                st.error(f"Failed: HTTP {s}")
