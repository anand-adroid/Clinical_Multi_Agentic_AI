"""Shared UI helpers and API client.

Pages import from this module via ``from components import ...``. Streamlit
adds the entrypoint script's directory (``frontend/``) to ``sys.path`` so the
import works without further path manipulation.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API: str = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


# --------------------------------------------------------------------------- #
#  API client                                                                 #
# --------------------------------------------------------------------------- #


def needs_review_runs() -> list[dict[str, Any]]:
    """Return every run currently waiting on human input. Cheap query —
    safe to call from a polling fragment on every page.
    Uses a short timeout so a slow/starting backend doesn't block page load."""
    runs = api_get("/runs", timeout=4.0, default=[]) or []
    return [r for r in runs if str(r.get("status", "")).startswith("awaiting_hitl_")]


def status_toast_watcher() -> None:
    """Fire a ``st.toast`` whenever any run's status changes since the last
    poll. Gives the user a peripheral notification when their pipeline
    advances — needed because polling fragments only redraw their own
    scope; the rest of the page stays silent.

    Per-run last-seen status is kept in ``st.session_state`` so we only
    toast on transitions, not on every poll.
    """
    @st.fragment(run_every=3)
    def _watch() -> None:
        runs = api_get("/runs", default=[]) or []
        cache = st.session_state.setdefault("_toast_last_status", {})
        for r in runs:
            rid = r.get("id")
            new = r.get("status", "")
            if not rid:
                continue
            prev = cache.get(rid)
            cache[rid] = new
            if prev is None or prev == new:
                continue
            # Only toast for transitions the user actually cares about.
            if new.startswith("awaiting_hitl_"):
                from components.journey import human_status
                label, _ = human_status(new)
                st.toast(f"Run {rid[:10]}: {label}", icon=":material/notifications_active:")
            elif new == "completed":
                st.toast(f"Run {rid[:10]}: completed", icon=":material/check_circle:")
            elif new == "completed_with_warnings":
                st.toast(f"Run {rid[:10]}: completed with warnings", icon=":material/warning:")
            elif new == "failed":
                st.toast(f"Run {rid[:10]}: failed", icon=":material/error:")
    _watch()


def global_review_banner() -> None:
    """Render a notification banner across the top of every page when any
    run needs the current reviewer's attention. Industry-standard pattern
    (Linear, GitHub, Slack): a single line, one click to act on it,
    invisible when nothing is pending so the screen stays calm.

    Polls every 5 seconds via ``st.fragment`` so a new pause raised by a
    background worker surfaces wherever the user happens to be.
    """
    @st.fragment(run_every=5)
    def _render() -> None:
        pending = needs_review_runs()
        n = len(pending)
        if n == 0:
            return
        first = pending[0]
        rid = first.get("id", "")[:10]
        label = (
            f"{n} run{'s' if n != 1 else ''} need your review"
            if n != 1
            else f"Run {rid} needs your review"
        )
        bar_l, bar_r = st.columns([5, 1])
        with bar_l:
            st.markdown(
                "<div style='background:#FEF3C7;border-left:4px solid #B45309;"
                "padding:10px 14px;border-radius:4px;margin-bottom:8px;'>"
                f"<span style='font-weight:600;color:#92400E;'>{label}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        with bar_r:
            if st.button(
                "Resolve",
                type="primary",
                use_container_width=True,
                key="_global_banner_resolve",
            ):
                remember_run(first.get("id"))
                # Open the workspace directly: the active HITL form is
                # rendered inline there, so the reviewer can act without an
                # extra page switch.
                st.switch_page("page_modules/3_Audit_Trail.py")

    _render()


def api_get(
    path: str,
    *,
    timeout: float = 60.0,
    default: Any = None,
    **query_params: Any,
) -> Any:
    """GET ``{API}{path}`` with optional query params. Returns ``default`` on
    4xx/5xx instead of raising — UI components are easier to write against a
    function that never throws on the unhappy path."""
    try:
        params = {k: v for k, v in query_params.items() if v is not None}
        resp = httpx.get(f"{API}{path}", params=params or None, timeout=timeout)
        if resp.status_code != 200:
            return default
        return resp.json()
    except httpx.HTTPError:
        return default


def api_post(
    path: str,
    *,
    json: Any = None,
    files: Any = None,
    data: Any = None,
    timeout: float = 600.0,
) -> tuple[int, Any]:
    """POST and return ``(status_code, body_or_none)``."""
    try:
        resp = httpx.post(
            f"{API}{path}", json=json, files=files, data=data, timeout=timeout
        )
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        return resp.status_code, body
    except httpx.HTTPError as exc:
        return 0, str(exc)


# --------------------------------------------------------------------------- #
#  Layout primitives                                                          #
# --------------------------------------------------------------------------- #


def page_header(title: str, description: str | None = None) -> None:
    """Consistent page title + one-line description. Compact sizing keeps
    the first useful element above the fold on a 13-inch laptop."""
    st.markdown(
        f"<div style='font-size:1.35rem;font-weight:600;color:#0F172A;"
        f"margin:0 0 2px 0;line-height:1.2;'>{title}</div>",
        unsafe_allow_html=True,
    )
    if description:
        st.markdown(
            f"<div style='color:#64748B;font-size:0.85rem;margin:0 0 12px 0;"
            f"line-height:1.35;'>{description}</div>",
            unsafe_allow_html=True,
        )


_STATUS_COLORS: dict[str, str] = {
    "created": "#475569",
    "running": "#1D4ED8",
    "awaiting_hitl_spec": "#B45309",
    "awaiting_hitl_refine": "#B45309",
    "completed": "#15803D",
    "completed_with_warnings": "#B45309",
    "failed": "#B91C1C",
}


def status_pill(status: str) -> str:
    """Return an inline HTML pill for a run status. Use with ``unsafe_allow_html=True``."""
    color = _STATUS_COLORS.get(status, "#475569")
    label = status.replace("_", " ")
    return (
        f"<span style='background:{color}1A;color:{color};"
        f"padding:2px 10px;border-radius:12px;font-size:0.85em;"
        f"font-weight:500;text-transform:capitalize;'>{label}</span>"
    )


def verdict_label(passed: bool) -> str:
    """Return ``Pass`` or ``Fail`` as inline HTML."""
    color = "#15803D" if passed else "#B91C1C"
    text = "Pass" if passed else "Fail"
    return (
        f"<span style='color:{color};font-weight:600;'>{text}</span>"
    )


# Order must match ``PHASE_ORDER`` in backend/core/orchestrator.py.
PHASES: list[tuple[str, str]] = [
    ("input_guard", "Input guard"),
    ("spec_review", "Spec review"),
    ("dag_build", "DAG build"),
    ("plan", "Plan"),
    ("code_generate", "Code gen"),
    ("code_preapproval", "Preapprove"),
    ("static_validate", "Static check"),
    ("test_cases", "Test cases"),
    ("execute", "Execute"),
    ("verify", "Verify"),
    ("refine", "Refine"),
    ("audit", "Audit"),
]


_PHASE_STEP_NAMES: dict[str, set[str]] = {
    "input_guard": {"input_guard"},
    "spec_review": {"review"},
    "dag_build": {"build"},
    "plan": {"plan"},
    "code_generate": {"generate"},
    "code_preapproval": {"preapproval"},
    "static_validate": {"static_check"},
    "test_cases": {"test_cases"},
    "execute": {"execute"},
    "verify": {"verify"},
    "refine": {"refine"},
    "audit": {"audit"},
}


def _phase_done(phase: str, events: list[dict[str, Any]]) -> bool:
    targets = _PHASE_STEP_NAMES.get(phase, {phase})
    for e in events:
        step = e.get("step", "")
        status = e.get("status", "")
        if step in targets and status in ("ok", "warn"):
            return True
    return False


def pipeline_indicator(events: list[dict[str, Any]], status: str) -> None:
    """Render a horizontal progress strip of the 9 pipeline phases."""
    done_states = {p: _phase_done(p, events) for p, _ in PHASES}
    # Current phase = first not-done one (or none if all done).
    current = next((p for p, _ in PHASES if not done_states[p]), None)
    if status in ("completed", "completed_with_warnings", "failed"):
        current = None

    cells = []
    for phase_id, phase_label in PHASES:
        if done_states[phase_id]:
            color, bg, weight = "#15803D", "#DCFCE7", "500"
            marker = "&#9679;"  # filled dot
        elif phase_id == current and status.startswith("awaiting_hitl"):
            color, bg, weight = "#B45309", "#FEF3C7", "600"
            marker = "&#9711;"  # ring (paused)
        elif phase_id == current and status == "running":
            color, bg, weight = "#1D4ED8", "#DBEAFE", "600"
            marker = "&#9679;"
        else:
            color, bg, weight = "#94A3B8", "#F1F5F9", "400"
            marker = "&#9675;"  # empty dot
        cells.append(
            f"<div style='flex:1;background:{bg};color:{color};"
            f"padding:8px 6px;border-radius:6px;text-align:center;"
            f"font-size:0.82em;font-weight:{weight};margin-right:4px;"
            f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
            f"{marker} {phase_label}</div>"
        )
    st.markdown(
        "<div style='display:flex;margin:8px 0 24px 0;'>"
        + "".join(cells)
        + "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Session helpers                                                            #
# --------------------------------------------------------------------------- #


def remember_run(run_id: str) -> None:
    st.session_state["last_run_id"] = run_id


def current_run(default_to_first: bool = True) -> str | None:
    rid = st.session_state.get("last_run_id")
    if rid:
        return rid
    if not default_to_first:
        return None
    runs = api_get("/runs", timeout=4.0, default=[]) or []
    return runs[0]["id"] if runs else None


def run_picker(*, label: str = "Run", key: str = "run_picker") -> str | None:
    """Render a run-selector pre-populated with the latest run."""
    runs = api_get("/runs", timeout=4.0, default=[]) or []
    if not runs:
        st.info("No runs yet. Start one from the **New Run** page.")
        return None
    ids = [r["id"] for r in runs]
    preferred = st.session_state.get("last_run_id")
    index = ids.index(preferred) if preferred in ids else 0
    selected = st.selectbox(label, options=ids, index=index, key=key)
    remember_run(selected)
    return selected


@st.cache_data(ttl=5)
def health_summary() -> dict[str, Any] | None:
    """Get backend health status. Cached for 5 seconds to avoid repeated
    calls on every page load. Uses a longer timeout since the backend may
    be slow on startup."""
    return api_get("/health", timeout=4, default=None)
