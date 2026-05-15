"""Streamlit entry point.

Navigation is intentionally narrow at the top — Runs, New run, Inbox,
Run workspace — because that is the daily flow. Power-user and admin
surfaces sit behind an Admin group that is hidden from non-admin users.
Role gating is driven by ``frontend/components/auth.py`` and swaps over
to SSO headers in production with no code changes on this page.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from components import needs_review_runs, status_toast_watcher
from components.auth import is_admin, role_switcher


st.set_page_config(
    page_title="Clinical Agentic AI",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": (
            "Clinical Agentic AI — multi-agent derivation pipeline with full "
            "audit traceability."
        ),
    },
)

# Global compactness pass. Streamlit defaults are tuned for marketing pages;
# a regulated-workflow UI needs information density closer to Linear / Jira.
st.markdown(
    """
    <style>
    [data-testid="stToolbar"] { visibility: hidden; height: 0 !important; }
    header[data-testid="stHeader"] { background: rgba(0,0,0,0); height: 0 !important; }
    .block-container { padding-top: 0.6rem !important;
                       padding-bottom: 2rem !important;
                       max-width: 1400px; }
    h1 { font-size: 1.45rem !important; font-weight: 600 !important;
         margin: 0 0 4px 0 !important; line-height: 1.2 !important; }
    h2 { font-size: 1.15rem !important; font-weight: 600 !important;
         margin: 8px 0 4px 0 !important; }
    h3 { font-size: 1.0rem  !important; font-weight: 600 !important;
         margin: 8px 0 4px 0 !important; }
    p, .stMarkdown, .stCaption, .stText { font-size: 0.88rem; line-height: 1.45; }
    [data-testid="stMetricValue"] { font-size: 1.25rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.35rem !important; }
    [data-testid="stExpander"] summary { font-size: 0.9rem !important; }
    .stButton button { font-size: 0.85rem !important; padding: 0.35rem 0.9rem !important; }
    [data-testid="stTabs"] button { padding-top: 6px !important; padding-bottom: 6px !important; }
    [data-testid="stDataFrame"] { font-size: 0.82rem !important; }
    section[data-testid="stSidebar"] { min-width: 230px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


_ASSETS = Path(__file__).parent / "assets"


def _find_logo() -> Path | None:
    for name in (
        "sanofi-logo.png",
        "sanofi.png",
        "logo.png",
        "sanofi-logo.svg",
        "logo.svg",
    ):
        candidate = _ASSETS / name
        if candidate.exists():
            return candidate
    return None


logo_path = _find_logo()

# Count runs currently waiting on a reviewer so the Inbox nav label can
# carry a badge. Cached for 5 seconds so navigation feels instant —
# without the cache, every page render and every polling fragment kicks
# off an extra HTTP GET against /runs, which makes the whole UI feel
# sluggish.
@st.cache_data(ttl=5)
def _cached_pending_count() -> int:
    return len(needs_review_runs())


_pending_count = _cached_pending_count()
_inbox_title = (
    f"Inbox ({_pending_count})" if _pending_count else "Inbox"
)

# Primary surfaces — everyone sees these.
primary_pages = [
    st.Page("pages/0_Dashboard.py", title="Runs", default=True),
    st.Page("pages/1_Run_Workflow.py", title="New run"),
    st.Page("pages/2_HITL_Review.py", title=_inbox_title),
    st.Page("pages/3_Audit_Trail.py", title="Run workspace"),
]

# Admin / observability — gated by role. In production the same check
# happens server-side on every admin route.
admin_pages = [
    st.Page("pages/7_HITL_Quality.py", title="Quality metrics"),
    st.Page("pages/4_Memory.py", title="Pattern library"),
    st.Page("pages/5_Evaluation.py", title="Evaluation"),
    st.Page("pages/6_Compare_Runs.py", title="Run comparison"),
]

if is_admin():
    pages = {
        "Workspace": primary_pages,
        "Admin": admin_pages,
    }
else:
    pages = {"Workspace": primary_pages}

nav = st.navigation(pages)

# Role switcher sits in the sidebar below the navigation list.
role_switcher()

# ----------------------------------------------------------------- header
# Product header bar at the top of the main content area: Sanofi logo
# on the left, product name + tagline next to it. Lives above every
# page, so brand and identity stay visible without competing with
# Streamlit's auto-rendered sidebar nav.
#
# Streamlit's ``st.image`` was flickering when used via ``st.logo``;
# rendering the image directly inside a column avoids that path entirely.
_h_l, _h_r = st.columns([1, 8], gap="small", vertical_alignment="center")
with _h_l:
    if logo_path:
        st.image(str(logo_path), width=100)
with _h_r:
    st.markdown(
        "<div style='display:flex;flex-direction:column;justify-content:center;"
        "height:100%;'>"
        "<div style='font-size:1.45rem;font-weight:700;color:#0F172A;"
        "line-height:1.1;'>Clinical Agentic AI</div>"
        "<div style='color:#64748B;font-size:0.92rem;margin-top:2px;'>"
        "Derivation, verification, audit"
        "</div></div>",
        unsafe_allow_html=True,
    )
st.markdown(
    "<hr style='margin:8px 0 14px 0;border:none;"
    "border-top:1px solid #E2E8F0;'>",
    unsafe_allow_html=True,
)

# A lightweight toast watcher fires a transient notification when any
# run's status transitions.
status_toast_watcher()
nav.run()
