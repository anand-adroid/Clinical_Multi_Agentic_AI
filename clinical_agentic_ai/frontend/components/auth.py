"""Identity and role gating for the frontend.

In production the upstream reverse proxy terminates SSO and injects
``X-User-Email`` / ``X-User-Roles`` headers. A FastAPI middleware parses
those into a request-scoped user, and the frontend reads its identity via
a ``/me`` endpoint.

For the prototype I read the same shape from env vars so the entire UI
runs unchanged against either source. Swap ``current_user`` to call
``/me`` in production and the rest of the codebase needs zero edits.
"""
from __future__ import annotations

import os
from functools import lru_cache

import streamlit as st


_DEFAULT_USER_EMAIL = "demo@local"
_DEFAULT_USER_ROLES = "user"


@lru_cache(maxsize=1)
def _identity_from_env() -> dict[str, object]:
    return {
        "email": os.getenv("USER_EMAIL", _DEFAULT_USER_EMAIL),
        "roles": [
            r.strip()
            for r in os.getenv("USER_ROLES", _DEFAULT_USER_ROLES).split(",")
            if r.strip()
        ],
    }


def current_user() -> dict[str, object]:
    """Return the current user. Override via session-state for the demo
    role switcher so the assessment reviewer can flip personas without
    restarting the app."""
    override = st.session_state.get("_user_override")
    if override:
        return override
    return _identity_from_env()


def current_email() -> str:
    return str(current_user().get("email") or _DEFAULT_USER_EMAIL)


def has_role(role: str) -> bool:
    roles = current_user().get("roles") or []
    return role in roles


def is_admin() -> bool:
    return has_role("admin")


def require_admin() -> None:
    """Block rendering of the current page for non-admin users.

    In production the backend enforces the same check server-side; this
    frontend guard exists to hide UI affordances that the user cannot use.
    """
    if is_admin():
        return
    st.error("Admin access required.")
    st.caption(
        "This page is gated by SSO role. In production access is granted by "
        "your identity provider; in development set ``USER_ROLES=admin`` in "
        "your ``.env`` or use the role switcher in the sidebar."
    )
    st.stop()


def role_switcher() -> None:
    """Sidebar control that lets the demo reviewer flip between user / admin
    without restarting. Rendered whenever identity comes from env vars
    (i.e. dev / demo). In production an SSO header injects identity and a
    middleware sets a flag here that hides this control."""
    if os.getenv("_SSO_ACTIVE") == "true":
        return  # production: identity is read-only, sourced from SSO
    with st.sidebar:
        st.caption("Demo identity")
        # Initial value comes from USER_ROLES env var; user can flip without restart.
        initial = "Admin" if is_admin() else "Reviewer"
        choice = st.selectbox(
            "Sign in as",
            options=["Reviewer", "Admin"],
            index=1 if initial == "Admin" else 0,
            key="_role_switcher",
            label_visibility="collapsed",
        )
        st.session_state["_user_override"] = {
            "email": f"{choice.lower()}@demo",
            "roles": ["admin", "user"] if choice == "Admin" else ["user"],
        }
