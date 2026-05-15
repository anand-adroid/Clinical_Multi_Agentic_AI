"""Visual stepper for the user's six-milestone journey.

Used in two places:
  - On the Run Detail page at the top, full-size, as the primary
    orientation cue.
  - On each row in the Runs list, compact, so the user can scan dozens of
    runs without clicking into each one.

Pure HTML rendered through ``st.markdown(..., unsafe_allow_html=True)`` —
Streamlit's native progress widgets do not support per-step colors or the
``awaiting_you`` semantic this product needs.
"""
from __future__ import annotations

from typing import Any

import streamlit as st


_STATE_COLORS: dict[str, str] = {
    "complete": "#15803D",
    "complete_with_decisions": "#15803D",
    "active": "#1D4ED8",
    "awaiting_you": "#B45309",
    "pending": "#CBD5E1",
    "skipped": "#94A3B8",
    "blocked": "#B91C1C",
}

_STATE_FILLED = {
    "complete", "complete_with_decisions", "active",
    "awaiting_you", "blocked", "skipped",
}


def _marker_html(state: str, size_em: float) -> str:
    color = _STATE_COLORS.get(state, "#CBD5E1")
    border = color if state in _STATE_FILLED else "#CBD5E1"
    bg = color if state in _STATE_FILLED else "transparent"
    diameter = f"{size_em}em"
    return (
        f"<div style='width:{diameter};height:{diameter};border-radius:50%;"
        f"background:{bg};border:2px solid {border};display:inline-block;"
        f"margin:0 auto;'></div>"
    )


def _connector_html(left_state: str, right_state: str) -> str:
    # The line between two markers is green if both are complete, otherwise
    # neutral. Subtle but tells the eye where progress has reached.
    both_done = (
        left_state in ("complete", "complete_with_decisions")
        and right_state in ("complete", "complete_with_decisions", "active", "awaiting_you")
    )
    color = "#15803D" if both_done else "#E2E8F0"
    return (
        f"<div style='flex:1;height:2px;background:{color};margin-top:0.6em;'></div>"
    )


def render_stepper(
    milestones: list[dict[str, Any]],
    *,
    compact: bool = False,
    show_details: bool = True,
) -> None:
    """Render the journey stepper.

    ``compact`` shrinks the markers and hides the detail line; used inside
    list rows where space is tight. ``show_details=False`` hides the small
    secondary text under each label without going fully compact.
    """
    if not milestones:
        return

    size_em = 0.9 if compact else 1.4
    label_size = "0.75em" if compact else "0.85em"
    detail_size = "0.7em" if compact else "0.78em"

    # Build a flex row: marker, connector, marker, connector, ...
    parts: list[str] = [
        "<div style='display:flex;align-items:flex-start;width:100%;"
        "padding:8px 0;'>"
    ]
    for i, m in enumerate(milestones):
        state = m.get("state", "pending")
        color = _STATE_COLORS.get(state, "#94A3B8")
        cell = [
            "<div style='display:flex;flex-direction:column;align-items:center;"
            "flex:0 0 auto;min-width:80px;'>",
            _marker_html(state, size_em),
            f"<div style='font-size:{label_size};color:#0F172A;font-weight:500;"
            f"margin-top:8px;text-align:center;'>{m.get('label','')}</div>",
        ]
        if show_details and not compact:
            detail = m.get("detail", "")
            if detail:
                cell.append(
                    f"<div style='font-size:{detail_size};color:{color};"
                    f"margin-top:2px;text-align:center;max-width:130px;"
                    f"line-height:1.25;'>{detail}</div>"
                )
        cell.append("</div>")
        parts.append("".join(cell))

        if i < len(milestones) - 1:
            next_state = milestones[i + 1].get("state", "pending")
            parts.append(_connector_html(state, next_state))

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def status_badge(label: str, color_name: str) -> str:
    """Return inline HTML for a small status pill. Use with
    ``unsafe_allow_html=True``."""
    palette = {
        "green": ("#15803D", "#DCFCE7"),
        "amber": ("#B45309", "#FEF3C7"),
        "red": ("#B91C1C", "#FEE2E2"),
        "blue": ("#1D4ED8", "#DBEAFE"),
        "gray": ("#475569", "#F1F5F9"),
    }
    fg, bg = palette.get(color_name, palette["gray"])
    return (
        f"<span style='display:inline-block;padding:2px 10px;border-radius:12px;"
        f"background:{bg};color:{fg};font-size:0.8em;font-weight:600;'>"
        f"{label}</span>"
    )
