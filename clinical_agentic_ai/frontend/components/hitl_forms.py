"""Reusable HITL form rendering.

The Reviews page and the Run Workspace (Run detail) page both need to show
the same active-gate form whenever an orchestrator pauses. This module owns
the form code so both pages can call ``render_active_hitl_form`` and stay in
sync. Without this extraction the user has to switch pages to act on a
decision — exactly the multi-page friction we want to eliminate.

The render function returns True if it drew an active form. Callers use that
return value to decide whether to also show idle / completion content.
"""
from __future__ import annotations

import time
from typing import Callable

import pandas as pd
import streamlit as st

from components import api_get, api_post
from components.auth import is_admin


__all__ = ["render_active_hitl_form"]


def _make_submitter(run_id: str) -> Callable[[dict], None]:
    """Build a submit callback bound to the given run. Submission forces a
    rerun so the surrounding page snaps to its next state immediately."""

    def _submit(payload: dict) -> None:
        with st.spinner("Applying decision and resuming workflow..."):
            status, body = api_post(
                f"/runs/{run_id}/hitl/decision", json=payload, timeout=600
            )
        if status != 200:
            st.error(f"Submission failed (HTTP {status}). {body!r}")
            return
        new_status = (body or {}).get("status", "?")
        st.toast(
            f"Decision applied. Pipeline status: {new_status}",
            icon=":material/check_circle:",
        )
        # Mark that we just submitted a decision for this run, and record
        # the exact moment so the Run Workspace can enforce a minimum
        # suppression window (5 s) regardless of how quickly the backend
        # responds. Without the timestamp, a fast backend can clear the
        # pending flag on the very first rerender and let the old form
        # flicker back before the status has actually transitioned.
        st.session_state[f"_hitl_submitted_{run_id}"] = True
        st.session_state[f"_hitl_submitted_at_{run_id}"] = time.monotonic()
        st.rerun()

    return _submit


def render_active_hitl_form(
    run_id: str,
    *,
    reviewer_default: str = "anonymous",
    key_prefix: str = "hitl",
) -> bool:
    """Render the appropriate HITL form for the run's active pause, if any.

    Returns True if a form was drawn (i.e. the run is currently paused on a
    gate the reviewer can act on), False if the run has no pending decision.
    """
    pending = api_get(f"/runs/{run_id}/hitl/pending", default={}) or {}
    if not pending.get("pending"):
        return False

    reason = pending.get("reason", "")
    ctx = pending.get("context") or {}
    target = pending.get("target")
    submit = _make_submitter(run_id)

    # ------------------------------------------------------------------
    # Gate 1 — spec clarifications
    # ------------------------------------------------------------------
    if reason == "spec_clarifications_required":
        clarifications = ctx.get("clarifications") or []
        prefilled = ctx.get("prefilled_answers") or {}

        st.markdown(
            f"**Spec Reviewer flagged {len(clarifications)} ambiguity(ies).** "
            "Write your answer in plain English — describe both the values "
            "the column can take and the rule that assigns each value."
        )
        with st.expander("Example: how to phrase an answer", expanded=False):
            st.markdown(
                "For a column called `COHORT_LABEL` sourced from `age` and "
                "`sex`, a good answer is a clear list of cases:\n\n"
                "> The column may take six values: `YOUNG_M` when age < 30 "
                "and sex is male; `YOUNG_F` when age < 30 and sex is "
                "female; ...etc. Return null if either age or sex is "
                "missing."
            )

        answers: dict[str, str] = {}
        for idx, c in enumerate(clarifications):
            name = c.get("name", "?")
            issue = c.get("issue", "")
            question = c.get("suggested_question", "")
            pf = prefilled.get(name) or {}
            default = pf.get("answer", "")
            used_n = pf.get("times_used", 0)
            with st.container(border=True):
                head_l, head_r = st.columns([3, 1])
                head_l.markdown(
                    f"**{name}** &nbsp; "
                    f"<span style='color:#64748B;font-size:0.85em;'>{issue}</span>",
                    unsafe_allow_html=True,
                )
                if pf:
                    head_r.markdown(
                        f"<div style='text-align:right;color:#15803D;"
                        f"font-size:0.85em;'>Pre-filled from memory "
                        f"(used {used_n}x)</div>",
                        unsafe_allow_html=True,
                    )
                if question:
                    st.markdown(f"_Question:_ {question}")
                answer = st.text_area(
                    "Your answer",
                    value=default,
                    key=f"{key_prefix}_ans_{idx}_{name}",
                    height=120,
                    label_visibility="collapsed",
                    placeholder=(
                        "Describe the values and the rule in plain English."
                    ),
                )
                if answer.strip():
                    answers[name] = answer.strip()

        reviewer = st.text_input(
            "Reviewer ID", value=reviewer_default,
            key=f"{key_prefix}_g1_reviewer",
        )
        comment = st.text_area(
            "Additional comment (optional)", value="", height=60,
            key=f"{key_prefix}_g1_comment",
        )
        btn_l, _, btn_r = st.columns([1, 1, 1])
        if btn_l.button(
            "Submit answers and resume",
            type="primary",
            use_container_width=True,
            key=f"{key_prefix}_g1_submit",
        ):
            submit({
                "reviewer": reviewer,
                "action": "approve",
                "target": None,
                "comment": comment or None,
                "clarification_answers": answers,
            })
        if btn_r.button(
            "Reject run",
            use_container_width=True,
            key=f"{key_prefix}_g1_reject",
        ):
            submit({
                "reviewer": reviewer,
                "action": "reject",
                "target": None,
                "comment": comment or None,
            })
        return True

    # ------------------------------------------------------------------
    # Gate 2 — code preapproval / codegen failure
    # ------------------------------------------------------------------
    if reason in ("code_preapproval_required", "codegen_failed"):
        targets = ctx.get("preapproval_targets") or []
        low_conf = set(ctx.get("low_confidence_targets") or [])
        is_failure_gate = reason == "codegen_failed"
        reviewer_can_edit_code = is_admin()

        if is_failure_gate:
            if reviewer_can_edit_code:
                st.error(
                    ctx.get("message")
                    or "Code generation failed. Provide code manually below."
                )
            else:
                st.error(
                    "The AI could not generate code automatically and your "
                    "role does not include code editing. Ask a statistical "
                    "programmer to complete this run."
                )
                return True
        else:
            forced = ctx.get("policy_forced_targets") or []
            if forced and not low_conf:
                st.info(
                    f"{len(forced)} regulatory-critical derivation(s) "
                    "require your sign-off before execution."
                )
            elif low_conf and not forced:
                st.info(
                    f"{len(low_conf)} derivation(s) reported low LLM "
                    "confidence. Review each before execution."
                )
            else:
                st.info(
                    f"{len(targets)} derivation(s) need your review: "
                    f"{len(forced)} regulatory-critical and "
                    f"{len(low_conf)} low-confidence."
                )

        overrides: dict[str, dict[str, object]] = {}
        for t in targets:
            tgt = t.get("target", "?")
            conf = t.get("confidence")
            notes = t.get("uncertainty_notes")
            trigger = t.get("trigger", "")
            preview = t.get("preview") or []
            with st.container(border=True):
                head_l, head_r = st.columns([3, 1])
                head_l.markdown(f"**{tgt}**")
                badge_bits = []
                if conf is not None:
                    color = (
                        "#15803D" if conf >= 0.85
                        else ("#B45309" if conf >= 0.7 else "#B91C1C")
                    )
                    badge_bits.append(
                        f"<span style='color:{color};font-weight:600;'>"
                        f"conf {conf:.2f}</span>"
                    )
                if trigger == "low_confidence":
                    badge_bits.append(
                        "<span style='color:#B91C1C;'>low-confidence</span>"
                    )
                elif trigger == "policy":
                    badge_bits.append(
                        "<span style='color:#475569;'>policy review</span>"
                    )
                head_r.markdown(
                    "<div style='text-align:right;'>"
                    + " &middot; ".join(badge_bits)
                    + "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"_Rule:_ {t.get('rule', '')}")
                risk_class = t.get("risk_class")
                if risk_class and risk_class != "routine":
                    risk_color = (
                        "#B91C1C" if "critical" in risk_class else "#64748B"
                    )
                    st.markdown(
                        f"<span style='color:{risk_color};font-weight:600;"
                        f"font-size:0.85em;'>risk class: {risk_class}</span>",
                        unsafe_allow_html=True,
                    )
                reasoning = t.get("reasoning")
                if reasoning:
                    st.markdown(
                        "<div style='background:#F1F5F9;padding:8px 10px;"
                        "border-radius:6px;border-left:3px solid #475569;"
                        "margin:6px 0;font-size:0.9em;'>"
                        f"<b>LLM reasoning:</b> {reasoning}</div>",
                        unsafe_allow_html=True,
                    )
                if notes:
                    st.markdown(
                        f"<div style='color:#B45309;'>LLM uncertainty: "
                        f"{notes}</div>",
                        unsafe_allow_html=True,
                    )
                current_code = (
                    t.get("code") or "def derive(row):\n    return None\n"
                )
                if reviewer_can_edit_code:
                    action_options = [
                        "approve", "request changes", "edit code",
                    ]
                    default_idx = 2 if is_failure_gate else 0
                    action_help = (
                        "Approve: accept this code unchanged. "
                        "Request changes: tell the AI what to fix in plain "
                        "English. Edit code: rewrite the Python directly."
                    )
                else:
                    action_options = ["approve", "request changes"]
                    default_idx = 0
                    action_help = (
                        "Approve: accept this code unchanged. "
                        "Request changes: describe in plain English what "
                        "should be different — the AI rewrites the code."
                    )
                sub_action = st.radio(
                    "Decision for this derivation",
                    action_options, horizontal=True,
                    key=f"{key_prefix}_sub_action_{tgt}",
                    index=default_idx, help=action_help,
                )
                edited_code = current_code
                hint = ""
                if sub_action == "edit code":
                    edited_code = st.text_area(
                        "Edited code",
                        value=current_code, height=220,
                        key=f"{key_prefix}_edit_code_{tgt}",
                    )
                elif sub_action == "request changes":
                    hint = st.text_area(
                        "What should be different? (plain English)",
                        key=f"{key_prefix}_hint_{tgt}",
                        placeholder=(
                            "e.g. The cutoff for HIGH should be lab_value "
                            ">= 7.0, not strictly greater than 7.0."
                        ),
                        height=80,
                    )
                else:
                    if not is_failure_gate and reviewer_can_edit_code:
                        with st.expander(
                            "Show generated code", expanded=False
                        ):
                            st.code(current_code, language="python")
                if preview and not is_failure_gate:
                    with st.expander(
                        "Dry-run preview on first dataset rows",
                        expanded=False,
                    ):
                        rows = []
                        for p in preview:
                            rows.append({
                                "input": p.get("input"),
                                "output": p.get("output"),
                                "error": p.get("error") or "",
                            })
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True, hide_index=True,
                        )
                wire_action = {
                    "approve": "approve", "edit code": "edit",
                    "request changes": "regenerate",
                }.get(sub_action, "approve")
                overrides[tgt] = {"action": wire_action}
                if wire_action == "edit":
                    overrides[tgt]["code"] = edited_code
                elif wire_action == "regenerate" and hint:
                    overrides[tgt]["hint"] = hint

        reviewer = st.text_input(
            "Reviewer ID", value=reviewer_default,
            key=f"{key_prefix}_g2_reviewer",
        )
        comment = st.text_area(
            "Additional comment (optional)", value="", height=60,
            key=f"{key_prefix}_g2_comment",
        )
        decision_action = "regenerate" if any(
            v.get("action") == "regenerate" for v in overrides.values()
        ) else "approve"
        btn_l, _, btn_r = st.columns([1, 1, 1])
        if btn_l.button(
            "Submit and resume", type="primary",
            use_container_width=True,
            key=f"{key_prefix}_g2_submit",
        ):
            submit({
                "reviewer": reviewer,
                "action": decision_action,
                "target": None,
                "comment": comment or None,
                "derivation_overrides": overrides,
            })
        if btn_r.button(
            "Reject run",
            use_container_width=True,
            key=f"{key_prefix}_g2_reject",
        ):
            submit({
                "reviewer": reviewer,
                "action": "reject",
                "target": None,
                "comment": comment or None,
            })
        return True

    # ------------------------------------------------------------------
    # Gate 3 — refinement exhausted
    # ------------------------------------------------------------------
    st.warning(
        "The AI's auto-refiner could not fix this derivation after 3 "
        "attempts. Your input is needed."
    )
    if target:
        st.markdown(f"**Target column**  \n`{target}`")
    if "current_code" in ctx and is_admin():
        with st.expander("Code that failed verification", expanded=False):
            st.code(ctx["current_code"], language="python")
    if "recent_findings" in ctx and ctx["recent_findings"]:
        st.markdown("**Why verification failed**")
        st.dataframe(
            pd.DataFrame(ctx["recent_findings"]),
            use_container_width=True, hide_index=True,
        )

    form_l, form_r = st.columns(2)
    with form_l:
        reviewer = st.text_input(
            "Reviewer ID", value=reviewer_default,
            key=f"{key_prefix}_g3_reviewer",
        )
    with form_r:
        if is_admin():
            action_choice = st.radio(
                "Action",
                ["approve", "edit code", "reject"],
                horizontal=True,
                key=f"{key_prefix}_g3_action",
            )
        else:
            action_choice = st.radio(
                "Action",
                ["approve", "reject"],
                horizontal=True,
                key=f"{key_prefix}_g3_action",
            )
    comment = st.text_area(
        "Comment (optional)", value="", height=80,
        key=f"{key_prefix}_g3_comment",
    )
    edited_code = None
    wire_action = "edit" if action_choice == "edit code" else action_choice
    if wire_action == "edit":
        edited_code = st.text_area(
            "Edited derive(row) function",
            value=ctx.get("current_code", "def derive(row):\n    return None\n"),
            height=240,
            key=f"{key_prefix}_g3_code",
        )
    if st.button(
        "Submit decision", type="primary",
        key=f"{key_prefix}_g3_submit",
    ):
        submit({
            "reviewer": reviewer,
            "action": wire_action,
            "target": target,
            "comment": comment or None,
            "edited_code": edited_code,
        })
    return True
