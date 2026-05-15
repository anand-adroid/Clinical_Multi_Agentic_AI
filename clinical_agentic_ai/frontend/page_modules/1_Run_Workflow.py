"""New Run — submit a dataset and spec, then monitor execution."""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from components import (
    API,
    api_get,
    api_post,
    page_header,
    pipeline_indicator,
    remember_run,
    status_pill,
)
from components.auth import is_admin


page_header(
    "New Run",
    "Upload a clinical dataset and a derivation spec to start a workflow.",
)


def _submit(files: dict[str, tuple], data: dict[str, str]) -> None:
    with st.spinner("Submitting and executing pipeline..."):
        status, body = api_post(
            "/runs/upload-and-run", files=files, data=data, timeout=600
        )
    if status != 200 or not isinstance(body, dict) or "id" not in body:
        st.error(f"Submission failed (HTTP {status}). {body!r}")
        return
    remember_run(body["id"])
    st.session_state["just_submitted"] = True
    st.rerun()


# --------------------------------------------------------------------------- #
#  Upload + preview                                                           #
# --------------------------------------------------------------------------- #

st.markdown(
    "Two spec formats are supported:\n\n"
    "- **YAML** or **JSON** with the full schema (nested test cases, "
    "invariants, risk classes).\n"
    "- **CSV** with columns `name, sources, type, allowed_values, rule, "
    "risk_class, max_null_rate` and rules written in plain English — "
    "matches how clinical data managers actually author specs in Excel today."
)

c1, c2 = st.columns(2)
with c1:
    st.caption(
        ":material/table_view: **Dataset** — patient-row data the "
        "pipeline derives from. Columns are subjects, dates, lab values, "
        "etc. (e.g. `subject_id, age, ...`)."
    )
    dataset_file = st.file_uploader("Dataset (CSV)", type=["csv"])
with c2:
    st.caption(
        ":material/list_alt: **Spec** — derivation rules. Rows describe "
        "the columns the pipeline should compute (e.g. `AGE_GROUP, "
        "TREATMENT_DURATION, ...`). Required headers for CSV: "
        "`name, sources, type, rule`."
    )
    spec_file = st.file_uploader(
        "Spec (YAML, JSON, or CSV)",
        type=["yaml", "yml", "json", "csv"],
    )

# Preview block — collapsed by default so the upload form stays visible
# above the fold. A short one-line summary appears next to each
# uploader after a file is picked; the user can expand to inspect rows.
if dataset_file is not None or spec_file is not None:
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        if dataset_file is not None:
            try:
                df_preview = pd.read_csv(io.BytesIO(dataset_file.getvalue()))
                st.caption(
                    f":material/check_circle: `{dataset_file.name}` "
                    f"— {len(df_preview):,} rows × {len(df_preview.columns)} cols"
                )
                with st.expander("Preview dataset", expanded=False):
                    st.caption(
                        f"Columns: `{', '.join(df_preview.columns[:12])}"
                        f"{'...' if len(df_preview.columns) > 12 else ''}`"
                    )
                    st.dataframe(
                        df_preview.head(8),
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as exc:
                st.error(f"Could not parse the dataset CSV: {exc}")
    with pcol2:
        if spec_file is not None:
            spec_bytes = spec_file.getvalue()
            spec_ext = Path(spec_file.name).suffix.lower()
            st.caption(
                f":material/check_circle: `{spec_file.name}` "
                f"— {len(spec_bytes):,} bytes"
            )
            with st.expander("Preview spec", expanded=False):
                if spec_ext == ".csv":
                    try:
                        spec_df = pd.read_csv(io.BytesIO(spec_bytes))
                        st.dataframe(
                            spec_df,
                            use_container_width=True,
                            hide_index=True,
                        )
                    except Exception as exc:
                        st.error(f"Could not parse the spec CSV: {exc}")
                else:
                    try:
                        text = spec_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        text = spec_bytes.decode("utf-8", errors="replace")
                    lang = "yaml" if spec_ext in (".yaml", ".yml") else (
                        "json" if spec_ext == ".json" else "text"
                    )
                    st.code(text, language=lang)

st.divider()

c3, c4 = st.columns(2)
with c3:
    run_id_input = st.text_input("Run ID (optional)", value="")
with c4:
    user_id = st.text_input(
        "Your reviewer ID", value=os.environ.get("USER", "anonymous")
    )

if st.button(
    "Submit run", type="primary", disabled=not (dataset_file and spec_file)
):
    data = {"auto_start": "true", "user_id": user_id}
    if run_id_input:
        data["run_id"] = run_id_input
    _submit(
        files={
            "dataset": (dataset_file.name, dataset_file.getvalue(), "text/csv"),
            "spec": (spec_file.name, spec_file.getvalue(), "application/yaml"),
        },
        data=data,
    )


st.divider()


# --------------------------------------------------------------------------- #
#  Latest run                                                                 #
# --------------------------------------------------------------------------- #
# Only show run progress if the user has actually submitted from this page in
# this session. ``just_submitted`` is set exclusively in ``_submit()`` above,
# so navigating here from another page never triggers this section — the user
# sees a clean upload form, not a stale previous run.

last_run_id = st.session_state.get("last_run_id")
if not st.session_state.get("just_submitted") or not last_run_id:
    st.info("Submit a run above to see its progress here.")
    st.stop()


@st.fragment(run_every=2)
def _live_run_panel() -> None:
    """Auto-refreshing block. Polls the backend every two seconds while
    a run is in flight so the user sees progress without manually
    refreshing. Stops re-running once the run reaches a terminal or
    HITL-pause state — there's nothing else to update at that point."""
    info = api_get(f"/runs/{last_run_id}", timeout=5.0)
    if not info:
        st.warning(f"Run `{last_run_id}` not found.")
        return

    state = api_get(f"/runs/{last_run_id}/state", timeout=5.0, default={}) or {}
    events = api_get(f"/runs/{last_run_id}/events", timeout=5.0, default=[]) or []

    st.subheader("Latest run")
    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown(
            f"<div style='font-family:monospace;font-size:1.05em;'>{info['id']}</div>"
            f"<div style='color:#64748B;font-size:0.85em;margin-top:2px;'>"
            f"spec <code>{info['spec_hash'][:12]}</code> &middot; "
            f"dataset <code>{info['dataset_hash'][:12]}</code></div>",
            unsafe_allow_html=True,
        )
    with hdr_r:
        st.markdown(status_pill(info["status"]), unsafe_allow_html=True)

    pipeline_indicator(events, info["status"])

    if str(info["status"]).startswith("awaiting_hitl"):
        st.warning(
            "**The run is paused for a human decision.** "
            "Open the **Run workspace** — the decision form is rendered "
            "inline alongside the run's full context."
        )
        if st.button(
            "Open Run workspace",
            type="primary",
            key="goto_reviews",
        ):
            st.switch_page("page_modules/3_Audit_Trail.py")

    if is_admin() and state.get("topo_order"):
        st.markdown(
            "<div style='color:#64748B;font-size:0.9em;margin-top:8px;'>"
            "Execution order: <span style='color:#0F172A;'>"
            + " &rsaquo; ".join(state["topo_order"])
            + "</span></div>",
            unsafe_allow_html=True,
        )

    if is_admin():
        dot_payload = api_get(f"/runs/{last_run_id}/dag.dot", timeout=5.0, default=None)
        if dot_payload and dot_payload.get("dot"):
            with st.expander("Dependency graph (admin)", expanded=False):
                st.graphviz_chart(dot_payload["dot"], use_container_width=True)

    if is_admin() and events:
        with st.expander("Agent timeline (admin)", expanded=False):
            ev_df = pd.DataFrame(events)
            keep = [c for c in ["agent", "step", "status", "duration_ms", "created_at"] if c in ev_df.columns]
            st.dataframe(ev_df[keep], use_container_width=True, hide_index=True, height=240)

    out = api_get(f"/runs/{last_run_id}/output")
    if out is None:
        if info["status"] == "failed":
            failed_step = events[-1].get("step") if events else None
            if failed_step == "input_guard":
                st.error(
                    "Run failed during input guardrails. "
                    "Open the Run workspace for the failed input validation details."
                )
                input_validations = [
                    v for v in (state.get("validations") or [])
                    if not v.get("passed")
                    and v.get("rule_id", "").startswith(
                        (
                            "MISSING_COLUMNS",
                            "PII_",
                            "INT_COERCION_LOSS",
                            "FLOAT_COERCION_LOSS",
                            "DATE_COERCION_LOSS",
                        )
                    )
                ]
                for v in input_validations:
                    st.markdown(f"- {v.get('message')}")
            else:
                st.error(
                    "Run failed before output was generated. "
                    "Open the Run workspace for the failure details."
                )
        else:
            st.caption(
                "No output table yet. The run may still be in progress or paused for review."
            )
    else:
        st.subheader("Output preview")
        st.caption(
            f"{out.get('preview_size', 0)} of {out.get('row_count', 0)} rows. "
            f"Saved at `{out.get('output_path', '')}`."
        )
        st.dataframe(
            pd.DataFrame(out.get("rows", [])),
            use_container_width=True,
            hide_index=True,
        )
        if is_admin():
            nav_l, nav_r = st.columns(2)
            if nav_l.button(
                "Open Run detail",
                use_container_width=True,
                key="goto_run_detail",
            ):
                st.switch_page("page_modules/3_Audit_Trail.py")
            if nav_r.button(
                "Score against golden",
                use_container_width=True,
                key="goto_eval",
            ):
                st.switch_page("page_modules/5_Evaluation.py")
        else:
            if st.button(
                "Open Run detail",
                use_container_width=False,
                key="goto_run_detail",
            ):
                st.switch_page("page_modules/3_Audit_Trail.py")


_live_run_panel()
