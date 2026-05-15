"""Evaluation — score a completed run against an expected golden table.

Admin-only. Used by QA to verify a release before promoting it; only
relevant when a hand-authored golden CSV is available.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from components import API, api_get, page_header, remember_run
from components.auth import require_admin


require_admin()


ROOT = Path(__file__).resolve().parents[2]


page_header(
    "Evaluation",
    "Compare a completed run against a golden expected table to measure "
    "correctness, coverage, and reliability.",
)


runs = api_get("/runs", default=[]) or []
runs = [r for r in runs if str(r.get("status", "")).startswith("completed")]
if not runs:
    st.info("No completed runs to evaluate. Finish a run first.")
    st.stop()


ids = [r["id"] for r in runs]
preferred = st.session_state.get("last_run_id")
index = ids.index(preferred) if preferred in ids else 0
selected = st.selectbox("Run", options=ids, index=index)
remember_run(selected)

default_golden = str(ROOT / "data" / "golden" / "expected.csv")
golden_path = st.text_input("Golden table path", value=default_golden)


if st.button("Run evaluation", type="primary"):
    with st.spinner("Scoring..."):
        try:
            resp = httpx.get(
                f"{API}/eval/{selected}",
                params={"golden_path": golden_path},
                timeout=60,
            )
        except httpx.HTTPError as exc:
            st.error(f"Could not reach evaluation endpoint: {exc}")
            st.stop()
    if resp.status_code != 200:
        st.error(f"Evaluation failed (HTTP {resp.status_code}). {resp.text}")
        st.stop()
    result = resp.json()

    m1, m2, m3 = st.columns(3)
    m1.metric("Correctness", f"{result['correctness']:.2%}")
    m2.metric("Coverage", f"{result['coverage']:.2%}")
    m3.metric("Rows evaluated", result["reliability"]["rows_evaluated"])

    st.subheader("Per-target match rates")
    rows = []
    for tgt, info in result["per_target"].items():
        if info.get("in_golden") and info.get("in_output"):
            rows.append(
                {
                    "target": tgt,
                    "match_rate": f"{info['match']:.2%}",
                    "matched": info["matched"],
                    "total": info["count"],
                }
            )
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    mismatches_found = False
    for tgt, info in result["per_target"].items():
        if info.get("mismatched_examples"):
            mismatches_found = True
            with st.expander(f"Mismatched examples — {tgt}", expanded=False):
                st.dataframe(
                    pd.DataFrame(info["mismatched_examples"]),
                    use_container_width=True,
                    hide_index=True,
                )
    if not mismatches_found:
        st.success("No mismatches against the golden table.")
