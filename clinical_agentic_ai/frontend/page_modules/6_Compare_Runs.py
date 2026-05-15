"""Run Comparison — diff two runs side-by-side.

Admin-only. Used by power users for reproducibility checks across spec
revisions, LLM-model upgrades, or memory promotions.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import api_get, page_header
from components.auth import require_admin


require_admin()


page_header(
    "Run Comparison",
    "Compare two runs side by side. When spec and dataset hashes match, "
    "every per-target code hash should match as well.",
)


runs = api_get("/runs", timeout=5.0, default=[]) or []
if len(runs) < 2:
    st.info("At least two runs are required for comparison.")
    st.stop()


ids = [r["id"] for r in runs]
col_a, col_b = st.columns(2)
with col_a:
    run_a = st.selectbox("Run A", ids, index=0)
with col_b:
    run_b = st.selectbox("Run B", ids, index=min(1, len(ids) - 1))

if run_a == run_b:
    st.warning("Select two different runs to compare.")
    st.stop()


def _short(h: str | None, n: int = 14) -> str:
    if not h:
        return "—"
    return h[:n] + ("…" if len(h) > n else "")


rep_a = api_get(f"/runs/{run_a}/audit/report")
rep_b = api_get(f"/runs/{run_b}/audit/report")
if not rep_a or not rep_b:
    st.error("Audit report unavailable for one or both runs. Complete the runs first.")
    st.stop()


# ---------------------------------------------------------------- fingerprints

st.subheader("Run fingerprints")

fp_rows = [
    {"field": "status", "A": rep_a["status"], "B": rep_b["status"]},
    {"field": "spec_hash", "A": _short(rep_a["spec_hash"]), "B": _short(rep_b["spec_hash"])},
    {"field": "dataset_hash", "A": _short(rep_a["dataset_hash"]), "B": _short(rep_b["dataset_hash"])},
    {"field": "llm_enabled", "A": rep_a["config"]["llm_enabled"], "B": rep_b["config"]["llm_enabled"]},
    {"field": "model", "A": rep_a["config"]["model"], "B": rep_b["config"]["model"]},
    {
        "field": "topo_order",
        "A": " > ".join(rep_a.get("topo_order") or []),
        "B": " > ".join(rep_b.get("topo_order") or []),
    },
]
fp = pd.DataFrame(fp_rows)
fp["match"] = fp["A"].astype(str) == fp["B"].astype(str)
fp["match"] = fp["match"].map(lambda x: "Match" if x else "Differ")
st.dataframe(fp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- lineage

st.subheader("Per-target lineage")

la = {item["target"]: item for item in rep_a.get("lineage", [])}
lb = {item["target"]: item for item in rep_b.get("lineage", [])}
targets = sorted(set(la) | set(lb))

rows = []
for t in targets:
    a = la.get(t, {})
    b = lb.get(t, {})
    rows.append(
        {
            "target": t,
            "generator (A)": a.get("generator", "—"),
            "generator (B)": b.get("generator", "—"),
            "code_hash (A)": _short(a.get("code_hash"), 10),
            "code_hash (B)": _short(b.get("code_hash"), 10),
            "same code": "Match" if a.get("code_hash") == b.get("code_hash") else "Differ",
            "status (A)": a.get("status", "—"),
            "status (B)": b.get("status", "—"),
            "attempts (A)": a.get("attempts", "—"),
            "attempts (B)": b.get("attempts", "—"),
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption(
    "Reproducibility: when spec_hash and dataset_hash match, every code_hash "
    "must match. Different generators (LLM vs rule) producing the same code_hash "
    "indicate that long-term memory has converged the two paths."
)


# ---------------------------------------------------------------- output diff

st.subheader("Output diff")


def _output(run_id: str) -> pd.DataFrame | None:
    o = api_get(f"/runs/{run_id}/output")
    if not o:
        return None
    return pd.DataFrame(o.get("rows", []))


df_a = _output(run_a)
df_b = _output(run_b)
if df_a is None or df_b is None:
    st.warning("Output not available for one or both runs.")
    st.stop()


common_cols = [c for c in df_a.columns if c in df_b.columns]
pk_candidates = ["patient_id", "id", "subject_id"]
pk = next((c for c in pk_candidates if c in common_cols), None)

if pk is None:
    st.info("No common primary key column. Showing first 20 rows of each side by side.")
    c1, c2 = st.columns(2)
    c1.write("**Run A**")
    c1.dataframe(df_a.head(20), use_container_width=True, hide_index=True)
    c2.write("**Run B**")
    c2.dataframe(df_b.head(20), use_container_width=True, hide_index=True)
else:
    merged = df_a.merge(df_b, on=pk, suffixes=("_A", "_B"), how="outer")
    raw_cols = {
        "age", "sex", "treatment_start_date", "visit_date", "lab_value", "response",
    }
    derived_cols = [
        c for c in df_a.columns
        if c != pk and c in df_b.columns and c not in raw_cols
    ]
    diffs = []
    for c in derived_cols:
        a_col = f"{c}_A"
        b_col = f"{c}_B"
        if a_col not in merged.columns or b_col not in merged.columns:
            continue
        mismatch = merged[a_col].astype(str) != merged[b_col].astype(str)
        diffs.append(
            {
                "column": c,
                "matched": int((~mismatch).sum()),
                "mismatched": int(mismatch.sum()),
                "total": len(merged),
                "rate": f"{(1 - mismatch.mean()):.2%}",
            }
        )

    if diffs:
        st.dataframe(pd.DataFrame(diffs), use_container_width=True, hide_index=True)
        any_mis = any(d["mismatched"] for d in diffs)
        if not any_mis:
            st.success("Outputs are bit-identical. Determinism verified.")
        else:
            st.warning(
                "Outputs differ. Inspect the mismatched columns above. This is "
                "expected if the runs took different generator paths."
            )
    else:
        st.caption("No derived columns in common to diff.")
