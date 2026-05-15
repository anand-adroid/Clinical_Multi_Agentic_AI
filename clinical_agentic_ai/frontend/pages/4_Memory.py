"""Pattern Library — long-term derivation patterns promoted from prior runs.

Admin-only. Curators use this page to audit what the system has learned
and to retire patterns that turned out to be wrong.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import api_get, page_header
from components.auth import require_admin


require_admin()


page_header(
    "Pattern Library",
    "Validated derivations promoted from prior runs. Reused by the code "
    "generator for consistency across studies.",
)

patterns = api_get("/memory/patterns", default=[]) or []
if not patterns:
    st.info("No patterns stored yet. Complete a run to populate the library.")
    st.stop()


df = pd.DataFrame(patterns)
keep = [c for c in ["id", "target", "score", "times_used", "signature"] if c in df.columns]
st.dataframe(
    df[keep],
    use_container_width=True,
    hide_index=True,
    height=320,
)


st.subheader("Inspect pattern")
selected = st.selectbox("Pattern", options=df["id"].tolist(), index=0)
pat = next(p for p in patterns if p["id"] == selected)

m1, m2, m3 = st.columns(3)
m1.markdown(f"**Target**  \n`{pat['target']}`")
m2.markdown(f"**Score**  \n{pat['score']:.2f}")
m3.markdown(f"**Reuse count**  \n{pat['times_used']}")

st.markdown(f"**Sources**  \n{', '.join(pat.get('sources') or []) or '—'}")
st.markdown(f"**Rule**  \n{pat.get('rule_text') or '—'}")
st.markdown("**Code**")
st.code(pat["code"], language="python")
