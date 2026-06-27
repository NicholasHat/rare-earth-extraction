"""Master database viewer (README §5) — papers, extraction run history, and
the review audit log. Moved here from the app.py sidebar so the sidebar stays
free for navigation.

Read-only: uses get_readonly_conn() throughout, same as Pillars B and C, even
though this lives under Pillar A's umbrella — browsing history should never
be able to mutate anything.
"""
from __future__ import annotations

import streamlit as st

import config
from database import browse, connection, extractions_repo

st.set_page_config(page_title="Master Database", layout="wide")
st.title("Master Database")
st.caption("Browse papers, extraction run history, and the review audit log.")

connection.init_db()
conn = connection.get_readonly_conn()
try:
    n_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    n_runs = conn.execute(
        "SELECT COUNT(*) FROM prompt_runs WHERE status='approved'"
    ).fetchone()[0]
    n_best = extractions_repo.count_current_best(conn)
    papers_df = browse.list_papers(conn)
    runs_df = browse.list_prompt_runs(conn)
    log_df = browse.list_review_log(conn)
finally:
    conn.close()

m1, m2, m3 = st.columns(3)
m1.metric("Papers", n_papers)
m2.metric("Approved runs", n_runs)
m3.metric("Rows in current-best view", n_best)

write_mode = "🔒 password required" if config.REQUIRE_PASSWORD else "🔓 open (no password)"
st.caption(
    f"Pinned prompt: `{config.EXTRACTION_PROMPT_VERSION}` &nbsp;·&nbsp; "
    f"Model: `{config.EXTRACTION_MODEL}` &nbsp;·&nbsp; Write mode: {write_mode}"
)

st.divider()
tab_papers, tab_runs, tab_log, tab_data = st.tabs(
    ["Papers", "Extraction runs", "Review log", "Browse current-best data"]
)

with tab_papers:
    if papers_df.empty:
        st.info("No papers yet — upload one on the home page.")
    else:
        st.dataframe(papers_df, use_container_width=True, hide_index=True)

with tab_runs:
    if runs_df.empty:
        st.info("No extraction runs yet.")
    else:
        st.dataframe(runs_df, use_container_width=True, hide_index=True)

with tab_log:
    if log_df.empty:
        st.info("No review actions logged yet.")
    else:
        st.dataframe(log_df, use_container_width=True, hide_index=True)

with tab_data:
    conn = connection.get_readonly_conn()
    try:
        best_df = extractions_repo.current_best(conn)
    finally:
        conn.close()
    if best_df.empty:
        st.info("No approved data yet.")
    else:
        elements = sorted(
            best_df["Rare Earth Elements (REY:La, Ce, Nd)"].dropna().unique()
        )
        chosen = st.multiselect("Filter by element", elements)
        view = (
            best_df[best_df["Rare Earth Elements (REY:La, Ce, Nd)"].isin(chosen)]
            if chosen
            else best_df
        )
        st.write(f"{len(view)} row(s)")
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=view.to_csv(index=False),
            file_name="current_best_export.csv",
            mime="text/csv",
        )
