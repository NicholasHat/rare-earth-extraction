"""Hosted read-only viewer: the Database and Analysis pages over a published
snapshot of the master DB — nothing else of the app.

    streamlit run viewer.py

This is the entry point for a hosted deployment (Streamlit Community Cloud:
main file `viewer.py`). It needs no API key and no .env. The extraction and
review page (app.py), the calculator and the assistant are not part of it, so
a visitor can neither upload, approve nor spend anything; both pages it shows
open the DB read-only.

The data is `viewer_data/master.db`, written by `python -m database.snapshot`
from the working DB. It is gitignored here — this repository is public — and
lives only in the private hosting repo. To publish newly approved papers (or
new code): `./publish_viewer.sh`.
"""
import os
from pathlib import Path

# Before anything imports config: point the app at the snapshot. A DB_PATH
# already in the environment wins.
os.environ.setdefault("DB_PATH", str(Path(__file__).resolve().parent / "viewer_data" / "master.db"))

import streamlit as st  # noqa: E402

st.navigation([
    st.Page("pages/1_Database.py", title="Database", default=True),
    st.Page("pages/4_Analysis.py", title="Analysis"),
]).run()
