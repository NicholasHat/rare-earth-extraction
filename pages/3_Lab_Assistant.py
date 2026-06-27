"""Pillar C — AI lab assistant (README §8).

Chat against the master DB (read-only) and the Pillar B calculator. This is
the open read path — no write-access gate — because `query_database` runs
through `assistant/sql_guard.py` against a read-only connection: even a
jailbroken model can't mutate anything through this page.
"""
from __future__ import annotations

import streamlit as st

from assistant.agent import run_turn
from database import connection

st.set_page_config(page_title="Lab Assistant", layout="wide")
st.title("Lab Assistant")
st.caption(
    "Pillar C — ask about the extraction database, or plan an experiment with the calculator. "
    "Every number it states comes from a database query or the calculator, never from memory."
)

connection.init_db()

if "chat_display" not in st.session_state:
    st.session_state["chat_display"] = []  # [(role, text)] for rendering
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # raw API message history for the agent

for role, text in st.session_state["chat_display"]:
    with st.chat_message(role):
        st.markdown(text)

if st.button("Clear conversation"):
    st.session_state["chat_display"] = []
    st.session_state["chat_history"] = []
    st.rerun()

user_text = st.chat_input("Ask about the data, or plan an extraction experiment…")
if user_text:
    st.session_state["chat_display"].append(("user", user_text))
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply, updated_history = run_turn(st.session_state["chat_history"], user_text)
            except Exception as e:  # API/auth/etc. — surface, don't crash the chat
                reply = f"Sorry, something went wrong talking to the model: {e}"
                updated_history = st.session_state["chat_history"]
        st.markdown(reply)
    st.session_state["chat_history"] = updated_history
    st.session_state["chat_display"].append(("assistant", reply))
