"""The single write-access gate (README §3).

Every mutating action (upload, extraction, approve/commit) calls
`require_write_access()` first. Read paths never call it.

  REQUIRE_PASSWORD=false  -> returns True immediately (current local-demo default)
  REQUIRE_PASSWORD=true   -> enforces the shared password, unlocking the whole
                             write-side UI for the session on success

Flipping from demo to shared use is one .env change, not a rebuild.
"""
from __future__ import annotations

import hmac

import streamlit as st

import config

_SESSION_KEY = "write_unlocked"


def is_unlocked() -> bool:
    if not config.REQUIRE_PASSWORD:
        return True
    return bool(st.session_state.get(_SESSION_KEY, False))


def require_write_access() -> bool:
    """Return True if writes are permitted, else render a password prompt.

    Call at the top of any write action. When it returns False, the caller
    should stop (the prompt has been rendered in its place).
    """
    if not config.REQUIRE_PASSWORD:
        return True

    if st.session_state.get(_SESSION_KEY, False):
        return True

    st.info("This action requires the shared write password.")
    with st.form("write_unlock", clear_on_submit=False):
        entered = st.text_input("Write password", type="password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        if config.WRITE_PASSWORD and hmac.compare_digest(entered, config.WRITE_PASSWORD):
            st.session_state[_SESSION_KEY] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False
