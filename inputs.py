"""Shared source for the product and competitor names.

Streamlit path: app.py stores the names in st.session_state, which is read first.
CLI path (python crew_flow.py ...): session_state does not persist in bare mode, so
the __main__ block fills INPUTS and the helper falls back to it.
"""
import streamlit as st

INPUTS = {"my_product": "", "competitor1": "", "competitor2": ""}


def get_input(key: str) -> str:
    """Return the named input as a str: session_state first, then INPUTS."""
    try:
        value = st.session_state.get(key)
    except Exception:
        value = None
    if isinstance(value, str) and value:
        return value
    fallback = INPUTS.get(key, "")
    return fallback if isinstance(fallback, str) else ""
