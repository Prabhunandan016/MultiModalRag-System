import os
import streamlit as st
from supabase import create_client

_client = None

def get_supabase():
    global _client
    if _client is not None:
        return _client
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
    _client = create_client(url, key)
    return _client
