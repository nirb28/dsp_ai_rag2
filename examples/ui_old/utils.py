import requests
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:9000")

def get_config_names():
    try:
        resp = requests.get(f"{API_BASE_URL}/configurations?names_only=true")
        resp.raise_for_status()
        return resp.json().get("names", [])
    except Exception as e:
        st.error(f"Failed to fetch configurations: {e}")
        return []

def get_configurations():
    try:
        resp = requests.get(f"{API_BASE_URL}/configurations")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch configurations: {e}")
        return {}

def query_rag(payload):
    try:
        resp = requests.post(f"{API_BASE_URL}/query", json=payload)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to query: {e}")
        return {"error": str(e)}

# Add more endpoint helpers as needed, e.g., add_configuration, delete_configuration, etc.
