import streamlit as st
import requests
from utils import API_BASE_URL, get_configurations

def configuration_section():
    st.header("🛠️ Configuration Panel")
    tabs = st.tabs([
        "List Configurations",
        "Add Configuration",
        "Delete Configuration",
        "Query",
        "Other Endpoints"
    ])
    # --- List Configurations ---
    with tabs[0]:
        st.subheader("List Configurations")
        if st.button("Refresh Configurations"):
            configs = get_configurations()
            st.json(configs)
    # --- Add Configuration ---
    with tabs[1]:
        st.subheader("Add Configuration")
        config_name = st.text_input("Configuration Name")
        model = st.text_input("Model")
        # Add more fields as needed
        if st.button("Add Configuration"):
            payload = {"name": config_name, "model": model}
            try:
                resp = requests.post(f"{API_BASE_URL}/add_configuration", json=payload)
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to add configuration: {e}")
    # --- Delete Configuration ---
    with tabs[2]:
        st.subheader("Delete Configuration")
        config_name = st.text_input("Configuration Name to Delete")
        if st.button("Delete Configuration"):
            try:
                resp = requests.delete(f"{API_BASE_URL}/delete_configuration/{config_name}")
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to delete configuration: {e}")
    # --- Query ---
    with tabs[3]:
        st.subheader("Query Endpoint Test")
        query = st.text_input("Query Text", key="admin_query_text")
        config = st.text_input("Configuration Name (optional)", key="admin_query_config")
        if st.button("Send Query", key="admin_query_btn"):
            payload = {"query": query}
            if config:
                payload["configuration"] = config
            try:
                resp = requests.post(f"{API_BASE_URL}/query", json=payload)
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to query: {e}")
    # --- Other Endpoints ---
    with tabs[4]:
        st.subheader("Other Endpoints")
        st.info("Add more endpoint forms here as needed.")
