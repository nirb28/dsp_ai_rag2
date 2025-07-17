import streamlit as st
import requests
from utils import API_BASE_URL

def retrieve_section(selected_config):
    st.header("🔎 Retrieve Endpoints")
    st.info(f"Current Configuration: {selected_config if selected_config else 'None selected'}")
    tabs = st.tabs([
        "Retrieve",
        "Retrieve by ID"
    ])
    # --- /retrieve ---
    with tabs[0]:
        st.subheader("POST /retrieve")
        query = st.text_input("Query Text", key="retrieve_query_text")
        if st.button("Send Retrieve Query", key="retrieve_query_btn"):
            payload = {"query": query}
            if selected_config:
                payload["configuration"] = selected_config
            try:
                resp = requests.post(f"{API_BASE_URL}/retrieve", json=payload)
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to call /retrieve: {e}")
    # --- /retrieve/{id} ---
    with tabs[1]:
        st.subheader("GET /retrieve/{id}")
        retrieve_id = st.text_input("Retrieve ID")
        if st.button("Get Retrieve by ID") and retrieve_id:
            try:
                params = {"configuration": selected_config} if selected_config else {}
                resp = requests.get(f"{API_BASE_URL}/retrieve/{retrieve_id}", params=params)
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to get retrieve by ID: {e}")
