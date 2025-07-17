import streamlit as st
import requests
from utils import API_BASE_URL

def documents_section(selected_config):
    st.header("📄 Documents Management")
    st.info(f"Current Configuration: {selected_config if selected_config else 'None selected'}")
    tabs = st.tabs([
        "List Documents",
        "Get Chunks",
        "Upload Document",
        "Delete Document"
    ])
    # --- List Documents ---
    with tabs[0]:
        st.subheader("List Documents")
        if st.button("Refresh Document List"):
            try:
                params = {"configuration_name": selected_config} if selected_config else {}
                resp = requests.get(f"{API_BASE_URL}/documents", params=params)
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to list documents: {e}")
    # --- Get Chunks ---
    with tabs[1]:
        st.subheader("Get Chunks by Document ID")
        doc_id = st.text_input("Document ID for Chunks")
        if st.button("Get Chunks") and doc_id:
            try:
                clean_doc_id = doc_id.strip().strip('"')
                params = {"configuration_name": selected_config} if selected_config else {}
                resp = requests.get(f"{API_BASE_URL}/documents/{clean_doc_id}/chunks", params=params)
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to get chunks: {e}")
    # --- Upload Document ---
    with tabs[2]:
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Choose a file")
        if st.button("Upload") and uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {"configuration_name": selected_config} if selected_config else {}
            try:
                resp = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to upload document: {e}")
    # --- Delete Document ---
    with tabs[3]:
        st.subheader("Delete Document by ID")
        doc_id = st.text_input("Document ID to Delete")
        if st.button("Delete Document") and doc_id:
            try:
                clean_doc_id = doc_id.strip().strip('"')
                params = {"configuration_name": selected_config} if selected_config else {}
                resp = requests.delete(f"{API_BASE_URL}/documents/{clean_doc_id}/chunks", params=params)
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to delete document: {e}")
