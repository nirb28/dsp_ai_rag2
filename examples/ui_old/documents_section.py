import streamlit as st
import requests
from utils import API_BASE_URL

def documents_section(selected_config):
    st.header("📄 Documents Management")

    if 'doc_tab_index' not in st.session_state:
        st.session_state['doc_tab_index'] = 0
    tabs = st.tabs([
        "List Documents",
        "Get Chunks",
        "Upload Document",
        "Upload Text",
        "Delete Document",
        "Delete All Documents"
    ])
    # --- List Documents ---
    with tabs[0]:
        st.subheader("List Documents")
        doc_list = []
        if st.button("Refresh Document List"):
            try:
                params = {"configuration_name": selected_config} if selected_config else {}
                resp = requests.get(f"{API_BASE_URL}/documents", params=params)
                st.write(f"Status: {resp.status_code}")
                data = resp.json()
                # Try to extract list of docs
                if isinstance(data, dict) and "documents" in data:
                    doc_list = data["documents"]
                elif isinstance(data, list):
                    doc_list = data
                else:
                    doc_list = []
                st.session_state['doc_table'] = doc_list
            except Exception as e:
                st.error(f"Failed to list documents: {e}")
        doc_list = st.session_state.get('doc_table', [])
        if doc_list:
            st.markdown("### Documents Table")
            columns = [c for c in list(doc_list[0].keys()) if c != 'filepath'] if doc_list else []
            columns_display = columns + ["Actions"]
            # Find document id column
            # All columns small except Actions
            col_widths = [1 for _ in columns]
            col_widths.append(1)  # Actions
            # Table header
            header_cols = st.columns(col_widths)
            for idx, col in enumerate(columns_display):
                header_cols[idx].markdown(f"**{col}**")
            # Table rows
            for i, row in enumerate(doc_list):
                row_cols = st.columns(col_widths)
                for j, col in enumerate(columns):
                    value = str(row.get(col, ""))
                    if col in ('id', 'document_id', 'doc_id'):
                        # Render as code with copy button
                        row_cols[j].code(value, language="")
                    else:
                        row_cols[j].write(value)
                doc_id = str(row.get('id') or row.get('document_id') or row.get('doc_id'))
                clean_doc_id = doc_id.strip().strip('"')
                with row_cols[-1]:
                    if st.button("Delete", key=f"delete_doc_btn_{i}"):
                        if st.confirm(f"Are you sure you want to delete document '{clean_doc_id}'?", key=f"confirm_delete_doc_{i}"):
                            params = {"configuration_name": selected_config} if selected_config else {}
                            try:
                                del_resp = requests.delete(f"{API_BASE_URL}/documents/{clean_doc_id}/chunks", params=params)
                                st.success(f"Deleted {clean_doc_id}, Status: {del_resp.status_code}")
                                st.json(del_resp.json())
                                # Refresh table after delete
                                params = {"configuration_name": selected_config} if selected_config else {}
                                resp = requests.get(f"{API_BASE_URL}/documents", params=params)
                                data = resp.json()
                                if isinstance(data, dict) and "documents" in data:
                                    st.session_state['doc_table'] = data["documents"]
                                elif isinstance(data, list):
                                    st.session_state['doc_table'] = data
                            except Exception as e:
                                st.error(f"Failed to delete document: {e}")
        else:
            st.info("No documents found.")
    # --- Get Chunks ---
    with tabs[1]:
        st.subheader("Get Chunks by Document ID")
        # If doc_id was set from a hyperlink click, use it
        doc_id = st.session_state.get('get_chunks_doc_id', "")
        doc_id = st.text_input("Document ID for Chunks", value=doc_id, key="get_chunks_doc_id_input")
        auto_trigger = st.session_state.get('doc_tab_index', 0) == 1 and st.session_state.get('get_chunks_doc_id')
        include_vector = st.checkbox("Include Vectors", value=False, key="get_chunks_include_vector")
        if st.button("Get Chunks") or auto_trigger:
            if doc_id:
                try:
                    clean_doc_id = doc_id.strip().strip('"')
                    params = {"configuration_name": selected_config} if selected_config else {}
                    params["include_vectors"] = str(include_vector).lower()
                    resp = requests.get(f"{API_BASE_URL}/documents/{clean_doc_id}/chunks", params=params)
                    st.write(f"Status: {resp.status_code}")
                    st.json(resp.json())
                except Exception as e:
                    st.error(f"Failed to get chunks: {e}")
            # Reset auto-trigger so it doesn't repeat
            if auto_trigger:
                st.session_state['doc_tab_index'] = 1
                st.session_state['get_chunks_doc_id'] = ""
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

    # --- Upload Text ---
    with tabs[3]:
        st.subheader("Upload Up to 4 Text Documents")
        st.markdown("Enter one document per box below. Leave unused boxes empty.")
        doc1 = st.text_area("Document 1", key="upload_text_doc1")
        doc2 = st.text_area("Document 2", key="upload_text_doc2")
        doc3 = st.text_area("Document 3", key="upload_text_doc3")
        doc4 = st.text_area("Document 4", key="upload_text_doc4")
        docs = [doc1, doc2, doc3, doc4]
        if st.button("Upload Texts"):
            payload_docs = [{"content": d} for d in docs if d.strip()]
            if not payload_docs:
                st.error("No documents to upload.")
            else:
                payload = {"documents": payload_docs}
                if selected_config:
                    payload["configuration_name"] = selected_config
                try:
                    resp = requests.post(f"{API_BASE_URL}/upload/text", json=payload)
                    st.write(f"Status: {resp.status_code}")
                    data = resp.json()
                    if isinstance(data, list):
                        import pandas as pd
                        st.dataframe(pd.DataFrame(data), use_container_width=True)
                        with st.expander("Show Raw JSON Response"):
                            st.json(data)
                    else:
                        with st.expander("Show Raw JSON Response", expanded=True):
                            st.json(data)
                except Exception as e:
                    st.error(f"Failed to upload text documents: {e}")

    # --- Delete Document ---
    with tabs[4]:
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
                
    # --- Delete All Documents ---
    with tabs[5]:
        st.subheader("Delete All Documents")
        st.warning("⚠️ This will delete ALL documents from the system. This action cannot be undone.")
        confirm = st.checkbox("I understand that this will permanently delete all documents")
        params = {"confirm": "true"}
        if selected_config:
            params["configuration_name"] = selected_config
        if st.button("Delete All Documents", disabled=not confirm):
            try:
                resp = requests.delete(f"{API_BASE_URL}/documents", params=params)
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
                st.success("All documents have been deleted successfully.")
            except Exception as e:
                st.error(f"Failed to delete all documents: {e}")
