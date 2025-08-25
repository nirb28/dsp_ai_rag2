import streamlit as st
import requests
from utils import API_BASE_URL

def retrieve_section(selected_config):
    st.header("Retrieve Endpoints")
    st.info(f"Current Configuration: {selected_config if selected_config else 'None selected'}")
    tabs = st.tabs([
        "Retrieve",
        "Multi Retrieve"
    ])
    # --- /retrieve ---
    with tabs[0]:
        st.subheader("POST /retrieve")
        col1, col2 = st.columns([2,2])
        with col1:
            query = st.text_input("Query", value="What is Computer Vision?", key="basic_retrieve_query_text")
        with col2:
            config_name = st.text_input("Configuration Name", value="batch_ml_ai_basics_test", key="basic_retrieve_config_name")
        col3, col4, col5 = st.columns([1,1,1])
        with col3:
            k = st.number_input("k", min_value=1, value=5, key="basic_retrieve_k")
        with col4:
            similarity_threshold = st.number_input("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="basic_retrieve_similarity_threshold")
        with col5:
            include_metadata = st.checkbox("Include Metadata", value=False, key="basic_retrieve_include_metadata")
        col6, col7 = st.columns([1,1])
        with col6:
            use_reranking = st.checkbox("Use Reranking", value=True, key="basic_retrieve_use_reranking")
        with col7:
            include_vectors = st.checkbox("Include Vectors", value=False, key="basic_retrieve_include_vectors")
        if st.button("Send Retrieve Query", key="basic_retrieve_query_btn"):
            payload = {
                "query": query,
                "configuration_name": config_name,
                "include_metadata": include_metadata,
                "similarity_threshold": similarity_threshold,
                "k": k,
                "use_reranking": use_reranking,
                "include_vectors": include_vectors
            }
            try:
                resp = requests.post(f"{API_BASE_URL}/retrieve", json=payload)
                data = resp.json()
                st.write(f"Status: {resp.status_code}")
                if isinstance(data, list):
                    import pandas as pd
                    st.dataframe(pd.DataFrame(data), use_container_width=True)
                    with st.expander("Show Raw JSON Response"):
                        st.json(data)
                else:
                    with st.expander("Show Raw JSON Response", expanded=True):
                        st.json(data)
            except Exception as e:
                st.error(f"Failed to call /retrieve: {e}")
    # --- Multi Retrieve ---
    with tabs[1]:
        st.subheader("Multi Retrieve (POST /retrieve)")
        col1, col2 = st.columns([2,2])
        with col1:
            multi_query = st.text_input("Query", value="What is Computer Vision?", key="multi_retrieve_query")
        with col2:
            config_names = st.text_input("Configuration Names (comma separated)", value="batch_ml_ai_basics_test, batch_rl-docs_test", key="multi_retrieve_config_names")
        col3, col4, col5 = st.columns([1,1,1])
        with col3:
            fusion_method = st.selectbox("Fusion Method", ["rrf", "other"], index=0, key="multi_retrieve_fusion_method")
        with col4:
            k = st.number_input("k", min_value=1, value=10, key="multi_retrieve_k")
        with col5:
            similarity_threshold = st.number_input("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01, key="multi_retrieve_similarity_threshold")
        col6, col7, col8 = st.columns([1,1,1])
        with col6:
            include_metadata = st.checkbox("Include Metadata", value=False, key="multi_retrieve_include_metadata")
        with col7:
            use_reranking = st.checkbox("Use Reranking", value=True, key="multi_retrieve_use_reranking")
        with col8:
            include_vectors = st.checkbox("Include Vectors", value=False, key="multi_retrieve_include_vectors")
        if st.button("Multi Retrieve"):
            payload = {
                "query": multi_query,
                "configuration_names": [c.strip() for c in config_names.split(",") if c.strip()],
                "fusion_method": fusion_method,
                "k": k,
                "include_metadata": include_metadata,
                "similarity_threshold": similarity_threshold,
                "use_reranking": use_reranking,
                "include_vectors": include_vectors
            }
            try:
                resp = requests.post(f"{API_BASE_URL}/retrieve", json=payload)
                data = resp.json()
                st.write(f"Status: {resp.status_code}")
                if isinstance(data, list):
                    import pandas as pd
                    st.dataframe(pd.DataFrame(data), use_container_width=True)
                    with st.expander("Show Raw JSON Response"):
                        st.json(data)
                else:
                    with st.expander("Show Raw JSON Response", expanded=True):
                        st.json(data)
            except Exception as e:
                st.error(f"Failed to multi-retrieve: {e}")
