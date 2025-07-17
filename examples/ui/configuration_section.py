import streamlit as st
import pandas as pd
import requests
from utils import API_BASE_URL, get_configurations

def configuration_section():
    st.header("🛠️ Configuration Panel")
    tabs = st.tabs([
        "List Configurations",
        "Add Configuration",
        "Delete Configuration",
        "Duplicate Configuration",
        "Get or Save Configuration"
    ])
    # --- List Configurations ---
    with tabs[0]:
        st.subheader("List Configurations")
        import pandas as pd
        if 'config_table' not in st.session_state:
            st.session_state.config_table = []
        if st.button("Refresh Configurations"):
            configs = get_configurations()
            # Try to extract list of configs
            config_list = []
            if isinstance(configs, dict) and "configurations" in configs:
                config_list = configs["configurations"]
            elif isinstance(configs, list):
                config_list = configs
            else:
                config_list = []
            st.session_state.config_table = config_list
        config_list = st.session_state.get('config_table', [])
        if config_list:
            st.markdown("### Configurations Table")
            columns = list(config_list[0].keys()) if config_list else []
            columns_display = columns + ["Actions"]
            # Find config name column
            # Only 'config' column should be wide
            col_widths = []
            for col in columns:
                if col == "config":
                    col_widths.append(8)
                else:
                    col_widths.append(1)
            col_widths.append(1)  # Actions
            # Table header
            header_cols = st.columns(col_widths)
            for idx, col in enumerate(columns_display):
                header_cols[idx].markdown(f"**{col}**")
            # Table rows
            for i, row in enumerate(config_list):
                row_cols = st.columns(col_widths)
                for j, col in enumerate(columns):
                    row_cols[j].write(str(row.get(col, "")))
                config_name = str(row.get('name') or row.get('config_name') or row.get('id'))
                with row_cols[-1]:
                    if st.button("Delete", key=f"delete_config_btn_{i}"):
                        if st.confirm(f"Are you sure you want to delete configuration '{config_name}'?", key=f"confirm_delete_{i}"):
                            try:
                                resp = requests.delete(f"{API_BASE_URL}/delete_configuration/{config_name}")
                                st.success(f"Deleted {config_name}, Status: {resp.status_code}")
                                st.json(resp.json())
                                # Refresh table after delete
                                configs = get_configurations()
                                if isinstance(configs, dict) and "configurations" in configs:
                                    st.session_state.config_table = configs["configurations"]
                                elif isinstance(configs, list):
                                    st.session_state.config_table = configs
                            except Exception as e:
                                st.error(f"Failed to delete configuration: {e}")
        else:
            st.info("No configurations found.")
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
        if st.button("Delete Configuration") and config_name:
            try:
                resp = requests.delete(f"{API_BASE_URL}/configurations/{config_name}")
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
                # Refresh configs for left section after delete
                if 'config_table' in st.session_state:
                    configs = get_configurations()
                    if isinstance(configs, dict) and "configurations" in configs:
                        st.session_state.config_table = configs["configurations"]
                    elif isinstance(configs, list):
                        st.session_state.config_table = configs
            except Exception as e:
                st.error(f"Failed to delete configuration: {e}")
    # --- Duplicate Configuration ---
    with tabs[3]:
        st.subheader("Duplicate Configuration")
        src_name = st.text_input("Source Configuration Name")
        new_name = st.text_input("New Configuration Name")
        if st.button("Duplicate Configuration") and src_name and new_name:
            payload = {"source": src_name, "new_name": new_name}
            try:
                resp = requests.post(f"{API_BASE_URL}/configurations/duplicate", json=payload)
                st.write(f"Status: {resp.status_code}")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to duplicate configuration: {e}")

    # --- Get or Save Configuration ---
    with tabs[4]:
        st.subheader("Get or Save Configuration")
        # Fetch available configurations for dropdown
        configs = get_configurations()
        config_names = []
        config_map = {}
        if isinstance(configs, dict) and "configurations" in configs:
            for c in configs["configurations"]:
                name = c.get("configuration_name") or c.get("name") or c.get("config_name") or c.get("id")
                if name:
                    config_names.append(name)
                    config_map[name] = c
        elif isinstance(configs, list):
            for c in configs:
                name = c.get("configuration_name") or c.get("name") or c.get("config_name") or c.get("id")
                if name:
                    config_names.append(name)
                    config_map[name] = c
        selected = st.selectbox("Select Configuration to Edit", config_names) if config_names else None
        config_json = None
        if selected:
            # Fetch the specific configuration using the /configurations/:configuration_name endpoint
            try:
                resp = requests.get(f"{API_BASE_URL}/configurations/{selected}")
                if resp.status_code == 200:
                    config_json = resp.json()
                else:
                    st.error(f"Failed to fetch configuration: Status {resp.status_code}")
                    st.json(resp.json())
            except Exception as e:
                st.error(f"Error fetching configuration: {e}")
                config_json = config_map[selected]  # Fallback to the list data
            import json
            col1, col2 = st.columns([1, 5])
            with col1:
                save_button = st.button("Save Configuration")
            
            editable_json = st.text_area(
                "Edit Configuration JSON",
                value=json.dumps(config_json, indent=2, ensure_ascii=False),
                height=500,
            )
            if save_button:
                try:
                    new_config = json.loads(editable_json)
                    resp = requests.post(f"{API_BASE_URL}/configurations", json=new_config)
                    st.write(f"Status: {resp.status_code}")
                    st.json(resp.json())
                    # Reload configuration after save
                    reload_resp = requests.post(f"{API_BASE_URL}/configurations/reload")
                    st.write(f"Reload Status: {reload_resp.status_code}")
                    st.json(reload_resp.json())
                except Exception as e:
                    st.error(f"Failed to save configuration: {e}")
        else:
            st.info("No configurations available to edit.")

