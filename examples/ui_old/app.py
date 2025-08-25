import streamlit as st
from chatbot_section import chatbot_section
from configuration_section import configuration_section
from documents_section import documents_section
from retrieve_section import retrieve_section
from utils import get_config_names

st.set_page_config(page_title="DSP AI RAG2 UI", layout="wide")
st.sidebar.title("DSP AI RAG2 UI")

# Config dropdown in sidebar
configs = get_config_names()
selected_config = st.sidebar.selectbox("Select Configuration", configs) if configs else None

section = st.sidebar.radio(
    "Navigation",
    ["Chatbot", "Configuration", "Documents", "Retrieve"],
    index=0
)

if section == "Chatbot":
    chatbot_section(selected_config)
elif section == "Configuration":
    configuration_section()
elif section == "Documents":
    documents_section(selected_config)
elif section == "Retrieve":
    retrieve_section(selected_config)
