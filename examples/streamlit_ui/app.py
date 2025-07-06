import streamlit as st
import requests
import json
import os
import time
from typing import List, Dict, Any, Optional
from utils import apply_custom_css, render_header, format_sources

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

# Apply custom styling
apply_custom_css()

# Define global variables for columns
# These will be initialized in the main function
col1 = None
col2 = None
col3 = None
col4 = None

# Constants
API_BASE_URL = "http://localhost:9000"  # Default API URL
MAX_RESPONSE_DISPLAY_LENGTH = 12000  # Characters to display in the chat

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "configurations" not in st.session_state:
    st.session_state.configurations = []
if "current_config" not in st.session_state:
    st.session_state.current_config = "default"

# Helper Functions
def show_status_message(message, type="info"):
    """Display a status message in the status container"""
    # Use the 4th column (extreme right) for status messages
    global col4
    if col4 is not None:
        with col4:
            if type == "info":
                st.info(message)
            elif type == "success":
                st.success(message)
            elif type == "error":
                st.error(message)
            elif type == "warning":
                st.warning(message)
    else:
        # Fallback if column is not initialized
        if type == "info":
            st.sidebar.info(message)
        elif type == "success":
            st.sidebar.success(message)
        elif type == "error":
            st.sidebar.error(message)
        elif type == "warning":
            st.sidebar.warning(message)

def reload_configurations():
    """Force reload configurations from file on the API server"""
    try:
        show_status_message("Requesting server to reload configurations from file...", "info")
        response = requests.post(f"{API_BASE_URL}/api/v1/configurations/reload", timeout=5)
        if response.status_code == 200:
            result = response.json()
            show_status_message(result.get("message", "Configurations reloaded successfully"), "success")
            return True
        else:
            show_status_message(f"Failed to reload configurations: HTTP {response.status_code}", "error")
            return False
    except Exception as e:
        show_status_message(f"Error reloading configurations: {str(e)}", "error")
        return False


def get_configurations():
    """Fetch available configurations from the API"""
    try:
        show_status_message(f"Attempting to connect to API at {API_BASE_URL}/api/v1/configurations", "info")
        response = requests.get(f"{API_BASE_URL}/api/v1/configurations", timeout=5)
        if response.status_code == 200:
            config_data = response.json()
            show_status_message("Successfully connected to API and fetched configurations", "success")
            return config_data.get("configurations", [])
        else:
            error_msg = f"Error fetching configurations: HTTP {response.status_code}"
            if response.status_code == 404:
                error_msg += " - Endpoint not found. Is the API server running at the correct URL?"
            show_status_message(error_msg, "error")
            show_status_message("You may need to start the API server or check the API_BASE_URL", "info")
            return []
    except requests.exceptions.ConnectionError:
        show_status_message(f"Connection Error: Unable to connect to {API_BASE_URL}", "error")
        show_status_message("Please make sure the API server is running and accessible", "info")
        return []
    except requests.exceptions.Timeout:
        show_status_message("Request timed out. The API server might be slow or unavailable.", "error")
        return []
    except Exception as e:
        show_status_message(f"Error connecting to API: {str(e)}", "error")
        return []

def query_rag(query: str, config_name: str):
    """Send query to RAG API and get response"""
    if not query or not config_name:
        return None
    
    # We'll avoid sending context items for now as it seems to cause issues
    # with the API expecting a specific format
    data = {
        "query": query,
        "configuration_name": config_name,
        "include_metadata": True
        # Removed context_items for now
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/query", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error querying RAG system: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def upload_document(file, config_name):
    """Upload a document to the specified configuration"""
    if not file or not config_name:
        show_status_message("Missing file or configuration", "error")
        return False
    
    try:
        show_status_message(f"Uploading document to {config_name} configuration...", "info")
        files = {"file": (file.name, file.getvalue())}
        data = {"configuration_name": config_name, "process_immediately": "true"}
        
        response = requests.post(f"{API_BASE_URL}/api/v1/upload", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            show_status_message(f"Document uploaded successfully: {result.get('document_id')}", "success")
            return True
        else:
            show_status_message(f"Error uploading document: {response.status_code}", "error")
            return False
    except Exception as e:
        show_status_message(f"Error connecting to API: {str(e)}", "error")
        return False

# UI Components
def render_sidebar():
    """Render the sidebar with configuration options"""
    global API_BASE_URL
    # Use only the sidebar without the column context
    with st.sidebar:
        st.title("RAG Chatbot Settings")
        
        # Configuration management buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh Configurations"):
                st.session_state.configurations = get_configurations()
        with col2:
            if st.button("Reload From File", help="Force reload configurations from storage file"):
                if reload_configurations():
                    # If successful, refresh the configurations
                    st.session_state.configurations = get_configurations()
        
        # Configuration selector
        config_options = ["default"] + [config["name"] for config in st.session_state.configurations if config["name"] != "default"]
        st.session_state.current_config = st.selectbox(
            "Select Configuration", 
            options=config_options,
            index=config_options.index(st.session_state.current_config) if st.session_state.current_config in config_options else 0
        )
        
        st.divider()
        
        # Document uploader
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx", "md"])
        if uploaded_file is not None:
            if st.button("Upload to Selected Configuration"):
                upload_document(uploaded_file, st.session_state.current_config)
        
        st.divider()
        
        # API connection settings
        st.subheader("API Connection")
        api_url = st.text_input("API URL", value=API_BASE_URL)
        if api_url != API_BASE_URL and st.button("Update API URL"):
            API_BASE_URL = api_url
            st.success(f"API URL updated to: {API_BASE_URL}")
            st.session_state.configurations = get_configurations()

# Function to display chat messages
def display_chat_messages():
    """Display chat messages from session state"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(source["content"])
                        st.markdown("---")

# Function to handle new user input
def handle_user_input():
    """Process new user input from chat_input"""
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with spinner while loading
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                response = query_rag(prompt, st.session_state.current_config)
                
                if response:
                    answer = response.get("answer", "Sorry, I couldn't generate a response.")
                    sources = response.get("sources", [])
                    
                    # Truncate extremely long responses for display
                    if len(answer) > MAX_RESPONSE_DISPLAY_LENGTH:
                        display_answer = answer[:MAX_RESPONSE_DISPLAY_LENGTH] + "... [Response truncated due to length]"
                    else:
                        display_answer = answer
                        
                    message_placeholder.markdown(display_answer)
                    
                    # Save assistant response to session
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": display_answer,
                        "sources": sources
                    })
                    
                    # Display sources
                    if sources:
                        with st.expander("View Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(source["content"])
                                st.markdown("---")
                else:
                    message_placeholder.markdown("Sorry, I couldn't generate a response. Please try again.")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Sorry, I couldn't generate a response. Please try again."
                    })

def render_chat():
    """Render the chat interface header"""
    render_header()
    st.caption(f"Using configuration: {st.session_state.current_config}")

# Main Application
def main():
    # Initialize the global columns
    global col1, col2, col3, col4
    
    # Create a simple two-column layout first
    # This ensures col4 is initialized before any status messages are displayed
    col1, col4 = st.columns([7, 3])
    
    # Add content to the status column
    with col4:
        st.subheader("Status Messages")
        st.markdown("""<div style='height: 10px;'></div>""", unsafe_allow_html=True)
    
    # Now render the sidebar (after columns are initialized)
    render_sidebar()
    
    # Add content to the main column
    with col1:
        # Render the chat header
        render_chat()
        
        # Display chat messages
        display_chat_messages()
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()
    
    # Initialize or fetch configurations
    if "configurations" not in st.session_state or not st.session_state.configurations:
        st.session_state.configurations = get_configurations()
    
    # Handle user input (MUST be outside any column, form, etc.)
    handle_user_input()

if __name__ == "__main__":
    main()
