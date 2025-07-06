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
def get_configurations():
    """Fetch available configurations from the API"""
    try:
        st.info(f"Attempting to connect to API at {API_BASE_URL}/api/v1/configurations")
        response = requests.get(f"{API_BASE_URL}/api/v1/configurations", timeout=5)
        if response.status_code == 200:
            config_data = response.json()
            st.success("Successfully connected to API and fetched configurations")
            return config_data.get("configurations", [])
        else:
            error_msg = f"Error fetching configurations: HTTP {response.status_code}"
            if response.status_code == 404:
                error_msg += " - Endpoint not found. Is the API server running at the correct URL?"
            st.error(error_msg)
            st.info("You may need to start the API server or check the API_BASE_URL")
            return []
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Unable to connect to {API_BASE_URL}")
        st.info("Please make sure the API server is running and accessible")
        return []
    except requests.exceptions.Timeout:
        st.error("Request timed out. The API server might be slow or unavailable.")
        return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
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
        return False, "Missing file or configuration"
    
    try:
        files = {"file": (file.name, file.getvalue())}
        data = {"configuration_name": config_name, "process_immediately": "true"}
        
        response = requests.post(f"{API_BASE_URL}/api/v1/upload", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            return True, f"Document uploaded successfully: {result.get('document_id')}"
        else:
            return False, f"Error uploading document: {response.status_code}"
    except Exception as e:
        return False, f"Error connecting to API: {str(e)}"

# UI Components
def render_sidebar():
    """Render the sidebar with configuration options"""
    global API_BASE_URL
    with st.sidebar:
        st.title("RAG Chatbot Settings")
        
        # Refresh configurations button
        if st.button("Refresh Configurations"):
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
                success, message = upload_document(uploaded_file, st.session_state.current_config)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        st.divider()
        
        # API connection settings
        st.subheader("API Connection")
        api_url = st.text_input("API URL", value=API_BASE_URL)
        if api_url != API_BASE_URL and st.button("Update API URL"):
            API_BASE_URL = api_url
            st.success(f"API URL updated to: {API_BASE_URL}")
            st.session_state.configurations = get_configurations()

def render_chat():
    """Render the chat interface"""
    render_header()
    st.caption(f"Using configuration: {st.session_state.current_config}")
    
    # Display chat messages
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

    # Input for new message
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

# Main Application
def main():
    # Fetch configurations on startup if needed
    if not st.session_state.configurations:
        st.session_state.configurations = get_configurations()
    
    # Render sidebar and chat interface
    render_sidebar()
    render_chat()
    
    # Clear chat button at the bottom
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
