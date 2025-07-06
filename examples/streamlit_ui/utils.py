import streamlit as st
import base64

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Chat styling */
    .stChatMessage {
        padding: 1rem 0;
    }
    
    .stChatMessage .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
    }
    
    .message-container {
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    
    .user-message {
        background-color: #e9f5ff;
        border-radius: 18px 18px 0 18px;
        margin-left: auto;
    }
    
    .assistant-message {
        background-color: #f0f0f0;
        border-radius: 18px 18px 18px 0;
        margin-right: auto;
    }
    
    /* Source container styling */
    .source-container {
        border-left: 3px solid #4a86e8;
        padding-left: 10px;
        margin-top: 10px;
        font-size: 0.9em;
        background-color: #f8f9fa;
    }
    
    /* Custom button styling */
    .custom-button {
        background-color: #4a86e8;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        cursor: pointer;
        text-align: center;
    }
    
    .custom-button:hover {
        background-color: #3a76d8;
    }
    </style>
    """, unsafe_allow_html=True)

def get_image_base64(image_path):
    """Convert an image to base64 encoding for embedding in HTML/CSS"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def render_header():
    """Render a custom header for the application"""
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <h1 style="margin: 0; flex-grow: 1;">RAG Chatbot</h1>
    </div>
    <p style="color: #666; margin-bottom: 2rem;">
        Ask questions about your documents and get answers powered by RAG technology.
    </p>
    """, unsafe_allow_html=True)

def format_sources(sources):
    """Format source documents for display in the UI"""
    if not sources:
        return ""
    
    html = '<div class="source-container">'
    for i, source in enumerate(sources):
        html += f'<p><strong>Source {i+1}:</strong> {source["content"][:200]}...</p>'
        if i < len(sources) - 1:
            html += '<hr style="margin: 0.5rem 0;">'
    html += '</div>'
    
    return html
