import streamlit as st
from utils import get_config_names, query_rag

def chatbot_section(selected_config):
    st.header("🤖 RAG Chatbot")
    st.markdown("""
    <style>
    .chat-message { padding: 0.5em; border-radius: 8px; margin-bottom: 0.5em; }
    .user { background: #e6f7ff; }
    .bot { background: #f6ffed; }
    </style>
    """, unsafe_allow_html=True)

    if not selected_config:
        st.warning("No configurations available. Please add one in the Configuration section.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Your message", key="chat_input")
    send_btn = st.button("Send", key="send_btn")

    if send_btn and user_input:
        # For now, both modes use the same endpoint; can be extended for /retrieve, etc.
        response = query_rag(user_input, selected_config)
        answer = response.get("answer") or response.get("result") or str(response)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))
        st.session_state.chat_input = ""  # clear input

    for role, msg in st.session_state.chat_history[-10:]:
        css_class = "user" if role == "user" else "bot"
        st.markdown(f'<div class="chat-message {css_class}"><b>{role.title()}:</b> {msg}</div>', unsafe_allow_html=True)
