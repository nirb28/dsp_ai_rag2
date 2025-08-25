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
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # Clear input safely before rendering the widget
    if st.session_state.get("clear_chat_input"):
        st.session_state.chat_input = ""
        st.session_state.clear_chat_input = False

    user_input = st.text_input("Your message", key="chat_input")

    # Advanced options
    with st.expander("Advanced Parameters", expanded=False):
        similarity_threshold = st.number_input("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="chatbot_similarity_threshold")
        max_sources = st.number_input("Max Sources", min_value=1, value=5, step=1, key="chatbot_max_sources")
        max_tokens = st.number_input("Max Tokens", min_value=1, value=200, step=1, key="chatbot_max_tokens")
        st.markdown("**Generation Parameters**")
        temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.2, step=0.01, key="chatbot_temperature")
        top_p = st.number_input("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.01, key="chatbot_top_p")
        top_k = st.number_input("Top K", min_value=1, value=40, step=1, key="chatbot_top_k")
        st.markdown("**Config**")
        system_prompt = st.text_area("System Prompt", value="", key="chatbot_system_prompt")

    col_send, col_clear = st.columns([1, 1])
    with col_send:
        send = st.button("Send")
    with col_clear:
        clear = st.button("Clear Chat", key="clear_chat_btn")

    if send and user_input:
        payload = {
            "query": user_input,
            "similarity_threshold": similarity_threshold,
            "max_sources": max_sources,
            "max_tokens": max_tokens,
            "generation_parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            },
            "config": {
                "system_prompt": system_prompt
            },
            "include_metadata": True,
            "generate": True
        }
        if selected_config:
            payload["configuration_name"] = selected_config
        from utils import query_rag
        response = query_rag(payload)
        answer = response.get("answer") or response.get("result") or str(response)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))
        st.session_state.raw_json = response
        st.session_state.clear_chat_input = True  # set flag to clear on next run
        st.experimental_rerun()

    if clear:
        st.session_state.chat_history = []
        st.session_state.raw_json = None
        st.experimental_rerun()

    for role, msg in st.session_state.chat_history[-10:]:
        css_class = "user" if role == "user" else "bot"
        st.markdown(f'<div class="chat-message {css_class}"><b>{role.title()}:</b> {msg}</div>', unsafe_allow_html=True)

    # Show raw JSON in collapsible section
    if st.session_state.get("raw_json"):
        with st.expander("Show Raw JSON Response"):
            st.json(st.session_state.raw_json)
