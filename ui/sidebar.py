"""Sidebar: navigation, conversation history, account."""
import streamlit as st

from config.settings import APP_NAME
from database.mongo_client import (
    delete_conversation,
    get_messages,
    list_conversations,
)


def _new_chat() -> None:
    st.session_state.current_conversation_id = None
    st.session_state.messages = []
    st.session_state.page = "Ask"


def _open_conversation(conversation_id: str) -> None:
    user = st.session_state.user
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages = get_messages(user["id"], conversation_id)
    st.session_state.page = "Ask"


def _delete_conversation(conversation_id: str) -> None:
    delete_conversation(st.session_state.user["id"], conversation_id)
    if st.session_state.current_conversation_id == conversation_id:
        _new_chat()


def _sign_out() -> None:
    for key in ("user", "current_conversation_id", "messages", "page", "upload_report"):
        st.session_state.pop(key, None)


def show_sidebar() -> None:
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f'<div class="brand"><span class="brand-mark"></span>{APP_NAME}</div>', unsafe_allow_html=True)
        st.toggle("Dark mode", key="dark_mode")
        st.radio("Section", ["Ask", "Documents"], key="page", label_visibility="collapsed")
        st.button("New conversation", on_click=_new_chat, use_container_width=True, type="primary")

        conversations = list_conversations(user["id"])
        st.markdown('<div class="muted" style="margin-top:1rem">Conversations</div>', unsafe_allow_html=True)
        if not conversations:
            st.caption("Your questions will appear here.")
        for conv in conversations:
            is_current = conv["id"] == st.session_state.get("current_conversation_id")
            title = conv["title"] if len(conv["title"]) <= 30 else conv["title"][:28] + "…"
            open_col, del_col = st.columns([5, 1])
            open_col.button(
                ("● " if is_current else "") + title,
                key=f"open_{conv['id']}",
                on_click=_open_conversation,
                args=(conv["id"],),
                use_container_width=True,
            )
            del_col.button("✕", key=f"del_{conv['id']}", on_click=_delete_conversation,
                           args=(conv["id"],), help="Delete conversation")

        st.divider()
        st.markdown(f'<div class="muted">{user["name"]}<br>{user["email"]}</div>', unsafe_allow_html=True)
        st.button("Sign out", on_click=_sign_out, use_container_width=True)
