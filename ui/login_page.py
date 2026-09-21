"""Sign in / create account."""
import streamlit as st

from auth.auth_manager import login, register
from config.settings import APP_NAME


def _start_session(user: dict) -> None:
    st.session_state.user = user
    st.session_state.current_conversation_id = None
    st.session_state.messages = []
    st.session_state.page = "Ask"
    st.rerun()


def show_login_page() -> None:
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        top_l, top_r = st.columns([3, 1])
        top_r.toggle("Dark", key="dark_mode")
        st.markdown(
            f'<div class="login-title">{APP_NAME}</div>'
            '<div class="login-sub">Upload your documents, then ask questions. '
            "Every answer comes from your files and shows where it was found.</div>",
            unsafe_allow_html=True,
        )
        sign_in, create = st.tabs(["Sign in", "Create account"])

        with sign_in:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                user, error = login(email, password)
                if error:
                    st.error(error)
                else:
                    _start_session(user)

        with create:
            with st.form("register_form"):
                name = st.text_input("Name")
                email = st.text_input("Email", key="reg_email")
                password = st.text_input("Password (8+ characters)", type="password", key="reg_pw")
                submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if submitted:
                user, error = register(name, email, password)
                if error:
                    st.error(error)
                else:
                    _start_session(user)
