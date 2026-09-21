"""
DocuLens — Streamlit entry point.

A document question-answering app: users upload files and ask questions;
answers come only from those files (with citations), never from the model's
general knowledge.

Routing:
  Unauthenticated -> login / register
  Authenticated   -> sidebar + (Ask | Documents)
"""
import streamlit as st

from config.settings import APP_NAME, APP_TAGLINE, get_settings

# Page config must be the first Streamlit call.
st.set_page_config(
    page_title=APP_NAME,
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": f"**{APP_NAME}** — {APP_TAGLINE}\n\nAnswers are generated only from the documents you upload.",
    },
)

from database.mongo_client import ensure_indexes  # noqa: E402
from database.vector_store import ensure_collection  # noqa: E402
from ui.chat_page import show_chat_page  # noqa: E402
from ui.documents_page import show_documents_page  # noqa: E402
from ui.login_page import show_login_page  # noqa: E402
from ui.sidebar import show_sidebar  # noqa: E402
from ui.styles import get_css  # noqa: E402

st.session_state.setdefault("dark_mode", False)
st.markdown(get_css(dark=st.session_state.dark_mode), unsafe_allow_html=True)

# ── Configuration check ───────────────────────────────────────────────────────
settings = get_settings()
missing_keys = settings.get_missing_keys()
if missing_keys:
    st.title(APP_NAME)
    st.error("Configuration is incomplete. Add these values to your `.env` file:")
    for key in missing_keys:
        st.markdown(f"- `{key}` is missing or still a placeholder")
    st.info("Save `.env`, then reload this page. `.env.example` lists every setting.")
    st.stop()


# ── One-time connection setup ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting…")
def _init_backends():
    errors = []
    try:
        ensure_indexes()
    except Exception as exc:
        errors.append(
            f"**MongoDB Atlas:** {exc}\n\n"
            "Check that your IP is allowed under Network Access (or `0.0.0.0/0` for testing) "
            "and that the username and password in `MONGODB_URI` are correct and URL-encoded."
        )
    try:
        ensure_collection()
    except Exception as exc:
        errors.append(f"**Qdrant Cloud:** {exc}\n\nCheck `QDRANT_URL` and `QDRANT_API_KEY`.")
    return errors


backend_errors = _init_backends()
if backend_errors:
    st.title(APP_NAME)
    for message in backend_errors:
        st.error(message)
    st.stop()

# ── Session defaults ──────────────────────────────────────────────────────────
st.session_state.setdefault("user", None)
st.session_state.setdefault("current_conversation_id", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("page", "Ask")

# ── Routing ───────────────────────────────────────────────────────────────────
if st.session_state.user is None:
    show_login_page()
else:
    show_sidebar()
    if st.session_state.page == "Documents":
        show_documents_page()
    else:
        show_chat_page()
