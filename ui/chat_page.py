"""The Ask page: questions in, document-grounded answers out."""
import html
import re

import streamlit as st

from chat.chat_engine import ask
from config.settings import get_settings
from database.mongo_client import add_message, create_conversation, list_documents
from rag.citations import CITATION_RE


def _chips(match: re.Match) -> str:
    return "".join(f'<span class="cite">{n}</span>' for n in re.findall(r"\d+", match.group(1)))


def _decorate(text: str) -> str:
    """Escape model/document text, then turn [1], 【1】 or [1, 2] into highlighter-style chips."""
    return CITATION_RE.sub(_chips, html.escape(text, quote=False))


def _where(s: dict) -> str:
    parts = []
    if s.get("section"):
        parts.append(html.escape(s["section"]))
    if s.get("page"):
        end = s.get("page_end")
        parts.append(f"pages {s['page']}–{end}" if end and end != s["page"] else f"page {s['page']}")
    return " · ".join(parts)


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            where = _where(s)
            where = f" · {where}" if where else ""
            st.markdown(
                f'<div class="src-head"><span class="cite">{s["n"]}</span>{html.escape(s["filename"])}{where}'
                f'<span class="src-score">match {s.get("score", 0):.2f}</span></div>'
                f'<div class="excerpt">{html.escape(s["text"])}</div>',
                unsafe_allow_html=True,
            )


def _render_debug(debug: dict) -> None:
    """Why the answer looks the way it does: queries tried, gate decision, ranked passages."""
    if not debug:
        return
    verdict = {
        "sent": "Passages were sent to the model.",
        "refused_by_gate": "Refused before calling the model: the question didn't match your documents closely enough.",
        "refused_by_model": "Passages were sent, but the model found no answer in them.",
    }.get(debug.get("verdict", ""), "")
    with st.expander("Search details"):
        if verdict:
            st.caption(verdict)
        st.caption("Searched for: " + "  |  ".join(debug.get("queries", [])))
        st.caption("Relevance check: " + debug.get("gate", ""))
        if debug.get("hits"):
            st.table(debug["hits"])
        else:
            st.caption("No passages were returned by the search.")


def _render_answer(text: str, grounded, sources: list[dict]) -> None:
    if grounded is False:
        st.markdown(f'<div class="note">{html.escape(text)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(_decorate(text), unsafe_allow_html=True)
        _render_sources(sources)


def _go_to_documents() -> None:
    st.session_state.page = "Documents"


def _empty_state() -> None:
    st.markdown(
        '<div class="empty"><h3>Add a document to begin</h3>'
        '<div class="muted">DocuLens answers only from files you upload, so there is nothing to ask yet. '
        "PDF, Word, text and Markdown files work.</div></div>",
        unsafe_allow_html=True,
    )
    st.button("Upload documents", on_click=_go_to_documents, type="primary")


def show_chat_page() -> None:
    user = st.session_state.user
    docs = [d for d in list_documents(user["id"]) if d.get("status") == "ready"]

    st.markdown("## Ask your documents")
    if not docs:
        _empty_state()
        return

    names = {d["id"]: d["filename"] for d in docs}
    st.markdown(
        f'<div class="ask-caption">Answers come only from your {len(docs)} uploaded '
        f'document{"s" if len(docs) != 1 else ""}.</div>',
        unsafe_allow_html=True,
    )
    chosen = st.multiselect(
        "Search only in", options=list(names), format_func=names.get, placeholder="All documents"
    )
    doc_ids = chosen or list(names)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.text(msg["content"])
            else:
                _render_answer(msg["content"], msg.get("grounded"), msg.get("sources", []))
                if get_settings().show_search_details:
                    _render_debug(msg.get("debug"))

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    history = list(st.session_state.messages[-6:])
    new_conversation = st.session_state.current_conversation_id is None
    if new_conversation:
        title = prompt.strip().replace("\n", " ")[:60] or "Conversation"
        st.session_state.current_conversation_id = create_conversation(user["id"], title)
    conversation_id = st.session_state.current_conversation_id

    with st.chat_message("user"):
        st.text(prompt)
    add_message(user["id"], conversation_id, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            stream, result = ask(user["id"], prompt, doc_ids, history)
            acc = ""
            for token in stream:
                acc += token
                placeholder.markdown(_decorate(acc) + " ▌", unsafe_allow_html=True)
            with placeholder.container():
                _render_answer(result.text, result.grounded, result.sources)
                if get_settings().show_search_details:
                    _render_debug(result.debug)
        except Exception as exc:
            placeholder.error(f"The search failed: {exc}. Try again in a moment.")
            return

    add_message(user["id"], conversation_id, "assistant", result.text, result.sources, result.grounded, result.debug)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.text,
            "sources": result.sources,
            "grounded": result.grounded,
            "debug": result.debug,
        }
    )
    if new_conversation:
        st.rerun()  # refresh the sidebar list
