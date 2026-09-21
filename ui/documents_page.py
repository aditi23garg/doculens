"""The Documents page: upload, index and manage the user's library."""
import streamlit as st

from config.settings import get_settings
from database.mongo_client import list_documents
from rag.document_processor import SUPPORTED_EXTENSIONS, DocumentError, ingest_document, remove_document


def _format_size(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


def _index_files(user_id: str, files) -> None:
    report = []
    with st.status("Indexing documents…", expanded=True) as status:
        for f in files:
            st.write(f"Reading {f.name}")
            bar = st.progress(0.0)
            try:
                info = ingest_document(
                    user_id, f.name, f.getvalue(), on_progress=lambda done, total: bar.progress(done / total)
                )
                report.append(("success", f"{f.name}: ready to query ({info['chunk_count']} passages)."))
            except DocumentError as exc:
                report.append(("error", str(exc)))
        status.update(label="Finished", state="complete", expanded=False)
    st.session_state.upload_report = report
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
    st.rerun()


def show_documents_page() -> None:
    user = st.session_state.user
    limit = get_settings().max_upload_mb

    st.markdown("## Your documents")
    st.caption(f"PDF, Word (.docx), text and Markdown files, up to {limit} MB each.")

    for kind, message in st.session_state.pop("upload_report", []):
        (st.success if kind == "success" else st.error)(message)

    files = st.file_uploader(
        "Add documents",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.get('uploader_key', 0)}",
    )
    if files and st.button("Index documents", type="primary"):
        _index_files(user["id"], files)

    st.divider()
    docs = list_documents(user["id"])
    if not docs:
        st.markdown(
            '<div class="empty"><h3>No documents yet</h3>'
            '<div class="muted">Files you add appear here. You can remove them at any time.</div></div>',
            unsafe_allow_html=True,
        )
        return

    for d in docs:
        info, action = st.columns([6, 1.4])
        ready = d.get("status") == "ready"
        pages = f"{d['page_count']} pages · " if d.get("page_count") else ""
        meta = (
            f"{d['file_type'].upper()} · {_format_size(d.get('size_bytes', 0))} · {pages}"
            f"{d.get('chunk_count', 0)} passages · added {d['uploaded_at']:%d %b %Y}"
            if ready
            else "Indexing did not finish. Remove it and upload again."
        )
        info.markdown(
            f'<div class="doc-name">{d["filename"]}</div><div class="doc-meta">{meta}</div>',
            unsafe_allow_html=True,
        )
        with action.popover("Remove"):
            st.write("Remove this document and everything indexed from it?")
            if st.button("Remove document", key=f"rm_{d['id']}", type="primary"):
                remove_document(user["id"], d["id"])
                st.rerun()
