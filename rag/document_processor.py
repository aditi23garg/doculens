"""Ingestion pipeline: validate -> extract -> structure -> semantic chunks -> embed -> index."""
import hashlib
import io
import re
from pathlib import Path
from typing import Callable, Optional

from config.settings import get_settings
from database import mongo_client as db
from database import vector_store
from rag.embeddings import embed_documents
from rag.outline import OVERVIEW_SECTION, build_overview
from rag.semantic_chunker import Chunk, semantic_chunks
from rag.structure import parse_sections

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")
CHUNKING_VERSION = "semantic-v2"


class DocumentError(Exception):
    """A problem with an uploaded file, worded for the end user."""


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Extraction ────────────────────────────────────────────────────────────────
def _extract_pdf(data: bytes) -> list[tuple[Optional[int], str]]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentError("This PDF is password-protected. Remove the password and upload it again.")
        return [(i, _clean(page.extract_text() or "")) for i, page in enumerate(reader.pages, start=1)]
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"This PDF couldn't be read ({exc}).")


def _heading_level(para, text: str) -> int:
    name = (para.style.name if para.style is not None else "") or ""
    if name == "Title":
        return 1
    m = re.match(r"Heading\s*(\d)", name)
    if m:
        return min(int(m.group(1)), 6)
    # Fully bold, short, sentence-less line that isn't a list item -> treat as a heading.
    if len(text) <= 70 and not text.endswith((".", ",", ";", ":")) and not name.startswith("List"):
        runs = [r for r in para.runs if r.text.strip()]
        if runs and all(r.bold for r in runs):
            return 2
    return 0


def _extract_docx(data: bytes) -> list[tuple[Optional[int], str]]:
    """Body text in document order, with headings emitted as '#'-markers."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentError(f"This Word file couldn't be read ({exc}).")

    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            if not text:
                continue
            level = _heading_level(para, text)
            if level:
                blocks.append(f"{'#' * level} {text}")
            elif ((para.style.name if para.style is not None else "") or "").startswith("List"):
                blocks.append(f"- {text}")
            else:
                blocks.append(text)
        elif tag == "tbl":
            rows = []
            for row in Table(child, document).rows:
                cells, seen = [], set()
                for cell in row.cells:
                    if id(cell._tc) in seen:  # merged cells repeat
                        continue
                    seen.add(id(cell._tc))
                    value = cell.text.strip().replace("\n", " ")
                    if value:
                        cells.append(value)
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append("\n".join(rows))
    return [(None, _clean("\n\n".join(blocks)))]


def _extract_text(data: bytes) -> list[tuple[Optional[int], str]]:
    # utf-16 only when a BOM says so: without one it "decodes" almost any bytes as garbage.
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates = ("utf-16", "utf-8-sig", "cp1252", "latin-1")
    else:
        candidates = ("utf-8-sig", "cp1252", "latin-1")
    for encoding in candidates:
        try:
            return [(None, _clean(data.decode(encoding)))]
        except UnicodeDecodeError:
            continue
    raise DocumentError("This text file uses an unsupported encoding.")


def extract_pages(ext: str, data: bytes) -> list[tuple[Optional[int], str]]:
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext == ".docx":
        return _extract_docx(data)
    return _extract_text(data)


# ── Chunking ──────────────────────────────────────────────────────────────────
def build_chunks(ext: str, pages: list[tuple[Optional[int], str]], title: str = "") -> list[Chunk]:
    """Content chunks, preceded by one 'Document overview' chunk (outline + opening text)."""
    s = get_settings()
    sections = parse_sections(pages, heuristic_headings=ext in (".pdf", ".txt"))
    content = semantic_chunks(
        sections,
        # SEMANTIC_SIMILARITY embeddings are used only to find topic shifts inside long sections.
        lambda texts: embed_documents(texts, task_type="SEMANTIC_SIMILARITY"),
        max_chars=s.chunk_max_chars,
        min_chars=s.chunk_min_chars,
        percentile=s.semantic_breakpoint_percentile,
    )
    if not content:
        return []
    overview = build_overview(title, sections)
    return ([overview] if overview else []) + content


def _embedding_text(title: str, chunk: Chunk) -> str:
    """What gets embedded: body plus document title and section path, so a chunk that
    says 'accuracy: 94%' still knows which model/section it belongs to."""
    header = f"{title} | {chunk.section}" if chunk.section else title
    return f"{header}\n\n{chunk.text}"


def remove_document(user_id: str, doc_id: str) -> None:
    vector_store.delete_document_chunks(user_id, doc_id)
    db.delete_document(user_id, doc_id)


def ingest_document(
    user_id: str,
    filename: str,
    data: bytes,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    s = get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentError(f"{filename}: unsupported type. Upload PDF, DOCX, TXT or MD files.")
    if not data:
        raise DocumentError(f"{filename}: the file is empty.")
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise DocumentError(f"{filename}: larger than the {s.max_upload_mb} MB limit.")

    digest = hashlib.sha256(data).hexdigest()
    existing = db.get_document_by_hash(user_id, digest)
    if existing:
        if existing.get("status") == "ready":
            raise DocumentError(f"{filename}: already in your library as “{existing['filename']}”.")
        remove_document(user_id, existing["id"])  # clear an interrupted earlier attempt

    try:
        pages = extract_pages(ext, data)
    except DocumentError as exc:
        raise DocumentError(f"{filename}: {exc}")

    title = Path(filename).stem.replace("_", " ").replace("-", " ")
    chunks = build_chunks(ext, pages, title)
    if not chunks:
        raise DocumentError(
            f"{filename}: no readable text found. Scanned or image-only files need OCR before upload."
        )

    doc_id = db.create_document(user_id, filename, ext.lstrip("."), len(data), digest)
    try:
        vectors = embed_documents([_embedding_text(title, c) for c in chunks], on_progress)
        if len(vectors) != len(chunks):
            # embed_documents/_embed (rag/embeddings.py) already guards against this, but
            # never let a length mismatch reach zip() below: zip() silently truncates to the
            # shorter list rather than erroring, which is exactly how this used to fail
            # silently - one vector landing on chunks[0] (always the overview chunk), every
            # other chunk simply never stored, and no exception anywhere to catch it.
            raise RuntimeError(f"got {len(vectors)} vectors for {len(chunks)} chunks")
        vector_store.upsert_chunks(
            [
                {
                    "vector": vec,
                    "payload": {
                        "user_id": user_id,
                        "doc_id": doc_id,
                        "filename": filename,
                        "section": chunk.section,
                        "page": chunk.page,
                        "page_end": chunk.page_end,
                        "chunk_index": idx,
                        "chunking": CHUNKING_VERSION,
                        "kind": "overview" if chunk.section == OVERVIEW_SECTION else "content",
                        "text": chunk.text,
                    },
                }
                for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
            ]
        )
        page_count = len(pages) if ext == ".pdf" else None
        db.update_document(user_id, doc_id, status="ready", chunk_count=len(chunks), page_count=page_count)
    except Exception as exc:
        remove_document(user_id, doc_id)
        raise DocumentError(f"{filename}: indexing failed ({exc}).")
    return {"id": doc_id, "filename": filename, "chunk_count": len(chunks), "page_count": page_count}
