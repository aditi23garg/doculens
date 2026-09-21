"""Run the current on-disk pipeline against a real file, using your real .env, with nothing
cached and nothing running in the background. Bypasses Streamlit and Qdrant completely - if
this prints the right chunk count, sections, and a matching embedding count, the code and
your embedding setup are both fine and whatever server process handled your upload is
running something older.

    python -m scripts.check_chunking path\\to\\OCR_Model_Research_CIG.docx
"""
import sys
from pathlib import Path

from config.settings import get_settings
from rag.document_processor import CHUNKING_VERSION, _embedding_text, build_chunks, extract_pages
from rag.embeddings import embed_documents


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.check_chunking path\\to\\file.docx")
    path = Path(sys.argv[1])
    data = path.read_bytes()
    ext = path.suffix.lower()

    print(f"document_processor.py loaded from: {build_chunks.__module__}")
    print(f"CHUNKING_VERSION on disk right now: {CHUNKING_VERSION}")
    s = get_settings()
    print(f"embedding_model={s.embedding_model}  embedding_dim={s.embedding_dim}")
    print(f"chunk_max_chars={s.chunk_max_chars}  chunk_min_chars={s.chunk_min_chars}  "
          f"semantic_breakpoint_percentile={s.semantic_breakpoint_percentile}\n")

    pages = extract_pages(ext, data)
    print(f"extracted {sum(len(t) for _, t in pages)} characters of text")

    title = path.stem.replace("_", " ").replace("-", " ")
    chunks = build_chunks(ext, pages, title)
    print(f"\nbuild_chunks() produced {len(chunks)} chunk(s):\n")
    for i, c in enumerate(chunks):
        head = " ".join(c.text.split())[:70]
        print(f"  #{i:>3} {len(c.text):>5} chars  [{c.section or '-'}]  {head}")

    if len(chunks) <= 1:
        print("\n! Only the overview (or nothing) came back. This is the code on disk itself,")
        print("  not a stale server process - something in your real environment behaves")
        print("  differently from a clean one. Check installed package versions next:")
        print("  pip show python-docx qdrant-client google-genai")
        return

    print(f"\n{len(chunks)} chunks look right. Now calling the real embedding API for all of "
          f"them (this is exactly what ingest_document() does, and what actually gets stored) ...")
    try:
        vectors = embed_documents([_embedding_text(title, c) for c in chunks])
    except Exception as exc:
        print(f"\n! The embedding call itself failed/raised: {exc}")
        print("  That's the good outcome here: ingest_document() would reject the upload")
        print("  cleanly instead of silently storing a partial document.")
        return

    print(f"got {len(vectors)} vector(s) back for {len(chunks)} chunk(s) sent")
    if len(vectors) != len(chunks):
        print(f"\n! MISMATCH: the embedding API returned {len(vectors)} vectors for "
              f"{len(chunks)} inputs. This is the bug - if you're on an older embeddings.py")
        print(f"  without the length-check fix, this pairs by position and silently stores")
        print(f"  only the first {len(vectors)} chunk(s), with no error anywhere. Check "
              f"EMBEDDING_MODEL in your .env: gemini-embedding-2 and gemini-embedding-2-preview")
        print(f"  are known to do this (googleapis/python-genai#2523) - switch to "
              f"gemini-embedding-001.")
    else:
        print(f"\nEverything checks out: {len(chunks)} chunks in, {len(vectors)} vectors out.")
        print("If your app still doesn't store all of them, the running `streamlit run app.py`")
        print("process is stale: fully close every terminal/window running it (check Task")
        print("Manager for lingering python.exe processes too), then start it fresh and re-upload.")


if __name__ == "__main__":
    main()
