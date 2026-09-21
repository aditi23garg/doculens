"""Inspect what is stored for an account and how a question is searched.

    python -m scripts.diagnose --email you@example.com --chunks
    python -m scripts.diagnose --email you@example.com "What models were discussed?"

--chunks prints every stored chunk (section, size, first words) so you can check that the
document was split sensibly. With a question it prints the search queries, the relevance
gate decision and the ranked passages with scores.
"""
import argparse

from chat.chat_engine import _plan_queries
from config.settings import get_settings
from database import mongo_client as db
from database import vector_store
from rag.document_processor import CHUNKING_VERSION
from rag.retriever import retrieve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="question to test")
    ap.add_argument("--email", required=True)
    ap.add_argument("--chunks", action="store_true", help="dump all stored chunks")
    args = ap.parse_args()

    s = get_settings()
    missing = s.get_missing_keys()
    if missing:
        raise SystemExit(f"Missing settings: {', '.join(missing)}")
    user = db.get_user_by_email(args.email.strip().lower())
    if not user:
        raise SystemExit("No account with that email.")
    docs = [d for d in db.list_documents(user["id"]) if d.get("status") == "ready"]
    print(f"Account: {user['email']}  |  ready documents: {len(docs)}")
    print(f"Gate minimum: {s.min_relevance_score}  margin: {s.relevance_margin}  top_k: {s.retrieval_top_k}\n")

    for d in docs:
        chunks = vector_store.list_chunks(user["id"], d["id"])
        kinds = {c.get("chunking", "old-fixed-size") for c in chunks}
        print(f"* {d['filename']}: {len(chunks)} chunks, chunking={','.join(sorted(kinds))}")
        if kinds != {CHUNKING_VERSION}:
            print(f"  ! not indexed with the current chunker ({CHUNKING_VERSION}): remove and re-upload this file")
        elif not any(c.get("kind") != "overview" for c in chunks):
            # Tagged with the current chunker but holds nothing except the table-of-contents
            # chunk. The current build_chunks() can't produce this state on its own (it
            # returns nothing at all, and the upload fails outright, if body chunking comes
            # up empty) - this is leftover data from an older run. Same remedy either way.
            print("  ! only the overview chunk is stored, no section content: remove and re-upload this file")
        if args.chunks:
            for c in chunks:
                head = " ".join(c.get("text", "").split())[:70]
                print(f"    #{c.get('chunk_index'):>3} {len(c.get('text', '')):>5} chars  [{c.get('section') or '-'}]  {head}")
    print()

    if args.question:
        queries = _plan_queries(args.question, [])
        result = retrieve(user["id"], queries, [d["id"] for d in docs])
        print("Queries :", queries)
        print("Gate    :", result.gate)
        print("Verdict :", "PASS -> sent to LLM" if result.chunks else "REFUSED before LLM")
        print(f"\n{'score':>6}  used                                      section")
        for h in result.debug_hits:
            print(f"{h['score']:>6}  {h['used']:<40}  {h['file']} > {h['section']}")


if __name__ == "__main__":
    main()
