"""Regression test for the split-section bug: a rich section (e.g. one model's full
write-up) gets divided into several chunks by rag/semantic_chunker.py when it exceeds
chunk_max_chars. Retrieval used to keep only whichever of those chunks scored within
relevance_margin of the best hit, so a short leading chunk (a tagline) could win while the
chunk holding the actual benchmark numbers was silently dropped. rag/retriever.py's
_complete_sections should pull the rest of the section back in once any part of it is kept.
"""
import sys
import types

# Stand in for the modules retriever.py imports that need real credentials/services.
sys.modules.setdefault("config.settings", types.ModuleType("config.settings"))
sys.modules.setdefault("database", types.ModuleType("database"))
sys.modules.setdefault("database.vector_store", types.ModuleType("database.vector_store"))
sys.modules.setdefault("rag.embeddings", types.ModuleType("rag.embeddings"))
sys.modules["database"].vector_store = sys.modules["database.vector_store"]

from types import SimpleNamespace  # noqa: E402

_ALL_CHUNKS = {
    "doc1": [
        {"doc_id": "doc1", "chunk_index": 0, "section": "Document overview",
         "filename": "OCR_Model_Research_CIG.docx", "text": "headings..."},
        {"doc_id": "doc1", "chunk_index": 2, "section": "Detailed Research > 1. PaddleOCR-VL 1.6",
         "filename": "OCR_Model_Research_CIG.docx",
         "text": "Small, dedicated OCR model. 1. What is the model? A small OCR model from Baidu."},
        {"doc_id": "doc1", "chunk_index": 3, "section": "Detailed Research > 1. PaddleOCR-VL 1.6",
         "filename": "OCR_Model_Research_CIG.docx",
         "text": "5. Printed-text accuracy | 96.34% on OmniDocBench v1.6. 6. Handwriting accuracy | 73%."},
        {"doc_id": "doc1", "chunk_index": 4, "section": "Detailed Research > 2. GLM-OCR",
         "filename": "OCR_Model_Research_CIG.docx", "text": "GLM-OCR details, unrelated section."},
    ]
}


def _install_fakes(search_hits, term_hits=()):
    settings = SimpleNamespace(
        retrieval_top_k=8,
        min_relevance_score=0.40,
        relevance_margin=0.18,
        max_context_chunks=10,
        section_complete_extra_chunks=6,
        lexical_rescue_extra_chunks=3,
    )
    sys.modules["config.settings"].get_settings = lambda: settings

    def fake_search(user_id, vec, doc_ids, top_k):
        return search_hits

    def fake_list_chunks(user_id, doc_id):
        return list(_ALL_CHUNKS.get(doc_id, []))

    def fake_search_terms(user_id, vec, doc_ids, terms, top_k):
        return list(term_hits)

    sys.modules["database.vector_store"].search = fake_search
    sys.modules["database.vector_store"].list_chunks = fake_list_chunks
    sys.modules["database.vector_store"].search_terms = fake_search_terms
    sys.modules["rag.embeddings"].embed_queries = lambda texts: [[1.0] for _ in texts]

    import importlib
    import rag.retriever as retriever_mod

    importlib.reload(retriever_mod)
    return retriever_mod


def _hit(chunk_index, score):
    payload = next(c for c in _ALL_CHUNKS["doc1"][:] if c["chunk_index"] == chunk_index)
    return {"id": f"pt{chunk_index}", "score": score, "payload": payload}


def test_only_the_leading_chunk_of_a_section_clears_the_margin_but_both_are_sent():
    # chunk 2 (tagline, no numbers) scores 0.50; chunk 3 (the actual benchmark numbers)
    # scores only 0.28 for this query - well outside the 0.18 margin on its own.
    hits = [_hit(2, 0.50)]
    retriever = _install_fakes(hits)
    result = retriever.retrieve("u1", ["what is paddleocr"], ["doc1"])
    sections_sent = {c.section for c in result.chunks}
    texts_sent = " ".join(c.text for c in result.chunks)
    assert "Detailed Research > 1. PaddleOCR-VL 1.6" in sections_sent
    assert "96.34%" in texts_sent, "the sibling chunk with the real answer must be included"


def test_completion_never_pulls_in_an_unrelated_section():
    hits = [_hit(2, 0.50)]
    retriever = _install_fakes(hits)
    result = retriever.retrieve("u1", ["what is paddleocr"], ["doc1"])
    assert not any("GLM-OCR" in c.section for c in result.chunks)


def test_completion_does_not_fire_off_the_overview_chunk():
    overview = _ALL_CHUNKS["doc1"][0]
    hits = [{"id": "pt0", "score": 0.50, "payload": overview}]
    retriever = _install_fakes(hits)
    result = retriever.retrieve("u1", ["what models are discussed"], ["doc1"])
    assert len(result.chunks) == 1  # no siblings for "Document overview" itself


def test_gate_is_unaffected_by_completion():
    # A genuinely off-topic query still gets refused; completion only ever runs after
    # the gate has already passed.
    hits = [_hit(2, 0.10)]
    retriever = _install_fakes(hits)
    result = retriever.retrieve("u1", ["what is the capital of france"], ["doc1"])
    assert result.chunks == []
