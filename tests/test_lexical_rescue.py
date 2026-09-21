"""Regression tests for the "specific term" bug, in two parts:

1. rag/gate.py: a short, direct question ("what is paddleocr") can legitimately score far
   below MIN_RELEVANCE_SCORE on raw cosine similarity, even when the document literally
   contains the term - a two-word query carries little embedding signal on its own, and the
   gate's existing borderline rescue only looks at the text of the top *semantic* hits, so a
   section that didn't rank in the semantic top-k is invisible to it. evaluate_gate's new
   term_hits parameter (a literal full-text match, independent of embedding score) closes
   that gap.
2. rag/retriever.py: once the gate has passed, that same literal match must actually be sent
   to the LLM, not just used to unblock the gate (_lexical_rescue).

The fake search_terms below filters by real substring containment of the terms it's given,
unlike a fake that just returns a fixed list regardless of the query - that would hide
exactly this bug (an off-topic question "passing" only because the fake ignored what terms
it was actually asked to match).
"""
import sys
import types

sys.modules.setdefault("config.settings", types.ModuleType("config.settings"))
sys.modules.setdefault("database", types.ModuleType("database"))
sys.modules.setdefault("database.vector_store", types.ModuleType("database.vector_store"))
sys.modules.setdefault("rag.embeddings", types.ModuleType("rag.embeddings"))
sys.modules["database"].vector_store = sys.modules["database.vector_store"]

from types import SimpleNamespace  # noqa: E402

_OVERVIEW = {
    "doc_id": "doc1", "chunk_index": 0, "section": "Document overview",
    "filename": "research.docx", "text": "headings: PaddleOCR-VL, GLM-OCR, Qwen2.5-VL",
}
_TAGLINE = {
    "doc_id": "doc1", "chunk_index": 1, "section": "Detailed Research > 1. PaddleOCR-VL 1.6",
    "filename": "research.docx", "text": "PaddleOCR-VL is a small, dedicated OCR model from Baidu for scanned documents.",
}
_ACCURACY_CHUNK = {
    "doc_id": "doc1", "chunk_index": 2, "section": "Detailed Research > 1. PaddleOCR-VL 1.6",
    "filename": "research.docx", "text": "Handwriting-text accuracy for PaddleOCR-VL is 73% on OmniDocBench v1.6.",
}
# The full text of every stored chunk, for the fake full-text index below.
_TEXT_INDEX = [_OVERVIEW, _TAGLINE, _ACCURACY_CHUNK]


def _install_fakes(search_hits, term_scores=None):
    """term_scores: {chunk_index: score} for chunks that should be "findable" by a real
    literal-term search, i.e. only ones that would genuinely be returned by Qdrant's
    full-text index for whatever terms the caller actually passes in.
    """
    term_scores = term_scores or {}
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
        return []

    def fake_search_terms(user_id, vec, doc_ids, terms, top_k):
        if not terms:
            return []
        out = []
        for payload in _TEXT_INDEX:
            if payload["chunk_index"] not in term_scores:
                continue
            text_l = payload["text"].lower()
            if any(t in text_l for t in terms):
                out.append({"id": f"pt{payload['chunk_index']}", "score": term_scores[payload["chunk_index"]], "payload": payload})
        return out[:top_k]

    sys.modules["database.vector_store"].search = fake_search
    sys.modules["database.vector_store"].list_chunks = fake_list_chunks
    sys.modules["database.vector_store"].search_terms = fake_search_terms
    sys.modules["rag.embeddings"].embed_queries = lambda texts: [[1.0] for _ in texts]

    import importlib
    import rag.retriever as retriever_mod

    importlib.reload(retriever_mod)
    return retriever_mod


def _hit(payload, score):
    return {"id": f"pt{payload['chunk_index']}", "score": score, "payload": payload}


def test_a_direct_short_question_that_scores_far_below_the_gate_still_gets_answered():
    # Nothing clears the gate on embedding score alone: the best semantic hit is 0.22, well
    # under min_relevance_score (0.40) and even under the 0.10 borderline band (0.30). This
    # is the exact "what is paddleocr" case: a short, direct question with weak embedding
    # signal against a document that genuinely contains the term.
    hits = [_hit(_TAGLINE, 0.22)]
    retriever = _install_fakes(hits, term_scores={1: 0.35, 2: 0.28})
    result = retriever.retrieve("u1", ["what is paddleocr"], ["doc1"])
    assert result.chunks, "a literal term match in the docs must not be refused"
    texts_sent = " ".join(c.text for c in result.chunks)
    assert "small, dedicated OCR model" in texts_sent


def test_rescued_gate_pass_still_delivers_the_chunk_holding_the_actual_number():
    # Same as above, but check that the *answer-bearing* chunk (the one with the accuracy
    # number, not just the tagline) makes it into what's sent to the LLM.
    hits = [_hit(_TAGLINE, 0.22)]
    retriever = _install_fakes(hits, term_scores={1: 0.35, 2: 0.28})
    result = retriever.retrieve("u1", ["what is paddleocr"], ["doc1"])
    texts_sent = " ".join(c.text for c in result.chunks)
    assert "73%" in texts_sent


def test_gate_is_not_fooled_by_an_unrelated_document_that_happens_to_score_low():
    # A genuinely off-topic question still gets refused. No stored chunk's text contains
    # "capital" or "france", so the fake's real substring filter (not a stub that ignores
    # the terms it's given) correctly returns nothing, and there's no rescue.
    hits = [_hit(_TAGLINE, 0.22)]
    retriever = _install_fakes(hits, term_scores={1: 0.35, 2: 0.28})
    result = retriever.retrieve("u1", ["what is the capital of france"], ["doc1"])
    assert result.chunks == []


def test_ordinary_high_scoring_questions_are_unaffected():
    hits = [_hit(_TAGLINE, 0.55)]
    retriever = _install_fakes(hits)
    result = retriever.retrieve("u1", ["what is paddleocr"], ["doc1"])
    assert result.chunks and result.chunks[0].text == _TAGLINE["text"]


def test_rescue_never_pulls_in_the_overview_chunk():
    hits = [_hit(_TAGLINE, 0.55)]
    retriever = _install_fakes(hits, term_scores={0: 0.20})
    result = retriever.retrieve("u1", ["what is paddleocr about"], ["doc1"])
    assert not any(c.section == "Document overview" for c in result.chunks)


def test_rescue_does_not_duplicate_a_chunk_already_kept():
    hits = [_hit(_TAGLINE, 0.55)]
    retriever = _install_fakes(hits, term_scores={1: 0.55})
    result = retriever.retrieve("u1", ["what is paddleocr"], ["doc1"])
    ids = [c.text for c in result.chunks]
    assert len(ids) == len(set(ids))


def test_rescue_never_fires_when_the_question_has_no_distinctive_terms():
    hits = [_hit(_TAGLINE, 0.55)]
    retriever = _install_fakes(hits, term_scores={2: 0.30})
    result = retriever.retrieve("u1", ["what is it about"], ["doc1"])
    assert not any("73%" in c.text for c in result.chunks)
