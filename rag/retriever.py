"""Semantic search over the current user's chunks.

Two separate decisions:
  * GATE   - does the user's own question relate to the documents at all?
             (see rag/gate.py; only the user's question counts, never the alternate phrasings)
  * WIDEN  - once it does, keep every passage close to the best score, across all query
             phrasings, so broad questions collect all their related sections.
"""
from dataclasses import dataclass, field
from typing import Optional

from config.settings import get_settings
from database import vector_store
from rag.embeddings import embed_queries
from rag.gate import evaluate_gate
from rag.lexical import key_terms
from rag.outline import OVERVIEW_SECTION


@dataclass
class RetrievedChunk:
    n: int  # 1-based citation number
    text: str
    filename: str
    page: Optional[int]
    score: float
    doc_id: str
    section: str = ""
    page_end: Optional[int] = None


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    top_score: float = 0.0  # best score for the user's own question
    queries: list[str] = field(default_factory=list)
    gate: str = ""  # human-readable gate decision
    debug_hits: list[dict] = field(default_factory=list)  # ranked hits with scores, for "Search details"


def _debug_row(hit: dict, kept: bool) -> dict:
    p = hit["payload"]
    return {
        "score": round(float(hit["score"]), 3),
        "used": "yes" if kept else "no",
        "file": p.get("filename", ""),
        "section": p.get("section", "") or "(none)",
    }


def _complete_sections(user_id: str, doc_ids: Optional[list[str]], kept: list[dict]) -> list[dict]:
    """Pull in the rest of a section once part of it has already been kept.

    A rich section (e.g. one model's full write-up) is often split into several chunks
    because it exceeds chunk_max_chars (see rag/semantic_chunker.py). Each split chunk is
    embedded and scored independently, so it's common for only the *leading* chunk of a
    section (a short tagline) to clear the relevance margin while a later chunk holding the
    actual detail (benchmark numbers, specs) scores lower and gets left out. Once any chunk
    of a section is deemed relevant, treat the whole section as relevant and include the
    rest of it too, instead of silently handing the model half a topic.

    This never affects the off-topic gate (rag/gate.py) - it only widens what gets sent to
    the LLM after the gate has already passed on the user's own question.
    """
    have = {(h["payload"].get("doc_id", ""), h["payload"].get("chunk_index")) for h in kept}
    wanted = {
        (h["payload"].get("doc_id", ""), h["payload"].get("section", ""))
        for h in kept
        if h["payload"].get("section") and h["payload"].get("section") != OVERVIEW_SECTION
    }
    if not wanted:
        return []
    # Reuse the anchor chunk's own score for "Search details" rather than fabricate one.
    anchor_score = {
        (h["payload"].get("doc_id", ""), h["payload"].get("section", "")): h["score"] for h in kept
    }
    extra: list[dict] = []
    for doc_id in {d for d, _ in wanted}:
        if doc_ids and doc_id not in doc_ids:
            continue
        for payload in vector_store.list_chunks(user_id, doc_id):
            key = (doc_id, payload.get("section", ""))
            if key not in wanted:
                continue
            idx_key = (doc_id, payload.get("chunk_index"))
            if idx_key in have:
                continue
            have.add(idx_key)
            extra.append(
                {
                    "id": f"section-complete:{doc_id}:{payload.get('chunk_index')}",
                    "score": anchor_score.get(key, 0.0),
                    "payload": payload,
                }
            )
    return extra


def _lexical_rescue(term_hits: list[dict], kept: list[dict], budget: int) -> list[dict]:
    """Add chunks that literally contain a distinctive term of the question but weren't
    already kept, up to `budget`. `term_hits` is the same full-text search result used to
    unblock the gate above (rag/gate.py) - computed once in retrieve() and reused here so a
    question that only just cleared the gate via a literal term match actually gets that
    chunk sent to the LLM too, not just a pass through the gate.
    """
    if budget <= 0 or not term_hits:
        return []
    have_ids = {h["id"] for h in kept}
    extra: list[dict] = []
    for h in term_hits:
        if h["id"] in have_ids or h["payload"].get("section") == OVERVIEW_SECTION:
            continue
        have_ids.add(h["id"])
        extra.append(h)
        if len(extra) >= budget:
            break
    return extra


def retrieve(user_id: str, queries: list[str], doc_ids: Optional[list[str]] = None) -> RetrievalResult:
    s = get_settings()
    queries = [q for q in queries if q.strip()]
    if not queries:
        return RetrievalResult()
    vectors = embed_queries(queries)

    best: dict[str, dict] = {}
    primary_hits: list[dict] = []
    for idx, vec in enumerate(vectors):
        hits = vector_store.search(user_id, vec, doc_ids, s.retrieval_top_k)
        if idx == 0:
            primary_hits = hits
        for h in hits:
            current = best.get(h["id"])
            if current is None or h["score"] > current["score"]:
                best[h["id"]] = h

    # Computed once, before the gate: a full-text search restricted to chunks that literally
    # contain a distinctive term of the user's own question. Used both to keep the gate from
    # refusing a question whose exact term is genuinely in the documents but scored low on
    # raw embedding similarity (see rag/gate.py), and, once the gate passes, to make sure
    # that same chunk is actually sent to the LLM (see _lexical_rescue above).
    question_terms = key_terms(queries[0])
    term_hits = vector_store.search_terms(user_id, vectors[0], doc_ids, question_terms, top_k=8) if question_terms else []
    for h in term_hits:  # fold into the normal candidate pool too, not just the forced rescue below
        current = best.get(h["id"])
        if current is None or h["score"] > current["score"]:
            best[h["id"]] = h

    passed, gate_reason = evaluate_gate(queries[0], primary_hits, s.min_relevance_score, term_hits)
    primary_top = float(primary_hits[0]["score"]) if primary_hits else 0.0
    ranked = sorted(best.values(), key=lambda h: h["score"], reverse=True)

    if not passed or not ranked:
        return RetrievalResult(
            top_score=primary_top,
            queries=queries,
            gate=gate_reason,
            debug_hits=[_debug_row(h, False) for h in ranked[:8]],
        )

    cutoff = max(ranked[0]["score"] - s.relevance_margin, s.min_relevance_score - 0.15)
    kept = [h for h in ranked if h["score"] >= cutoff][: s.max_context_chunks]
    completions = _complete_sections(user_id, doc_ids, kept)[: s.section_complete_extra_chunks]
    kept = kept + completions
    rescued = _lexical_rescue(term_hits, kept, s.lexical_rescue_extra_chunks)
    kept = kept + rescued
    kept_ids = {h["id"] for h in kept}
    debug_hits = [_debug_row(h, h["id"] in kept_ids) for h in ranked[:10]]
    debug_hits += [
        {**_debug_row(h, True), "used": "yes (completes a matched section)"} for h in completions
    ]
    debug_hits += [
        {**_debug_row(h, True), "used": "yes (contains a key term of your question)"} for h in rescued
    ]
    chunks = [
        RetrievedChunk(
            n=i,
            text=h["payload"].get("text", ""),
            filename=h["payload"].get("filename", "document"),
            page=h["payload"].get("page"),
            score=float(h["score"]),
            doc_id=h["payload"].get("doc_id", ""),
            section=h["payload"].get("section", "") or "",
            page_end=h["payload"].get("page_end"),
        )
        for i, h in enumerate(kept, start=1)
    ]
    return RetrievalResult(
        chunks=chunks,
        top_score=primary_top,
        queries=queries,
        gate=gate_reason,
        debug_hits=debug_hits,
    )
