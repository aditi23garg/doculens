"""The off-topic gate: decide whether a question plausibly relates to the documents."""
from rag.lexical import shares_keywords

BORDERLINE_BAND = 0.10


def evaluate_gate(
    question: str, hits: list[dict], min_score: float, term_hits: list[dict] | None = None
) -> tuple[bool, str]:
    """hits: search results for the user's own question, best first ({"score", "payload"}).

    term_hits: results of a full-text search restricted to chunks that literally contain a
    distinctive term of the question (database.vector_store.search_terms). A short, direct
    question ("what is paddleocr") can legitimately score far below min_score on raw cosine
    similarity - a two-word query carries little signal for embedding similarity even when
    the exact term it names is sitting right there in the document. Without this, the
    borderline rescue below is not enough: it only looks at the text of the top semantic
    hits, so a document that literally contains the term is invisible to it whenever that
    chunk didn't happen to rank in the top few results of the *semantic* search. A literal
    term match inside the user's own uploaded documents is strong evidence on its own, so it
    passes the gate regardless of how low the embedding score is. This is safe to be lenient
    about: layers 4-5 (the strict prompt and the NOT_FOUND sentinel) are the real backstop,
    so the worst case is one extra LLM call that correctly says it can't find the answer.
    """
    top = float(hits[0]["score"]) if hits else 0.0
    if top >= min_score:
        return True, f"best match {top:.2f} reaches the minimum {min_score:.2f}"
    if hits and top >= min_score - BORDERLINE_BAND:
        # Passage text and section titles only. A filename says nothing about whether a passage answers.
        texts = [f"{h['payload'].get('section', '')} {h['payload'].get('text', '')}" for h in hits[:5]]
        if shares_keywords(question, texts):
            return True, f"borderline match {top:.2f}, but the question's key terms appear in the passages"
    if term_hits:
        return True, f"best match {top:.2f} is below the minimum, but a document literally contains a key term of your question"
    return False, f"best match {top:.2f} is below the minimum {min_score:.2f} and no key terms overlap"
