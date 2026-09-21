"""Document-grounded answering pipeline.

ask() returns (token_stream, result). `result` is filled in once the stream has
been fully consumed: final text, whether the answer was grounded, and sources.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator, Optional

from groq import Groq

from config.settings import get_settings
from rag import guardrails
from rag.citations import cited_numbers
from rag.prompts import build_answer_messages
from rag.query_planner import QUERY_PLAN_SYSTEM_PROMPT, merge_queries, parse_queries
from rag.retriever import RetrievedChunk, retrieve

_SNIPPET_CHARS = 700


@dataclass
class AnswerResult:
    text: str = ""
    grounded: bool = False
    sources: list[dict] = field(default_factory=list)
    top_score: float = 0.0
    debug: dict = field(default_factory=dict)  # queries, gate decision, ranked hits


@lru_cache
def _groq() -> Groq:
    return Groq(api_key=get_settings().groq_api_key)


def _plan_queries(question: str, history: list[dict]) -> list[str]:
    """Search queries for the index: the question (made standalone) plus alternate phrasings.

    Used only for retrieval. Never a knowledge source, never shown to the user.
    """
    s = get_settings()
    if not history and not s.query_expansion:
        return [question]
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-4:])
    user = (f"Conversation so far:\n{convo}\n\n" if history else "") + f"Latest message: {question}"
    try:
        res = _groq().chat.completions.create(
            model=s.groq_model,
            messages=[
                {"role": "system", "content": QUERY_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=200,
        )
        parsed = parse_queries(res.choices[0].message.content or "")
    except Exception:
        parsed = []
    if not parsed:
        return [question]
    # With history, the first parsed query is the standalone rewrite; without, keep the user's own words.
    primary = parsed[0] if history else question
    return merge_queries(primary, parsed[1:] if history else parsed, limit=3)


def _stream_llm(messages: list[dict]) -> Iterator[str]:
    stream = _groq().chat.completions.create(
        model=get_settings().groq_model,
        messages=messages,
        temperature=0,
        max_tokens=1200,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _pick_sources(text: str, chunks: list[RetrievedChunk]) -> list[dict]:
    cited = cited_numbers(text)
    chosen = [c for c in chunks if c.n in cited] or chunks
    return [
        {
            "n": c.n,
            "filename": c.filename,
            "section": c.section,
            "page": c.page,
            "page_end": c.page_end,
            "score": round(c.score, 3),
            "text": c.text[:_SNIPPET_CHARS],
        }
        for c in chosen
    ]


def _canned(text: str, result: AnswerResult, top_score: float = 0.0) -> Iterator[str]:
    result.text, result.grounded, result.sources, result.top_score = text, False, [], top_score
    yield text


def ask(
    user_id: str,
    question: str,
    doc_ids: Optional[list[str]],
    history: list[dict],
) -> tuple[Iterator[str], AnswerResult]:
    result = AnswerResult()
    question = question.strip()

    # 1. Greetings / thanks / "what can you do": answered locally.
    small_talk = guardrails.small_talk_reply(question)
    if small_talk:
        return _canned(small_talk, result), result

    # 2. Nothing to search.
    if not doc_ids:
        return _canned(guardrails.NO_DOCUMENTS_MESSAGE, result), result

    # 3. Retrieve. If the question itself doesn't match the documents -> refuse, no answer call.
    queries = _plan_queries(question, history)
    retrieval = retrieve(user_id, queries, doc_ids)
    result.debug = {
        "queries": retrieval.queries or queries,
        "gate": retrieval.gate,
        "hits": retrieval.debug_hits,
        "verdict": "sent" if retrieval.chunks else "refused_by_gate",
    }
    if not retrieval.chunks:
        return _canned(guardrails.REFUSAL_MESSAGE, result, retrieval.top_score), result

    # 4. Generate strictly from the retrieved passages, with recent turns only to disambiguate
    # what the user means (see rag/prompts.py rule 11) - never as an extra source of facts.
    messages = build_answer_messages(question, retrieval.chunks, history)

    def stream() -> Iterator[str]:
        parts: list[str] = []
        for tok in guardrails.guard_stream(_stream_llm(messages)):
            parts.append(tok)
            yield tok
        text = "".join(parts).strip()
        result.text = text
        result.top_score = retrieval.top_score
        result.grounded = text != guardrails.REFUSAL_MESSAGE
        if not result.grounded:
            result.debug["verdict"] = "refused_by_model"
        result.sources = _pick_sources(text, retrieval.chunks) if result.grounded else []

    return stream(), result
