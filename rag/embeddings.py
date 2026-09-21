"""Google Gemini embeddings (retrieval task types for documents vs. queries)."""
import time
from functools import lru_cache
from typing import Callable, Optional

from google import genai
from google.genai import types

from config.settings import get_settings

_BATCH = 50


@lru_cache
def _client() -> genai.Client:
    return genai.Client(api_key=get_settings().google_api_key)


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    s = get_settings()
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            res = _client().models.embed_content(
                model=s.embedding_model,
                # Each text wrapped as its own Content/Part, not a bare list of strings.
                # Some Gemini embedding models/SDK versions treat a bare list of strings as
                # PARTS OF ONE document rather than a batch of separate documents, and
                # silently return a single embedding back no matter how many texts were
                # sent - no error, just a short result. Wrapping explicitly is what Google's
                # own bug reports recommend as the fix (see googleapis/python-genai#2523).
                contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=s.embedding_dim
                ),
            )
            vectors = [list(e.values) for e in res.embeddings]
            if len(vectors) != len(texts):
                # Even with the wrapping above, refuse to silently return a short list: the
                # caller pairs these 1:1 with chunks (see document_processor.ingest_document),
                # and a length mismatch there used to mean silent data loss - one lucky
                # vector landing on the wrong chunk, most of the document simply never
                # getting stored, with no error anywhere. Fail loudly instead so ingestion
                # is rejected outright rather than silently corrupted.
                raise RuntimeError(
                    f"embedding API returned {len(vectors)} vectors for {len(texts)} inputs"
                )
            return vectors
        except Exception as exc:  # rate limits / transient errors / the mismatch above
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Embedding request failed: {last_exc}")


def embed_documents(
    texts: list[str],
    on_progress: Optional[Callable[[int, int], None]] = None,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH):
        vectors.extend(_embed(texts[start : start + _BATCH], task_type))
        if on_progress:
            on_progress(min(start + _BATCH, len(texts)), len(texts))
    return vectors


def embed_queries(texts: list[str]) -> list[list[float]]:
    return _embed(texts, "RETRIEVAL_QUERY")


def embed_query(text: str) -> list[float]:
    return embed_queries([text])[0]
