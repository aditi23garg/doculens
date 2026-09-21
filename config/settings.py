"""Application settings, loaded from environment variables / a local .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_NAME = "DocuLens"
APP_TAGLINE = "Answers from your documents, and nothing else."

# env var name -> Settings attribute
_REQUIRED = {
    "GROQ_API_KEY": "groq_api_key",
    "GOOGLE_API_KEY": "google_api_key",
    "MONGODB_URI": "mongodb_uri",
    "QDRANT_URL": "qdrant_url",
    "QDRANT_API_KEY": "qdrant_api_key",
}
_PLACEHOLDER_HINTS = ("xxxx", "your_", "your-", "<your", "changeme")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM (answer generation)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Embeddings
    google_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768

    # Storage
    mongodb_uri: str = ""
    mongodb_db_name: str = "doculens"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "doculens_chunks"

    # Retrieval / grounding
    retrieval_top_k: int = 8  # hits fetched per search query
    # Off-topic gate. If the best match for the user's question reaches this cosine score it
    # passes. Within 0.10 below it, it still passes when the question's key terms actually
    # appear in the retrieved passages. Anything lower is refused WITHOUT calling the LLM.
    # The answering prompt is the real backstop, so this is deliberately lenient.
    min_relevance_score: float = 0.40
    # Once the gate passes, keep every passage within this distance of the best score.
    # This lets broad questions ("what models are discussed?") pull in all related sections.
    relevance_margin: float = 0.18
    max_context_chunks: int = 10  # cap on passages sent to the LLM
    # A rich section can be split across several chunks (rag/semantic_chunker.py splits
    # anything over chunk_max_chars). If only the leading chunk of a section clears the
    # margin above, the rest of that same section is pulled in anyway, up to this many
    # extra chunks, so the model isn't left with just the opening of a topic.
    section_complete_extra_chunks: int = 6
    # Safety net for questions about one specific term (a model name, a number, jargon):
    # a chunk that literally contains a distinctive term of the question is pulled in even
    # if its embedding score falls outside relevance_margin, up to this many extra chunks.
    # See rag/retriever.py:_lexical_rescue and the "text" full-text index in vector_store.py.
    lexical_rescue_extra_chunks: int = 3
    # Ask the LLM for 1-2 alternative search phrasings (search queries only, never answers).
    query_expansion: bool = True
    # Show a "Search details" panel under answers (queries, scores, gate decision).
    show_search_details: bool = True

    # Ingestion
    # Semantic chunking: split on headings first, then on meaning shifts inside long sections.
    chunk_max_chars: int = 1200
    chunk_min_chars: int = 250
    semantic_breakpoint_percentile: int = 85  # higher = fewer, larger chunks
    max_upload_mb: int = 20

    def get_missing_keys(self) -> list[str]:
        missing = []
        for env_name, attr in _REQUIRED.items():
            value = (getattr(self, attr) or "").strip()
            if not value or any(h in value.lower() for h in _PLACEHOLDER_HINTS):
                missing.append(env_name)
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
