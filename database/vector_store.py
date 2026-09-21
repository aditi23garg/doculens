"""Qdrant Cloud vector operations.

One shared collection; every point carries `user_id` and `doc_id` payloads and
EVERY query is filtered by user_id, so users only ever search their own files.
(Renamed from qdrant_client.py to avoid shadowing the `qdrant_client` package.)
"""
import uuid
from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchText,
    MatchValue,
    MinShould,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from config.settings import get_settings


@lru_cache
def get_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30)


def ensure_collection() -> None:
    s = get_settings()
    client = get_client()
    if not client.collection_exists(s.qdrant_collection):
        client.create_collection(
            collection_name=s.qdrant_collection,
            vectors_config=VectorParams(size=s.embedding_dim, distance=Distance.COSINE),
        )
    for field in ("user_id", "doc_id"):
        try:
            client.create_payload_index(s.qdrant_collection, field, PayloadSchemaType.KEYWORD)
        except Exception:
            pass  # index already exists
    try:
        # Full-text index on the chunk body, so search_terms() below can find a chunk that
        # literally contains a distinctive query term even when its embedding doesn't rank
        # highest (specific model names, numbers and jargon are exactly where dense cosine
        # similarity is weakest). Runs on every startup, so it also backfills collections
        # created before this index existed.
        client.create_payload_index(s.qdrant_collection, "text", PayloadSchemaType.TEXT)
    except Exception:
        pass  # index already exists


def upsert_chunks(items: list[dict], batch_size: int = 64) -> None:
    """items: [{"vector": [...], "payload": {...}}]"""
    s = get_settings()
    client = get_client()
    points = [PointStruct(id=str(uuid.uuid4()), vector=i["vector"], payload=i["payload"]) for i in items]
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=s.qdrant_collection, points=points[start : start + batch_size])


def search(user_id: str, query_vector: list[float], doc_ids: Optional[list[str]], top_k: int) -> list[dict]:
    s = get_settings()
    must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if doc_ids:
        must.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))
    res = get_client().query_points(
        collection_name=s.qdrant_collection,
        query=query_vector,
        query_filter=Filter(must=must),
        limit=top_k,
        with_payload=True,
    )
    return [{"id": str(p.id), "score": p.score, "payload": p.payload or {}} for p in res.points]


def search_terms(
    user_id: str, query_vector: list[float], doc_ids: Optional[list[str]], terms: set[str], top_k: int
) -> list[dict]:
    """Vector search restricted to chunks whose text literally contains at least one of
    `terms` (a full-text payload match on the `text` field, see the index created in
    ensure_collection). Candidates are still ranked by embedding similarity among that
    filtered set - this narrows *which* chunks are eligible, it doesn't replace ranking.

    Used as a lexical safety net: a chunk that is a perfect literal match for a distinctive
    query term (a model name, a number, a rare technical word) can otherwise rank low enough
    on pure cosine similarity to miss the relevance margin entirely. Returns [] on any error
    (e.g. an older collection where the text index hasn't been created yet) so this is purely
    additive and never breaks plain semantic search.
    """
    if not terms:
        return []
    s = get_settings()
    must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if doc_ids:
        must.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))
    term_conditions = [FieldCondition(key="text", match=MatchText(text=t)) for t in sorted(terms)]
    try:
        res = get_client().query_points(
            collection_name=s.qdrant_collection,
            query=query_vector,
            query_filter=Filter(must=must, min_should=MinShould(conditions=term_conditions, min_count=1)),
            limit=top_k,
            with_payload=True,
        )
        return [{"id": str(p.id), "score": p.score, "payload": p.payload or {}} for p in res.points]
    except Exception:
        return []


def delete_document_chunks(user_id: str, doc_id: str) -> None:
    s = get_settings()
    get_client().delete(
        collection_name=s.qdrant_collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                ]
            )
        ),
    )


def list_chunks(user_id: str, doc_id: str) -> list[dict]:
    """All stored chunk payloads for one document, in reading order (for diagnostics)."""
    s = get_settings()
    flt = Filter(
        must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
        ]
    )
    out: list[dict] = []
    offset = None
    while True:
        points, offset = get_client().scroll(
            collection_name=s.qdrant_collection,
            scroll_filter=flt,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        out.extend(p.payload or {} for p in points)
        if offset is None:
            break
    return sorted(out, key=lambda p: p.get("chunk_index", 0))
