"""MongoDB Atlas access: users, conversations, messages, document metadata.

Every read/write that touches user data is scoped by user_id so one account
can never see or modify another account's records.
"""
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

from config.settings import get_settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


@lru_cache
def _client() -> MongoClient:
    return MongoClient(get_settings().mongodb_uri, serverSelectionTimeoutMS=8000)


def _db():
    return _client()[get_settings().mongodb_db_name]


def _oid(value) -> Optional[ObjectId]:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def _ser(doc: Optional[dict]) -> Optional[dict]:
    if doc is None:
        return None
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    return doc


def ensure_indexes() -> None:
    _client().admin.command("ping")  # fail fast with a clear error
    d = _db()
    d.users.create_index("email", unique=True)
    d.conversations.create_index([("user_id", 1), ("updated_at", DESCENDING)])
    d.messages.create_index([("conversation_id", 1), ("created_at", 1)])
    d.documents.create_index([("user_id", 1), ("uploaded_at", DESCENDING)])
    d.documents.create_index([("user_id", 1), ("content_hash", 1)], unique=True)


# ── Users ─────────────────────────────────────────────────────────────────────
def create_user(name: str, email: str, password_hash: str) -> Optional[dict]:
    doc = {"name": name, "email": email, "password_hash": password_hash, "created_at": _now()}
    try:
        res = _db().users.insert_one(doc)
    except DuplicateKeyError:
        return None
    doc["_id"] = res.inserted_id
    return _ser(doc)


def get_user_by_email(email: str) -> Optional[dict]:
    return _ser(_db().users.find_one({"email": email}))


# ── Conversations & messages ──────────────────────────────────────────────────
def create_conversation(user_id: str, title: str) -> str:
    now = _now()
    res = _db().conversations.insert_one(
        {"user_id": user_id, "title": title, "created_at": now, "updated_at": now}
    )
    return str(res.inserted_id)


def list_conversations(user_id: str, limit: int = 40) -> list[dict]:
    cur = _db().conversations.find({"user_id": user_id}).sort("updated_at", DESCENDING).limit(limit)
    return [_ser(c) for c in cur]


def get_conversation(user_id: str, conversation_id: str) -> Optional[dict]:
    oid = _oid(conversation_id)
    return _ser(_db().conversations.find_one({"_id": oid, "user_id": user_id})) if oid else None


def delete_conversation(user_id: str, conversation_id: str) -> None:
    oid = _oid(conversation_id)
    if not oid:
        return
    if _db().conversations.delete_one({"_id": oid, "user_id": user_id}).deleted_count:
        _db().messages.delete_many({"conversation_id": conversation_id, "user_id": user_id})


def add_message(
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
    sources: Optional[list] = None,
    grounded: Optional[bool] = None,
    debug: Optional[dict] = None,
) -> None:
    now = _now()
    _db().messages.insert_one(
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "grounded": grounded,
            "debug": debug or None,
            "created_at": now,
        }
    )
    oid = _oid(conversation_id)
    if oid:
        _db().conversations.update_one({"_id": oid, "user_id": user_id}, {"$set": {"updated_at": now}})


def get_messages(user_id: str, conversation_id: str) -> list[dict]:
    cur = _db().messages.find({"user_id": user_id, "conversation_id": conversation_id}).sort("created_at", 1)
    return [
        {
            "role": m["role"],
            "content": m["content"],
            "sources": m.get("sources", []),
            "grounded": m.get("grounded"),
            "debug": m.get("debug"),
        }
        for m in cur
    ]


# ── Documents (metadata only; chunks live in Qdrant) ──────────────────────────
def create_document(user_id: str, filename: str, file_type: str, size_bytes: int, content_hash: str) -> str:
    res = _db().documents.insert_one(
        {
            "user_id": user_id,
            "filename": filename,
            "file_type": file_type,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
            "status": "processing",
            "chunk_count": 0,
            "page_count": None,
            "uploaded_at": _now(),
        }
    )
    return str(res.inserted_id)


def update_document(user_id: str, doc_id: str, **fields) -> None:
    oid = _oid(doc_id)
    if oid:
        _db().documents.update_one({"_id": oid, "user_id": user_id}, {"$set": fields})


def get_document_by_hash(user_id: str, content_hash: str) -> Optional[dict]:
    return _ser(_db().documents.find_one({"user_id": user_id, "content_hash": content_hash}))


def list_documents(user_id: str) -> list[dict]:
    return [_ser(d) for d in _db().documents.find({"user_id": user_id}).sort("uploaded_at", DESCENDING)]


def delete_document(user_id: str, doc_id: str) -> None:
    oid = _oid(doc_id)
    if oid:
        _db().documents.delete_one({"_id": oid, "user_id": user_id})
