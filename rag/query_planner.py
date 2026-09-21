"""Search-query planning: resolve follow-ups and add alternative phrasings.

The LLM is used here ONLY to write search queries for the document index. Its output is
never shown to the user and never used as a source of facts.
"""
import json
import re

QUERY_PLAN_SYSTEM_PROMPT = """You write search queries for a document search engine.
Given the user's latest message (and the conversation so far, if any), output a JSON array of 1 to 3 short search queries.
- Query 1: the user's message rewritten as a standalone query. Resolve pronouns and references ("it", "the second one") using the conversation. If it is already standalone, keep it unchanged.
- Queries 2-3 (optional): different phrasings, or the specific sub-topics, that would appear in passages containing the answer. For "what models are discussed?" good extras are "model names compared or evaluated" and "candidate models and their details".
- Do NOT answer the question. Do not add facts, names or numbers that are not implied by the message.
Output only the JSON array of strings, nothing else."""


def parse_queries(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text or "", re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [q.strip() for q in data if isinstance(q, str) and q.strip()]


def merge_queries(primary: str, extras: list[str], limit: int = 3) -> list[str]:
    out = [primary]
    for q in extras:
        if q.lower() not in (x.lower() for x in out):
            out.append(q)
    return out[:limit]
