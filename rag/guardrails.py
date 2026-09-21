"""Guardrails that keep the assistant strictly inside the uploaded documents.

Layers (cheapest first):
  1. small-talk handling   - greetings/thanks answered locally, no LLM call
  2. no-documents check    - nothing uploaded -> ask user to upload
  3. relevance threshold   - (in the engine) no chunk above threshold -> refuse, no LLM call
  4. strict system prompt  - LLM may only use the retrieved excerpts
  5. sentinel filter       - if the LLM says NOT_FOUND_IN_DOCUMENTS, show the refusal
"""
import re
from typing import Iterable, Iterator, Optional

NOT_FOUND_SENTINEL = "NOT_FOUND_IN_DOCUMENTS"

REFUSAL_MESSAGE = (
    "I couldn't find this in your documents. DocuLens only answers from the files "
    "you've uploaded, so try rephrasing the question or add a document that covers it."
)
NO_DOCUMENTS_MESSAGE = (
    "You haven't added any documents yet. Upload a PDF, Word or text file in the "
    "Documents section, then ask your question."
)
GREETING_MESSAGE = (
    "Hello. Ask me anything that's covered in your uploaded documents and I'll answer "
    "from them, with the source shown."
)
THANKS_MESSAGE = "You're welcome. Ask another question about your documents whenever you're ready."
ABOUT_MESSAGE = (
    "I answer questions using only the documents you've uploaded, and every answer "
    "points to the passage it came from. I don't answer general-knowledge questions."
)

_GREETING = re.compile(
    r"^\s*(hi+|hello+|hey+|hola|namaste|yo|sup|good\s+(morning|afternoon|evening))\b[\s!.,?]*$", re.I
)
_THANKS = re.compile(r"^\s*(thanks|thank\s+you|thx|ty|cheers|ok(ay)?|cool|got\s+it|great)\b[\s!.,]*$", re.I)
_ABOUT = re.compile(
    r"^\s*(who\s+are\s+you|what\s+(are|can)\s+you(\s+do)?|help|how\s+(do|does)\s+(this|you)\s+work)\b[\s?!.]*$",
    re.I,
)


def small_talk_reply(text: str) -> Optional[str]:
    """Return a canned reply for greetings/thanks/'what can you do', else None."""
    if _GREETING.match(text):
        return GREETING_MESSAGE
    if _THANKS.match(text):
        return THANKS_MESSAGE
    if _ABOUT.match(text):
        return ABOUT_MESSAGE
    return None


def guard_stream(tokens: Iterable[str]) -> Iterator[str]:
    """Pass LLM tokens through, replacing a NOT_FOUND sentinel reply with REFUSAL_MESSAGE.

    The first few tokens are buffered only while they could still turn out to be
    the sentinel; once the reply diverges, everything streams straight through.
    """
    buffer = ""
    decided = False
    refused = False
    for tok in tokens:
        if decided:
            if not refused:
                yield tok
            continue
        buffer += tok
        probe = buffer.lstrip()
        if not probe:
            continue
        if probe.startswith(NOT_FOUND_SENTINEL):
            decided = refused = True
            yield REFUSAL_MESSAGE
        elif NOT_FOUND_SENTINEL.startswith(probe):
            continue  # could still become the sentinel
        else:
            decided = True
            yield buffer
    if not decided and buffer.strip():
        probe = buffer.strip()
        if NOT_FOUND_SENTINEL.startswith(probe) and len(probe) >= 8:
            yield REFUSAL_MESSAGE
        else:
            yield buffer
