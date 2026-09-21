"""Dependency-free text chunking (paragraph -> line -> sentence -> word)."""
import re
from typing import Callable

_SPLITTERS: list[Callable[[str], list[str]]] = [
    lambda t: t.split("\n\n"),
    lambda t: t.split("\n"),
    lambda t: re.split(r"(?<=[.!?])\s+", t),
    lambda t: t.split(" "),
]


def _split(text: str, size: int, level: int = 0) -> list[str]:
    if len(text) <= size:
        return [text]
    if level >= len(_SPLITTERS):  # unbroken string: hard slice
        return [text[i : i + size] for i in range(0, len(text), size)]
    out: list[str] = []
    for part in _SPLITTERS[level](text):
        out.extend(_split(part, size, level + 1) if len(part) > size else [part])
    return out


def _overlap_tail(chunk: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tail = chunk[-overlap:]
    space = tail.find(" ")  # start on a word boundary
    return tail[space + 1 :] if space != -1 else tail


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into chunks of roughly `size` characters with `overlap` carry-over."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    pieces = [p.strip() for p in _split(text, size)]
    pieces = [p for p in pieces if p]

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= size:
            current += "\n" + piece
        else:
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail} {piece}".strip() if tail else piece
    if current:
        chunks.append(current)
    return chunks
