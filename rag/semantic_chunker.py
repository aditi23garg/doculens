"""Structure-aware semantic chunking.

1. A section (the text under one heading) that fits in `max_chars` becomes ONE chunk,
   so a topic is never cut in half.
2. A longer section is split where the *meaning* shifts: each sentence is embedded and
   the sharpest drops in similarity between neighbours become chunk boundaries.
3. Tiny neighbouring pieces are merged so no chunk is a stray fragment.

`embed_fn` is injected (list[str] -> list[vector]) so this module is pure and testable.
If embedding fails, it falls back to size-based packing on sentence boundaries.
"""
import math
import re
from dataclasses import dataclass
from typing import Callable, Optional

from rag.chunking import chunk_text
from rag.structure import Section

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class Chunk:
    text: str
    section: str  # "Heading > Subheading"
    page: Optional[int]
    page_end: Optional[int]


@dataclass
class _Unit:
    text: str
    page: Optional[int]
    lead: str  # separator that preceded it: "\n\n" paragraph, "\n" line, " " same line


_WRAPPED_LINE = re.compile(r"(?<![.!?:;|])\n(?=[a-z(])")  # hard-wrapped PDF line, not a new item
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _units(section: Section, max_chars: int) -> list[_Unit]:
    units: list[_Unit] = []
    for page, para in section.paragraphs:
        para = _WRAPPED_LINE.sub(" ", para)
        first_in_para = True
        for line in para.split("\n"):
            first_in_line = True
            for sentence in _SENTENCE.split(line):
                sentence = sentence.strip()
                if not sentence:
                    continue
                lead = "\n\n" if first_in_para else ("\n" if first_in_line else " ")
                pieces = chunk_text(sentence, max_chars, 0) if len(sentence) > max_chars else [sentence]
                for i, piece in enumerate(pieces):
                    units.append(_Unit(piece, page, lead if i == 0 else " "))
                first_in_para = first_in_line = False
    return units


def _assemble(units: list[_Unit]) -> str:
    return "".join((u.lead if i else "") + u.text for i, u in enumerate(units))


def _semantic_groups(units: list[_Unit], embed_fn: EmbedFn, max_chars: int, percentile: float) -> list[list[_Unit]]:
    n = len(units)
    breaks: set[int] = set()  # a break after unit i separates i from i+1
    if n >= 4:
        try:
            vectors = embed_fn([u.text for u in units])
            if len(vectors) == n:
                dist = [1 - _cosine(vectors[i], vectors[i + 1]) for i in range(n - 1)]
                threshold = _percentile(dist, percentile)
                for i, d in enumerate(dist):
                    neighbourhood = dist[max(0, i - 2) : i + 3]
                    if d >= threshold and d > 1e-9 and d == max(neighbourhood):
                        breaks.add(i)
        except Exception:
            breaks = set()  # fall back to size-only packing

    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    size = 0
    for i, unit in enumerate(units):
        if current and size + len(unit.text) + 1 > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append(unit)
        size += len(unit.text) + 1
        if i in breaks:
            groups.append(current)
            current, size = [], 0
    if current:
        groups.append(current)
    return groups


def _joined(a: Chunk, b: Chunk) -> Chunk:
    if a.section == b.section:
        text = f"{a.text}\n\n{b.text}"
    else:
        title = b.section.split(" > ")[-1] if b.section else ""
        text = f"{a.text}\n\n{title}\n{b.text}" if title else f"{a.text}\n\n{b.text}"
    return Chunk(text, a.section, a.page, b.page_end if b.page_end is not None else a.page_end)


def _merge_small(atoms: list[Chunk], max_chars: int, min_chars: int) -> list[Chunk]:
    out: list[Chunk] = []
    for atom in atoms:
        if out and len(out[-1].text) < min_chars and len(_joined(out[-1], atom).text) <= max_chars:
            out[-1] = _joined(out[-1], atom)
        else:
            out.append(atom)
    if len(out) >= 2 and len(out[-1].text) < min_chars and len(_joined(out[-2], out[-1]).text) <= max_chars:
        last = out.pop()
        out[-1] = _joined(out[-1], last)
    return out


def semantic_chunks(
    sections: list[Section],
    embed_fn: EmbedFn,
    max_chars: int = 1800,
    min_chars: int = 300,
    percentile: float = 85,
) -> list[Chunk]:
    atoms: list[Chunk] = []
    for sec in sections:
        label = " > ".join(sec.path)
        pages = [p for p, _ in sec.paragraphs if p is not None]
        first, last = (pages[0], pages[-1]) if pages else (None, None)
        body = "\n\n".join(t for _, t in sec.paragraphs)
        if not body.strip():
            continue
        if len(body) <= max_chars:
            atoms.append(Chunk(body, label, first, last))
            continue
        for group in _semantic_groups(_units(sec, max_chars), embed_fn, max_chars, percentile):
            page_vals = [u.page for u in group if u.page is not None]
            atoms.append(
                Chunk(_assemble(group), label, page_vals[0] if page_vals else None, page_vals[-1] if page_vals else None)
            )
    return _merge_small(atoms, max_chars, min_chars)
