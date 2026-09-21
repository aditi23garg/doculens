"""Per-document overview chunk: the heading outline plus the opening text.

Broad questions ("what models were discussed?", "what is this document about?") don't
resemble any single section, so they score poorly against section chunks. The overview
chunk lists every heading, so those questions have something that actually matches.
"""
from typing import Optional

from rag.semantic_chunker import Chunk
from rag.structure import Section

OVERVIEW_SECTION = "Document overview"


def build_overview(
    title: str, sections: list[Section], max_chars: int = 1500, intro_chars: int = 500
) -> Optional[Chunk]:
    headings: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for sec in sections:
        for depth in range(len(sec.path)):
            key = sec.path[: depth + 1]
            if key not in seen:
                seen.add(key)
                headings.append("  " * depth + "- " + sec.path[depth])

    listed: list[str] = []
    size = 0
    for i, line in enumerate(headings):
        if size + len(line) + 1 > max_chars:
            listed.append(f"- … and {len(headings) - i} more headings")
            break
        listed.append(line)
        size += len(line) + 1

    intro, first_page = "", None
    for sec in sections:
        if sec.paragraphs:
            first_page = sec.paragraphs[0][0]
            flat = " ".join(" ".join(t for _, t in sec.paragraphs).split())
            intro = flat[:intro_chars] + ("…" if len(flat) > intro_chars else "")
            break

    if not listed and not intro:
        return None
    parts = [f"Document overview: {title}"]
    if listed:
        parts.append("Sections:\n" + "\n".join(listed))
    if intro:
        parts.append("Opening text: " + intro)
    return Chunk("\n\n".join(parts), OVERVIEW_SECTION, first_page, first_page)
