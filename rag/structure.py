"""Turn extracted text into sections using the document's own headings.

Headings come from explicit markers (`#`, `##`, ... emitted for DOCX headings and used
by Markdown files) and, for PDF/TXT only, conservative heuristics (numbered headings such
as "2.1 Setup", or short ALL-CAPS lines). Each section remembers its heading path
("Phase 1 > PaddleOCR-VL") so chunks keep their context.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

_MD_HEADING = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,3})[.)]?\s+([A-Z][^\n]{2,70})$")


@dataclass
class Section:
    path: tuple[str, ...] = ()
    paragraphs: list[tuple[Optional[int], str]] = field(default_factory=list)  # (page, text)


def _heuristic_heading(line: str) -> Optional[tuple[int, str]]:
    s = line.strip()
    if not s or len(s) > 80 or s.endswith((".", ",", ";", ":")):
        return None
    m = _NUMBERED.match(s)
    if m:
        return m.group(1).count(".") + 1, s
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4 and s.upper() == s:
        return 1, s
    return None


def parse_sections(pages: list[tuple[Optional[int], str]], heuristic_headings: bool = False) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current = Section()

    def flush() -> None:
        if current.paragraphs:
            sections.append(current)

    for page, text in pages:
        buf: list[str] = []

        def end_paragraph() -> None:
            if buf:
                current.paragraphs.append((page, "\n".join(buf)))
                buf.clear()

        for raw in text.split("\n"):
            line = raw.rstrip()
            if not line.strip():
                end_paragraph()
                continue
            heading: Optional[tuple[int, str]] = None
            m = _MD_HEADING.match(line)
            if m:
                heading = (len(m.group(1)), m.group(2))
            elif heuristic_headings:
                heading = _heuristic_heading(line)
            if heading:
                end_paragraph()
                flush()
                level, title = heading
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current = Section(path=tuple(t for _, t in stack))
            else:
                buf.append(line)
        end_paragraph()
    flush()
    return sections
