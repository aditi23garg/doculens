"""Citation-marker parsing. Models sometimes emit [1], 【1】 or [1, 2]; accept them all."""
import re

# 1-2 digit numbers only, so years like [2023] are never mistaken for citations.
CITATION_RE = re.compile(r"[\[【〔]\s*(\d{1,2}(?:\s*[,，、]\s*\d{1,2})*)\s*[\]】〕]")


def cited_numbers(text: str) -> set[int]:
    nums: set[int] = set()
    for group in CITATION_RE.findall(text):
        nums.update(int(n) for n in re.findall(r"\d+", group))
    return nums
