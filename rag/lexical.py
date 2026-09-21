"""Tiny keyword helpers. Used as a second opinion when embedding scores are borderline,
because raw cosine scores vary with chunk size and embedding model."""
import re

_STOP = set(
    """a an the of in on at to for from by with about into over under between among and or but not no nor so yet
    if then than as is are was were be been being am do does did done doing have has had having it its this that
    these those there here what which who whom whose when where why how can could should would will shall may might
    must i me my we our you your he she they them their his her him us
    document documents doc docs file files report reports pdf text page pages paper section sections
    discussed discuss discusses discussing mentioned mention mentions described describe describes covered cover
    covers include includes included including tell show explain give list summary summarize summarise overview
    main key points point please information details detail""".split()
)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def key_terms(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOP}


def shares_keywords(question: str, texts: list[str]) -> bool:
    """True if any distinctive term of the question appears in any of the texts."""
    wanted = key_terms(question)
    if not wanted:
        return False
    found: set[str] = set()
    for t in texts:
        found |= key_terms(t)
    return bool(wanted & found)
