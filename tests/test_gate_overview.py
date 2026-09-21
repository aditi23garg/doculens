"""Offline tests: off-topic gate, keyword helper, document-overview chunk."""
from rag.gate import evaluate_gate
from rag.lexical import key_terms, shares_keywords
from rag.outline import OVERVIEW_SECTION, build_overview
from rag.structure import parse_sections


def _hit(score, text="", section="", filename="doc.docx"):
    return {"score": score, "payload": {"text": text, "section": section, "filename": filename}}


# ── keyword helper ────────────────────────────────────────────────────────────
def test_meta_words_are_not_key_terms():
    assert key_terms("What models were discussed in the document?") == {"model"}


def test_plural_and_singular_match():
    assert shares_keywords("which models were discussed", ["PaddleOCR is a strong model for scans"])


def test_no_overlap_means_no_match():
    assert not shares_keywords("what is the capital of france", ["PaddleOCR reads scanned insurance forms"])


def test_question_with_only_filler_words_never_matches():
    assert not shares_keywords("what is the document about", ["anything at all"])


# ── gate ──────────────────────────────────────────────────────────────────────
def test_score_at_or_above_minimum_passes():
    assert evaluate_gate("anything", [_hit(0.41)], 0.40)[0]


def test_borderline_score_passes_only_with_shared_keywords():
    hits = [_hit(0.34, text="The candidate model list includes GLM-OCR")]
    assert evaluate_gate("What models were discussed in document", hits, 0.40)[0]
    assert not evaluate_gate("What is the capital of France?", hits, 0.40)[0]


def test_filename_alone_never_rescues_a_borderline_score():
    hits = [_hit(0.34, text="Premium invoices and billing cycles", filename="OCR_Model_Research.docx")]
    assert not evaluate_gate("What models were discussed in document", hits, 0.40)[0]


def test_far_below_minimum_is_refused_even_with_keywords():
    assert not evaluate_gate("models", [_hit(0.20, text="model model model")], 0.40)[0]


def test_literal_term_match_passes_even_when_the_embedding_score_is_far_below_the_floor():
    # A short, direct question ("what is paddleocr") can score far below both the minimum
    # and the borderline band on raw cosine similarity, even when the document genuinely
    # contains the term - a two-word query carries little embedding signal on its own, and
    # the score is nowhere near the 0.30 floor the ordinary borderline rescue requires.
    hits = [_hit(0.05, text="unrelated tagline")]
    term_hits = [_hit(0.10, text="PaddleOCR-VL is a compact OCR model", section="PaddleOCR-VL 1.6")]
    ok, why = evaluate_gate("what is paddleocr", hits, 0.40, term_hits)
    assert ok and "literally contains" in why


def test_no_hits_at_all_is_still_refused_without_a_term_match():
    assert not evaluate_gate("models", [], 0.40, term_hits=[])[0]


def test_empty_term_hits_do_not_change_the_ordinary_refusal():
    hits = [_hit(0.20, text="model model model")]
    assert not evaluate_gate("models", hits, 0.40, term_hits=[])[0]


def test_no_hits_is_refused():
    assert not evaluate_gate("models", [], 0.40)[0]


def test_gate_explains_itself():
    ok, why = evaluate_gate("x", [_hit(0.55)], 0.40)
    assert ok and "0.55" in why


# ── document overview chunk ───────────────────────────────────────────────────
_DOC = (
    "# OCR Model Research\n\nWe compare six OCR models for insurance claim documents.\n\n"
    "## PaddleOCR-VL 1.6\n\nPaddle details.\n\n## GLM-OCR\n\nGLM details.\n\n## Qwen2.5-VL\n\nQwen details."
)


def test_overview_lists_every_heading_and_the_opening_text():
    chunk = build_overview("OCR Model Research CIG", parse_sections([(None, _DOC)]))
    assert chunk.section == OVERVIEW_SECTION
    for name in ("PaddleOCR-VL 1.6", "GLM-OCR", "Qwen2.5-VL"):
        assert name in chunk.text
    assert "compare six OCR models" in chunk.text
    assert "  - GLM-OCR" in chunk.text  # nested under the title


def test_overview_caps_a_huge_outline():
    text = "\n\n".join(f"## Heading number {i}\n\nBody." for i in range(400))
    chunk = build_overview("Big", parse_sections([(None, text)]), max_chars=500)
    assert len(chunk.text) < 1200 and "more headings" in chunk.text


def test_overview_without_headings_still_gives_opening_text():
    chunk = build_overview("Notes", parse_sections([(None, "Just a plain paragraph about budgets.")]))
    assert "plain paragraph about budgets" in chunk.text and "Sections:" not in chunk.text


def test_empty_document_has_no_overview():
    assert build_overview("Empty", []) is None
