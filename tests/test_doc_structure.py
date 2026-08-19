"""
Tests for swarn/capabilities/doc_structure.py — the document tree derived from
stored data rather than from a second parse of the file.

The point of these is not that the tree is *good* (it is heuristic, and its
limits are documented), but that it is built from the SAME data `swarn ask`
answers from. Two parses that disagree is the failure this module removed.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import pdfplumber  # noqa: F401
    import reportlab   # noqa: F401
    from pydantic import BaseModel  # noqa: F401
    _HAVE_DEPS = True
except ImportError:
    _HAVE_DEPS = False


def _skip(why: str) -> None:
    print(f"    (skipped: {why})")


def _bilingual_bill(dest: str) -> None:
    """A bill in the shape the UPPCL one has: bilingual labels, ' | ' cells."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(dest, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, 780, "Electricity Bill Cum Notice")
    c.setFont("Helvetica", 10)
    # Two fields on one visual row, each label at one tab stop and its
    # colon-prefixed value at the next — the layout that produced zero fields
    # before _kv_segments existed.
    c.drawString(60, 740, "Bill No")
    c.drawString(160, 740, ": 382-209-037-556")
    c.drawString(300, 740, "Bill Month")
    c.drawString(400, 740, ": JUL-2026")
    c.drawString(60, 720, "Account No.")
    c.drawString(160, 720, ": 3821-043-405")
    c.setFont("Helvetica", 11)
    c.drawString(60, 680, "Meter readings are taken monthly by the department.")
    c.showPage()
    c.save()


def _stored(path: str):
    from swarn.capabilities.doc_store import ingest_document
    return ingest_document(path, save=False)


def _tree(path: str) -> dict:
    from agent.memory.multimodal_rag import get_multimodal_indexer
    return get_multimodal_indexer().extract_pdf_document(path)


# ── the bridge ─────────────────────────────────────────────────────────────

def test_elements_come_from_the_store_not_a_second_parse():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    from swarn.capabilities.doc_structure import elements_from_stored

    tmp = os.path.join(tempfile.mkdtemp(), "bill.pdf")
    _bilingual_bill(tmp)
    document = _stored(tmp)
    elements = elements_from_stored(document)

    assert elements, "no elements derived from the stored document"
    # Every line element must correspond to a line the store actually holds —
    # that identity is the whole point of deriving rather than re-parsing.
    stored_ids = {line.line_id for page in document.pages for line in page.lines}
    for element in elements:
        if element["kind"] == "line":
            assert element["line_id"] in stored_ids


def test_every_line_element_carries_a_box():
    # The old path had no coordinates at all, so nothing downstream could
    # verify anything it produced.
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    from swarn.capabilities.doc_structure import elements_from_stored

    tmp = os.path.join(tempfile.mkdtemp(), "bill.pdf")
    _bilingual_bill(tmp)
    for element in elements_from_stored(_stored(tmp)):
        if element["kind"] == "line":
            box = element["box"]
            assert box and len(box) == 4
            assert box[0] < box[2] and box[1] < box[3]


def test_elements_are_in_page_then_reading_order():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    from swarn.capabilities.doc_structure import elements_from_stored

    tmp = os.path.join(tempfile.mkdtemp(), "bill.pdf")
    _bilingual_bill(tmp)
    elements = elements_from_stored(_stored(tmp))
    keys = [(e["page"], e["top"]) for e in elements]
    assert keys == sorted(keys)


def test_font_size_and_weight_survive_the_round_trip():
    # Heading detection runs off these two attributes; if the store dropped
    # them the tree would flatten into one heading-less section.
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    from swarn.capabilities.doc_structure import elements_from_stored

    tmp = os.path.join(tempfile.mkdtemp(), "bill.pdf")
    _bilingual_bill(tmp)
    lines = [e for e in elements_from_stored(_stored(tmp)) if e["kind"] == "line"]
    title = next(e for e in lines if "Electricity Bill" in e["text"])
    body = next(e for e in lines if "Meter readings" in e["text"])
    assert title["size"] > body["size"]
    assert title["bold"] and not body["bold"]


# ── segment-aware field matching ───────────────────────────────────────────

def test_two_fields_on_one_row_are_both_found():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp = os.path.join(tempfile.mkdtemp(), "bill.pdf")
    _bilingual_bill(tmp)
    tree = _tree(tmp)
    assert "error" not in tree, tree.get("error")
    assert tree["fields"].get("Bill No") == "382-209-037-556"
    assert tree["fields"].get("Bill Month") == "JUL-2026"


def test_a_colon_prefixed_cell_is_the_previous_cells_value():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    from agent.memory.multimodal_rag import MultiModalIndexer

    assert MultiModalIndexer._kv_segments("Bill No | : 382 | Bill Month | : JUL") == \
        ["Bill No : 382", "Bill Month : JUL"]


def test_a_value_never_keeps_a_leading_colon():
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from agent.memory.multimodal_rag import MultiModalIndexer

    assert MultiModalIndexer._kv_match("Subdivision | : CHINHAT") == \
        ("Subdivision", "CHINHAT")


def test_all_pairs_on_a_line_are_returned():
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from agent.memory.multimodal_rag import MultiModalIndexer

    pairs = MultiModalIndexer._kv_pairs("Bill No | : 382 | Bill Month | : JUL")
    assert pairs == {"Bill No": "382", "Bill Month": "JUL"}


def test_a_line_with_no_pipes_behaves_exactly_as_before():
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from agent.memory.multimodal_rag import MultiModalIndexer

    assert MultiModalIndexer._kv_match("Invoice date: 2024-09-17") == \
        ("Invoice date", "2024-09-17")
    assert MultiModalIndexer._kv_match("There was one conclusion: continue") is None


# ── bilingual labels ───────────────────────────────────────────────────────

def test_a_bilingual_label_keeps_its_latin_half():
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from agent.memory.multimodal_rag import MultiModalIndexer

    assert MultiModalIndexer._kv_match("देय ितिथ / Due Date : 17-JUL-2026") == \
        ("Due Date", "17-JUL-2026")


def test_a_label_that_already_validates_is_never_rewritten():
    # "Net / Gross" is a legitimate label with a slash in it; rewriting it to
    # "Net" would quietly change what the field means.
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from agent.memory.multimodal_rag import MultiModalIndexer

    assert MultiModalIndexer._kv_match("Net / Gross: 100") == ("Net / Gross", "100")


def test_a_label_with_no_latin_half_is_still_rejected():
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from agent.memory.multimodal_rag import MultiModalIndexer

    assert MultiModalIndexer._kv_match("देय ितिथ / िबल संखया : 17-JUL") is None


# ── the tree as a whole ────────────────────────────────────────────────────

def test_the_tree_still_reports_its_standard_shape():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp = os.path.join(tempfile.mkdtemp(), "bill.pdf")
    _bilingual_bill(tmp)
    tree = _tree(tmp)
    for key in ("file", "n_pages", "metadata", "title", "fields", "sections", "counts"):
        assert key in tree, f"missing {key}"
    assert tree["n_pages"] == 1
    assert tree["counts"]["fields"] == len(tree["fields"])


def test_a_missing_file_is_reported_not_raised():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + pydantic")
    tree = _tree("/nonexistent/nope.pdf")
    assert "error" in tree
