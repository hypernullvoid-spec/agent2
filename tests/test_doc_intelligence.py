"""
Tests for swarn/capabilities/doc_intelligence.py — the bounding-box inspector.

Two groups, following the same split as test_pdf_extract.py:

  * Schema/parsing tests (BoundingBox coercion, VLM payload parsing, OCR line
    segmentation, tag placement) run on plain Python values and need only
    pydantic — no image rendering, no document.
  * Rendering and end-to-end tests need Pillow. They SKIP (print and return)
    rather than fail when it is missing, so run_tests.py stays green on a
    machine that never installed it.

Nothing here touches the network or needs an API key: the mock backend is the
default precisely so this file can assert on real output.
"""

import json
import os
import tempfile


def _have(*mods) -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in mods)


_HAVE_PYDANTIC = _have("pydantic")
_HAVE_PIL = _have("PIL")

if _HAVE_PYDANTIC:
    from swarn.capabilities.doc_intelligence import (
        CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, MOCK_INVOICE_LAYOUT, REAL_BACKENDS,
        BoundingBox, DocumentExtractionResult, DocumentInspector,
        DocumentIntelligenceError, ExtractedField, _assert_no_mock_leak,
        _has_text_layer, _snake_case, _split_columns, _stacked_pairs,
        _words_after_colon, extract_entities, fields_from_payload, group_lines,
        mock_fields, parse_vlm_response,
    )


def _skip(reason: str) -> bool:
    print(f"    (skipped: {reason})")
    return True


# ─────────────────────────────── BoundingBox coercion


def test_normalized_floats_pass_through_unchanged():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    box = BoundingBox(xmin=0.1, ymin=0.2, xmax=0.5, ymax=0.3)
    assert (box.xmin, box.ymin, box.xmax, box.ymax) == (0.1, 0.2, 0.5, 0.3)


def test_thousandths_convention_is_rescaled():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    # Gemini / Qwen2.5-VL emit integers 0-1000.
    box = BoundingBox(xmin=100, ymin=200, xmax=500, ymax=300)
    assert (box.xmin, box.ymin, box.xmax, box.ymax) == (0.1, 0.2, 0.5, 0.3)


def test_slight_overshoot_clamps_instead_of_rescaling():
    # 1.02 is rounding noise on a 0.0-1.0 box, NOT the 0-1000 convention —
    # rescaling it would collapse the whole box to a thousandth of the page.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    box = BoundingBox(xmin=0.1, ymin=0.1, xmax=1.02, ymax=1.05)
    assert box.xmax == 1.0 and box.ymax == 1.0


def test_reversed_corners_are_reordered():
    # A negative-width rectangle draws as nothing at all, silently.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    box = BoundingBox(xmin=0.5, ymin=0.3, xmax=0.1, ymax=0.2)
    assert box.xmin < box.xmax and box.ymin < box.ymax
    assert (box.xmin, box.xmax) == (0.1, 0.5)


def test_negative_coordinates_are_clamped_to_the_page():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    box = BoundingBox(xmin=-0.2, ymin=-0.1, xmax=0.4, ymax=0.4)
    assert box.xmin == 0.0 and box.ymin == 0.0


def test_from_vlm_reads_ymin_xmin_ymax_xmax_order():
    # The single most common silent bug when consuming VLM boxes: reading the
    # list as [xmin, ymin, ...] transposes every rectangle on the page.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    box = BoundingBox.from_vlm([200, 100, 300, 500])
    assert (box.xmin, box.ymin, box.xmax, box.ymax) == (0.1, 0.2, 0.5, 0.3)


def test_from_vlm_rejects_wrong_arity():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    try:
        BoundingBox.from_vlm([1, 2, 3])
    except DocumentIntelligenceError:
        return
    raise AssertionError("a 3-element box should not validate")


def test_from_pixels_and_to_pixels_round_trip():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    box = BoundingBox.from_pixels((100, 200, 500, 300), width=1000, height=1000)
    assert box.to_pixels(1000, 1000) == (100, 200, 500, 300)


def test_to_pixels_never_returns_a_zero_width_box():
    # A hairline box must still render; 0-width would be invisible.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    x0, y0, x1, y1 = BoundingBox(xmin=0.5, ymin=0.5, xmax=0.5, ymax=0.5).to_pixels(100, 100)
    assert x1 > x0 and y1 > y0


# ─────────────────────────────── ExtractedField


def test_confidence_tiers_and_colors_partition_the_range():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    box = BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1)

    def tier(conf):
        return ExtractedField(field_name="f", field_value="v", confidence=conf, box=box).tier

    assert tier(0.99) == "high"
    assert tier(CONFIDENCE_HIGH) == "high"          # boundary is inclusive upward
    assert tier(CONFIDENCE_HIGH - 0.01) == "medium"
    assert tier(CONFIDENCE_MEDIUM) == "medium"
    assert tier(CONFIDENCE_MEDIUM - 0.01) == "low"
    assert len({tier(0.99), tier(0.7), tier(0.1)}) == 3


def test_confidence_outside_zero_to_one_is_rejected():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    box = BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1)
    for bad in (1.5, -0.1):
        try:
            ExtractedField(field_name="f", field_value="v", confidence=bad, box=box)
        except Exception:
            continue
        raise AssertionError(f"confidence={bad} should not validate")


def test_overlay_label_ellipsizes_without_touching_the_value():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    long_value = "x" * 200
    field = ExtractedField(field_name="addr", field_value=long_value, confidence=0.9,
                            box=BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1))
    assert len(field.overlay_label()) < 60
    assert field.field_value == long_value      # JSON keeps the full text


# ─────────────────────────────── VLM response parsing


def test_parses_bare_json_and_fenced_json_alike():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    payload = '{"fields": [{"field_name": "a", "field_value": "b", "box_2d": [0, 0, 10, 10]}]}'
    assert parse_vlm_response(payload)["fields"][0]["field_name"] == "a"
    fenced = f"Sure, here you go:\n```json\n{payload}\n```\n"
    assert parse_vlm_response(fenced)["fields"][0]["field_name"] == "a"


def test_parses_json_wrapped_in_prose_without_a_fence():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    text = 'I found one field. {"fields": []} Let me know if you need more.'
    assert parse_vlm_response(text) == {"fields": []}


def test_unparseable_response_raises_instead_of_returning_no_fields():
    # "zero fields" and "the model did not answer in JSON" are very different
    # findings — collapsing them would hide a broken endpoint.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    for junk in ("", "I cannot read this document."):
        try:
            parse_vlm_response(junk)
        except DocumentIntelligenceError:
            continue
        raise AssertionError(f"{junk!r} should not parse")


def test_payload_shapes_and_box_key_aliases_all_convert():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    for payload in (
        {"fields": [{"field_name": "n", "field_value": "v", "box_2d": [100, 200, 300, 400]}]},
        [{"field_name": "n", "field_value": "v", "bbox": [100, 200, 300, 400]}],
        {"entities": [{"name": "n", "value": "v", "box": [100, 200, 300, 400]}]},
        {"fields": [{"field_name": "n", "field_value": "v",
                     "boundingBox": {"ymin": 100, "xmin": 200, "ymax": 300, "xmax": 400}}]},
    ):
        fields = fields_from_payload(payload)
        assert len(fields) == 1, payload
        assert fields[0].field_name == "n"
        assert fields[0].box.ymin == 0.1


def test_malformed_entries_are_dropped_not_fatal():
    # One bad field out of three should not cost the caller the other two.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    fields = fields_from_payload({"fields": [
        {"field_name": "good", "field_value": "1", "box_2d": [0, 0, 100, 100]},
        {"field_name": "no_box", "field_value": "2"},
        {"field_name": "bad_box", "field_value": "3", "box_2d": [0, 0]},
        {"field_name": "nan_box", "field_value": "4", "box_2d": ["a", "b", "c", "d"]},
        {"field_name": "also_good", "field_value": "5", "box_2d": [0, 0, 100, 100]},
    ]})
    assert [f.field_name for f in fields] == ["good", "also_good"]


def test_missing_confidence_defaults_to_certain():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    fields = fields_from_payload(
        {"fields": [{"field_name": "n", "field_value": "v", "box_2d": [0, 0, 10, 10]}]})
    assert fields[0].confidence == 1.0


# ─────────────────────────────── OCR helpers


def _word(text, left, width, top=100, height=20, conf=95.0):
    return {"text": text, "left": left, "width": width, "top": top,
            "height": height, "conf": conf}


def test_wide_gap_splits_two_columns_of_one_ocr_line():
    # Tesseract returns side-by-side columns as a single line; without this
    # split the field ends up named after the neighbouring column's text.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    words = [_word("Nashik", 50, 80), _word("Municipal", 135, 110),
             _word("DUE", 600, 55), _word("DATE:", 660, 70), _word("2024-10-17", 740, 120)]
    segments = _split_columns(words)
    assert len(segments) == 2
    assert [w["text"] for w in segments[0]] == ["Nashik", "Municipal"]
    assert [w["text"] for w in segments[1]][0] == "DUE"


def test_gap_after_a_colon_does_not_split_a_right_aligned_value():
    # "INVOICE NO.:" at a tab stop with the value flush right is one field,
    # not two columns.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    words = [_word("INVOICE", 600, 90), _word("NO.:", 695, 50), _word("INV-2024-0917", 900, 130)]
    assert len(_split_columns(words)) == 1


def test_column_split_threshold_scales_with_text_height():
    # The same pixel gap is a column break in 10pt text and ordinary word
    # spacing in 40pt text.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    small = [_word("A", 0, 20, height=10), _word("B", 60, 20, height=10)]
    large = [_word("A", 0, 20, height=60), _word("B", 60, 20, height=60)]
    assert len(_split_columns(small)) == 2
    assert len(_split_columns(large)) == 1


def test_value_words_are_those_after_the_colon():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    words = [_word("Invoice", 0, 50), _word("No:", 60, 30), _word("INV-1", 100, 60)]
    assert [w["text"] for w in _words_after_colon(words)] == ["INV-1"]
    # A colon with no space after it still carries the value.
    glued = [_word("Date:2024-09-17", 0, 120)]
    assert _words_after_colon(glued) == glued
    # No colon at all means no field.
    assert _words_after_colon([_word("just", 0, 40), _word("prose", 50, 50)]) == []


# ─────────────────────────────── mock backend + result model


def test_mock_fields_match_the_shared_layout_table():
    # The renderer and the mock extractor must read from ONE table, or the
    # drawn boxes stop matching the drawn text.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    fields = mock_fields()
    assert len(fields) == len(MOCK_INVOICE_LAYOUT)
    assert {f.field_name for f in fields} == {i["name"] for i in MOCK_INVOICE_LAYOUT}


def test_mock_layout_covers_all_three_confidence_tiers():
    # A demo that only ever draws green boxes does not demonstrate the point.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    assert {f.tier for f in mock_fields()} == {"high", "medium", "low"}


def test_mock_layout_boxes_are_all_on_the_page():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    for field in mock_fields():
        box = field.box
        assert 0.0 <= box.xmin < box.xmax <= 1.0, field.field_name
        assert 0.0 <= box.ymin < box.ymax <= 1.0, field.field_name


def test_target_schema_narrows_the_mock_output():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    names = mock_fields(["invoice_number", "total_due"])
    assert sorted(f.field_name for f in names) == ["invoice_number", "total_due"]


def test_result_helpers_summarize_without_an_image():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    result = DocumentExtractionResult(
        document_name="x.pdf", page_number=2, fields=mock_fields(), backend="mock")
    assert result.field_map()["invoice_number"] == "INV-2024-0917"
    assert all(f.confidence < CONFIDENCE_MEDIUM for f in result.low_confidence())
    assert "page 2" in result.summary()
    assert json.loads(result.to_json())["page_number"] == 2


# ─────────────────────────────── rendering + end to end (needs Pillow)


def test_annotated_image_is_written_and_matches_the_source_size():
    if not (_HAVE_PYDANTIC and _HAVE_PIL):
        return _skip("needs pydantic + Pillow")
    from PIL import Image
    from swarn.capabilities.doc_intelligence import render_mock_invoice

    with tempfile.TemporaryDirectory() as tmp:
        source = render_mock_invoice((500, 700))
        path = DocumentInspector(artifacts_dir=tmp).draw_bounding_boxes(
            source, mock_fields(), "annotated.png")

        assert os.path.dirname(path) == tmp     # relative names resolve into artifacts_dir
        with Image.open(path) as annotated:
            assert annotated.size == source.size
            # Boxes are drawn ON the page, so the result cannot be identical
            # to the input, and must not be a blank canvas either.
            assert annotated.convert("RGB").tobytes() != source.convert("RGB").tobytes()
            assert len(annotated.convert("RGB").getcolors(maxcolors=1 << 20)) > 3


def test_absolute_output_path_is_honored_verbatim():
    if not (_HAVE_PYDANTIC and _HAVE_PIL):
        return _skip("needs pydantic + Pillow")
    from swarn.capabilities.doc_intelligence import render_mock_invoice

    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "nested", "out.png")
        path = DocumentInspector(artifacts_dir="/nonexistent").draw_bounding_boxes(
            render_mock_invoice((300, 400)), mock_fields()[:2], target)
        assert path == target and os.path.exists(target)


def test_label_tags_do_not_overlap_each_other():
    # Stacked form fields otherwise print their labels on top of one another,
    # which is worse than no label at all — you cannot tell which tag belongs
    # to which box.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    placed = []
    for i in range(6):
        # Six boxes crammed into a narrow band, all at the same x.
        rect = DocumentInspector._place_tag(
            box=(10, 100 + i * 30, 200, 120 + i * 30),
            tag_w=180, tag_h=24, gap=4, width=1000, height=1400, obstacles=placed)
        placed.append(rect)
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            assert a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3], (a, b)


def test_tags_stay_inside_the_page():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    # A box hugging the top-right corner: the tag cannot go above it, and
    # must not run off the right edge either.
    rect = DocumentInspector._place_tag(
        box=(950, 0, 999, 20), tag_w=200, tag_h=24, gap=4,
        width=1000, height=1400, obstacles=[])
    assert rect[0] >= 0 and rect[1] >= 0
    assert rect[2] <= 1000 and rect[3] <= 1400


def test_process_document_end_to_end_on_a_generated_image():
    if not (_HAVE_PYDANTIC and _HAVE_PIL):
        return _skip("needs pydantic + Pillow")
    from swarn.capabilities.doc_intelligence import create_mock_document

    with tempfile.TemporaryDirectory() as tmp:
        document = create_mock_document(os.path.join(tmp, "invoice.png"))
        result = DocumentInspector(artifacts_dir=tmp, backend="mock").process_document(document)

        assert result.document_name == "invoice.png"
        assert result.page_number == 1
        assert result.backend == "mock"
        assert result.field_map()["total_due"] == "1,77,944.00"
        assert os.path.exists(result.annotated_image_path)
        assert result.raw_json["image_size"] == [1000, 1400]


def test_target_schema_validation_result_is_reported_not_raised():
    if not (_HAVE_PYDANTIC and _HAVE_PIL):
        return _skip("needs pydantic + Pillow")
    from pydantic import BaseModel
    from swarn.capabilities.doc_intelligence import create_mock_document

    class Invoice(BaseModel):
        invoice_number: str
        total_due: str

    class NotOnThisDocument(BaseModel):
        passport_number: str

    with tempfile.TemporaryDirectory() as tmp:
        document = create_mock_document(os.path.join(tmp, "invoice.png"))
        inspector = DocumentInspector(artifacts_dir=tmp, backend="mock")

        ok = inspector.process_document(document, target_schema=Invoice, annotate=False)
        assert sorted(f.field_name for f in ok.fields) == ["invoice_number", "total_due"]
        assert ok.raw_json["validated"]["invoice_number"] == "INV-2024-0917"
        assert "schema_error" not in ok.raw_json

        # A document that lacks the requested field is a FINDING, not a crash:
        # the boxes that were found are still worth returning.
        bad = inspector.process_document(document, target_schema=NotOnThisDocument,
                                          annotate=False)
        assert "schema_error" in bad.raw_json


def test_input_problems_raise_document_intelligence_error():
    if not (_HAVE_PYDANTIC and _HAVE_PIL):
        return _skip("needs pydantic + Pillow")
    from swarn.capabilities.doc_intelligence import create_mock_document

    with tempfile.TemporaryDirectory() as tmp:
        inspector = DocumentInspector(artifacts_dir=tmp)
        document = create_mock_document(os.path.join(tmp, "invoice.png"))
        odd = os.path.join(tmp, "notes.txt")
        with open(odd, "w", encoding="utf-8") as fh:
            fh.write("not a document")

        for call, expected in (
            (lambda: inspector.process_document("/definitely/not/here.pdf"), "not found"),
            (lambda: inspector.process_document(odd), "unsupported"),
            (lambda: inspector.process_document(document, page_number=3), "single-page"),
            (lambda: inspector.process_document(document, page_number=0), ">= 1"),
        ):
            try:
                call()
            except DocumentIntelligenceError as exc:
                assert expected in str(exc), (expected, str(exc))
                continue
            raise AssertionError(f"expected a DocumentIntelligenceError mentioning {expected!r}")


def test_unknown_backend_is_rejected_at_construction():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    try:
        DocumentInspector(backend="magic")
    except DocumentIntelligenceError as exc:
        assert "magic" in str(exc)
        return
    raise AssertionError("an unknown backend should not construct")


def test_vlm_backend_without_a_key_is_a_clear_error_not_a_silent_mock():
    # Asking for the VLM and quietly getting synthetic data back would be the
    # worst possible failure mode for this module.
    if not (_HAVE_PYDANTIC and _HAVE_PIL):
        return _skip("needs pydantic + Pillow")
    from swarn.capabilities.doc_intelligence import create_mock_document

    with tempfile.TemporaryDirectory() as tmp:
        document = create_mock_document(os.path.join(tmp, "invoice.png"))
        inspector = DocumentInspector(artifacts_dir=tmp, backend="vlm", vlm_api_key="")
        inspector.vlm_api_key = ""       # ignore any key in the ambient environment
        try:
            inspector.process_document(document)
        except DocumentIntelligenceError as exc:
            assert "SWARN_VLM_API_KEY" in str(exc)
            return
    raise AssertionError("the vlm backend must not fall back to mock silently")


# ─────────────────────────────── PDF path (optional deps)


def _make_pdf(dest: str) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

    styles = getSampleStyleSheet()
    SimpleDocTemplate(dest, pagesize=letter).build([
        Paragraph("Invoice INV-2024-0917", styles["Title"]),
        PageBreak(),
        Paragraph("Continuation page", styles["Title"]),
    ])


def test_pdf_pages_rasterize_and_page_selection_works():
    if not (_HAVE_PYDANTIC and _HAVE_PIL and _have("reportlab")):
        return _skip("needs pydantic + Pillow + reportlab")
    if not (_have("pdf2image") or _have("pdfplumber")):
        return _skip("needs pdf2image or pdfplumber to rasterize")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "invoice.pdf")
        _make_pdf(pdf)
        inspector = DocumentInspector(artifacts_dir=tmp)

        pages = inspector.convert_pdf_to_images(pdf)
        assert len(pages) == 2
        assert pages[0].size[0] > 100 and pages[0].mode == "RGB"

        result = inspector.process_document(pdf, page_number=2)
        assert result.page_number == 2
        assert result.raw_json["page_count"] == 2
        assert os.path.exists(result.annotated_image_path)

        try:
            inspector.process_document(pdf, page_number=99)
        except DocumentIntelligenceError as exc:
            assert "out of range" in str(exc)
        else:
            raise AssertionError("page 99 of a 2-page PDF should be an error")


def test_process_all_pages_annotates_every_page_separately():
    if not (_HAVE_PYDANTIC and _HAVE_PIL and _have("reportlab")):
        return _skip("needs pydantic + Pillow + reportlab")
    if not (_have("pdf2image") or _have("pdfplumber")):
        return _skip("needs pdf2image or pdfplumber to rasterize")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "invoice.pdf")
        _make_pdf(pdf)
        results = DocumentInspector(artifacts_dir=tmp).process_all_pages(pdf)

        assert [r.page_number for r in results] == [1, 2]
        paths = {r.annotated_image_path for r in results}
        assert len(paths) == 2 and all(os.path.exists(p) for p in paths)


# ─────────────────────────────── agent tool + CLI wiring


def test_tool_wrapper_returns_json_and_reports_errors_as_strings():
    if not (_HAVE_PYDANTIC and _HAVE_PIL):
        return _skip("needs pydantic + Pillow")
    from agent.tools import TOOL_REGISTRY, swarn_doc_inspect
    from swarn.capabilities.doc_intelligence import create_mock_document

    assert "swarn_doc_inspect" in TOOL_REGISTRY

    with tempfile.TemporaryDirectory() as tmp:
        document = create_mock_document(os.path.join(tmp, "invoice.png"))

        data = json.loads(swarn_doc_inspect(document, backend="mock"))
        assert data["n_fields"] == len(MOCK_INVOICE_LAYOUT)
        assert data["backend"] == "mock"
        assert data["fields"][0]["box"]["xmin"] >= 0.0
        assert os.path.exists(data["annotated_image_path"])

        # raw_json is trimmed by default — it would otherwise duplicate every
        # field into the agent's context a second time.
        assert "fields" not in data["raw_json"]
        assert "fields" in json.loads(
            swarn_doc_inspect(document, backend="mock", include_raw=True))["raw_json"]

        assert swarn_doc_inspect("/definitely/not/here.pdf").startswith("Error:")


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION: mock must never stand in for a real document
# ══════════════════════════════════════════════════════════════════════════════
#
# The bug these cover: `auto` resolved to the mock backend whenever no VLM key
# was set, so a real PDF came back reporting the synthetic invoice's vendor,
# invoice number, and total as if they had been read off the user's document.


def _fixture_pdf(dest: str) -> None:
    """A small deterministic PDF with a real text layer and known contents.

    Values are chosen to share no substring with the mock invoice, so any mock
    leakage into a real extraction is unambiguous rather than a coincidence."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    ss = getSampleStyleSheet()
    caption = ParagraphStyle("caption", parent=ss["BodyText"], fontSize=8,
                              textColor="#666666")
    value = ParagraphStyle("value", parent=ss["BodyText"], fontSize=13)

    SimpleDocTemplate(dest, pagesize=A4).build([
        Paragraph("QUARTZ LOGISTICS PRIVATE LIMITED", ss["Title"]),
        Spacer(1, 18),
        Paragraph("CONSIGNMENT REFERENCE", caption),
        Paragraph("Zephyr Northbound Route", value),
        Spacer(1, 14),
        Paragraph("Docket No: QLX-5591-KT", value),
        Paragraph("Dispatch Date: 2027-03-04", value),
        Paragraph("Contact: ops.desk@quartzlogistics.example", value),
    ])


FIXTURE_VALUES = ("QUARTZ LOGISTICS", "Zephyr Northbound Route",
                  "QLX-5591-KT", "2027-03-04", "ops.desk@quartzlogistics.example")

# Verbatim from MOCK_INVOICE_LAYOUT. None of these appear in the fixture, so
# seeing one in a real extraction means synthetic data leaked.
MOCK_VALUES = ("ESDS Software Solution Ltd.", "INV-2024-0917",
               "Nashik Municipal Corporation", "Rajiv Gandhi Bhavan, Nashik 422001",
               "PO-77120", "1,77,944.00", "S. Deshmukh")


def _can_make_pdf() -> bool:
    return _HAVE_PYDANTIC and _HAVE_PIL and _have("reportlab", "pdfplumber")


def test_real_document_without_a_vlm_key_never_selects_mock():
    # THE regression. Previously this returned backend='mock' and the
    # synthetic invoice's contents.
    if not _can_make_pdf():
        return _skip("needs pydantic + Pillow + reportlab + pdfplumber")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "consignment.pdf")
        _fixture_pdf(pdf)

        inspector = DocumentInspector(artifacts_dir=tmp, backend="auto")
        inspector.vlm_api_key = ""          # no credentials, whatever the environment says
        result = inspector.process_document(pdf)

        assert result.backend != "mock"
        assert result.backend in REAL_BACKENDS
        values = " ".join(str(f.field_value) for f in result.fields)
        for leaked in MOCK_VALUES:
            assert leaked not in values, f"mock value {leaked!r} leaked into a real extraction"


def test_auto_prefers_the_text_layer_when_the_pdf_has_one():
    if not _can_make_pdf():
        return _skip("needs pydantic + Pillow + reportlab + pdfplumber")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "consignment.pdf")
        _fixture_pdf(pdf)
        assert _has_text_layer(pdf) is True

        inspector = DocumentInspector(artifacts_dir=tmp, backend="auto")
        inspector.vlm_api_key = ""
        assert inspector._resolve_backend(pdf) == "text"


def test_auto_falls_back_to_ocr_for_an_image_with_no_text_layer():
    # An image can have no text layer by definition, so auto must route it to
    # OCR — never to mock.
    if not (_HAVE_PYDANTIC and _HAVE_PIL):
        return _skip("needs pydantic + Pillow")
    from swarn.capabilities.doc_intelligence import _tesseract_available, create_mock_document

    with tempfile.TemporaryDirectory() as tmp:
        image = create_mock_document(os.path.join(tmp, "scan.png"))
        assert _has_text_layer(image) is False

        inspector = DocumentInspector(artifacts_dir=tmp, backend="auto")
        inspector.vlm_api_key = ""
        if not _tesseract_available():
            return _skip("needs the tesseract binary")
        assert inspector._resolve_backend(image) == "ocr"


def test_missing_ocr_dependency_raises_an_actionable_error_not_mock():
    if not (_HAVE_PYDANTIC and _HAVE_PIL):
        return _skip("needs pydantic + Pillow")
    import swarn.capabilities.doc_intelligence as di
    from swarn.capabilities.doc_intelligence import create_mock_document

    with tempfile.TemporaryDirectory() as tmp:
        image = create_mock_document(os.path.join(tmp, "scan.png"))
        inspector = DocumentInspector(artifacts_dir=tmp, backend="auto")
        inspector.vlm_api_key = ""

        original = di._tesseract_available
        di._tesseract_available = lambda: False          # simulate a machine without it
        try:
            inspector._resolve_backend(image)
        except DocumentIntelligenceError as exc:
            message = str(exc)
            # Must name the fix, and must not offer mock as a way out.
            assert "pytesseract" in message and "tesseract-ocr" in message
            assert "SWARN_VLM_API_KEY" in message
            assert "must not be used" in message
        else:
            raise AssertionError("a document with no usable backend must raise")
        finally:
            di._tesseract_available = original


def test_explicit_mock_backend_still_works_on_a_real_document():
    # Explicitly asking for synthetic data is legitimate (demos, fixtures) —
    # it just must never happen by default.
    if not _can_make_pdf():
        return _skip("needs pydantic + Pillow + reportlab + pdfplumber")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "consignment.pdf")
        _fixture_pdf(pdf)
        result = DocumentInspector(artifacts_dir=tmp, backend="mock").process_document(pdf)
        assert result.backend == "mock"
        assert result.field_map()["invoice_number"] == "INV-2024-0917"


def test_tool_payload_flags_synthetic_results():
    if not _can_make_pdf():
        return _skip("needs pydantic + Pillow + reportlab + pdfplumber")
    from swarn.capabilities.doc_intelligence import inspect_document

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "consignment.pdf")
        _fixture_pdf(pdf)

        synthetic = inspect_document(pdf, backend="mock", annotate=False)
        assert synthetic["synthetic"] is True
        assert "SYNTHETIC" in synthetic["warning"]

        real = inspect_document(pdf, backend="text", annotate=False)
        assert "synthetic" not in real and real["backend"] == "text"


def test_extracted_values_are_all_present_in_the_real_document():
    # The core honesty property: every value reported must be text that is
    # actually on the page. No invented fields, no plausible fabrications.
    if not _can_make_pdf():
        return _skip("needs pydantic + Pillow + reportlab + pdfplumber")
    import pdfplumber

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "consignment.pdf")
        _fixture_pdf(pdf)
        result = DocumentInspector(artifacts_dir=tmp, backend="text").process_document(
            pdf, annotate=False)

        with pdfplumber.open(pdf) as handle:
            page_text = " ".join((handle.pages[0].extract_text() or "").split())

        assert result.fields, "the fixture has extractable content"
        for field in result.fields:
            assert str(field.field_value) in page_text, (
                f"{field.field_name}={field.field_value!r} is not on the page")


def test_known_fixture_fields_are_recovered_with_real_boxes():
    if not _can_make_pdf():
        return _skip("needs pydantic + Pillow + reportlab + pdfplumber")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "consignment.pdf")
        _fixture_pdf(pdf)
        result = DocumentInspector(artifacts_dir=tmp, backend="text").process_document(
            pdf, annotate=False)
        values = " ".join(str(f.field_value) for f in result.fields)

        # The labelled and stacked rules should both have fired.
        assert "QLX-5591-KT" in values          # "Docket No: ..."  (labelled)
        assert "Zephyr Northbound Route" in values   # stacked under its caption
        assert "ops.desk@quartzlogistics.example" in values   # email pattern

        # Boxes are real: distinct per field, inside the page, non-degenerate.
        boxes = [(f.box.xmin, f.box.ymin, f.box.xmax, f.box.ymax) for f in result.fields]
        assert len(set(boxes)) == len(boxes), "every field must have its own box"
        for box in [f.box for f in result.fields]:
            assert 0.0 <= box.xmin < box.xmax <= 1.0
            assert 0.0 <= box.ymin < box.ymax <= 1.0

        # And they are NOT the mock's hard-coded coordinates.
        mock_boxes = {(f.box.xmin, f.box.ymin, f.box.xmax, f.box.ymax) for f in mock_fields()}
        assert not (set(boxes) & mock_boxes)


def test_annotated_image_for_a_real_pdf_is_named_after_the_document():
    if not _can_make_pdf():
        return _skip("needs pydantic + Pillow + reportlab + pdfplumber")
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "consignment.pdf")
        _fixture_pdf(pdf)
        result = DocumentInspector(artifacts_dir=tmp).process_document(pdf, page_number=1)

        name = os.path.basename(result.annotated_image_path)
        assert name == "consignment_p1_annotated.png"
        assert "annotated_doc_sample" not in result.annotated_image_path
        # Rendered from the actual page, so it has the page's aspect ratio.
        with Image.open(result.annotated_image_path) as img:
            assert img.height > img.width          # the fixture is A4 portrait


def test_mock_leak_guard_rejects_synthetic_values_from_a_real_backend():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    leaked = mock_fields()

    # 1. wholesale mock geometry coming back from a real backend
    try:
        _assert_no_mock_leak(leaked, "text")
    except DocumentIntelligenceError as exc:
        assert "synthetic" in str(exc).lower()
    else:
        raise AssertionError("wholesale mock output must not pass the leak guard")

    # 2. the extractor's own provenance tag
    try:
        _assert_no_mock_leak([], "text", source="mock")
    except DocumentIntelligenceError as exc:
        assert "synthetic mock extractor" in str(exc)
    else:
        raise AssertionError("mock-tagged output must not pass the leak guard")

    # A stray coordinate collision must NOT trip it, and — the case that
    # matters in practice — OCR reading the mock invoice image is a correct
    # read of a real page, not a leak.
    _assert_no_mock_leak(leaked[:2], "ocr")


# ─────────────────────────────── entity extraction rules


def _w(text, left, top, width=None, height=10, size=None, conf=1.0):
    return {"text": text, "left": float(left), "top": float(top),
            "width": float(width if width is not None else len(text) * 6),
            "height": float(height), "conf": conf, "size": float(size or height)}


def test_entities_come_only_from_supplied_words():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    words = [_w("Docket", 10, 10), _w("No:", 50, 10), _w("QLX-5591-KT", 80, 10)]
    fields, _ = extract_entities(words, 600, 800)
    page_text = " ".join(w["text"] for w in words)
    assert fields
    for field in fields:
        assert str(field.field_value) in page_text


def test_reference_number_patterns_do_not_match_ordinary_words():
    # Regression: re.I on the whole pattern made "powered" a po_number and
    # "billed" an invoice_number — a real value under a fabricated name.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    words = [_w("servers", 10, 10), _w("are", 60, 10), _w("already", 90, 10),
             _w("bought,", 140, 10), _w("powered", 200, 10), _w("and", 260, 10),
             _w("billed", 290, 10), _w("points", 340, 10)]
    fields, _ = extract_entities(words, 600, 800)
    assert not [f for f in fields if f.field_name.startswith(("po_number", "invoice_number"))]


def test_reference_number_patterns_still_match_real_identifiers():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    for text, expected in (("INV-2024-0917", "invoice_number"), ("PO-77120", "po_number")):
        fields, _ = extract_entities([_w(text, 10, 10)], 600, 800)
        assert any(f.field_name == expected and f.field_value == text for f in fields), text


def test_pattern_box_covers_only_the_matched_span():
    # The box must wrap the VALUE, not the whole line it was found on.
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    words = [_w("Contact", 0, 10, width=48), _w("us", 52, 10, width=14),
             _w("at", 70, 10, width=14), _w("ops@quartz.example", 90, 10, width=110)]
    fields, _ = extract_entities(words, 600, 800)
    email = next(f for f in fields if f.field_name == "email")
    assert email.box.xmin >= 88 / 600 - 1e-6      # starts at the address, not at "Contact"
    assert email.box.xmax <= 201 / 600 + 1e-6


def test_stacked_caption_pairs_with_the_line_below_it_in_its_own_column():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    # Left column caption+value, with an unrelated right-column line in
    # between in reading order — the pair must still be found.
    segments = [
        [_w("TEAM NAME", 20, 10, width=60, height=8, size=8)],
        [_w("Right column text", 400, 12, width=110, height=11, size=11)],
        [_w("Cloud Catalyst", 20, 26, width=80, height=11, size=11)],
    ]
    pairs = _stacked_pairs(segments, page_width=600)
    assert len(pairs) == 1
    assert pairs[0][1][0]["text"] == "Cloud Catalyst"


def test_a_list_of_similar_items_does_not_pair_row_with_row():
    # Regression: a roster of Title Case names made each name the label of the
    # next ("anupriya_raj" = "Edunoori Spoorthi").
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    segments = [
        [_w("Anupriya Raj", 20, 10, width=70, height=11, size=11)],
        [_w("Edunoori Spoorthi", 20, 24, width=90, height=11, size=11)],
        [_w("Sargam Maurya", 20, 38, width=80, height=11, size=11)],
    ]
    assert _stacked_pairs(segments, page_width=600) == []


def test_duplicate_names_are_numbered_so_none_is_lost():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    words = [_w("₹2.69", 10, 10), _w("₹4.43", 10, 40), _w("₹1,200", 10, 70)]
    fields, _ = extract_entities(words, 600, 800)
    names = [f.field_name for f in fields if f.field_name.startswith("amount")]
    assert names == ["amount_1", "amount_2", "amount_3"]
    assert len({f.field_value for f in fields}) == len(fields)


def test_group_lines_merges_by_vertical_overlap_not_exact_top():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    lines = group_lines([_w("A", 0, 10, height=10), _w("B", 20, 12, height=10),
                          _w("C", 0, 60, height=10)])
    assert [[w["text"] for w in line] for line in lines] == [["A", "B"], ["C"]]


def test_snake_case_normalizes_printed_labels():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    assert _snake_case("Invoice No.") == "invoice_no"
    assert _snake_case("GST @ 18%") == "gst_18"
    assert _snake_case("TOTAL DUE (INR)") == "total_due_inr"


def test_ocr_backend_on_a_generated_image_returns_only_real_values():
    if not (_HAVE_PYDANTIC and _HAVE_PIL):
        return _skip("needs pydantic + Pillow")
    from swarn.capabilities.doc_intelligence import _tesseract_available, create_mock_document
    if not _tesseract_available():
        return _skip("needs the tesseract binary")

    with tempfile.TemporaryDirectory() as tmp:
        # The mock invoice is a convenient image whose true contents we know.
        image = create_mock_document(os.path.join(tmp, "scan.png"))
        result = DocumentInspector(artifacts_dir=tmp, backend="ocr").process_document(
            image, annotate=False)

        assert result.backend == "ocr"
        # Boxes come from tesseract, so they are not the layout's own numbers.
        mock_boxes = {(f.box.xmin, f.box.ymin, f.box.xmax, f.box.ymax) for f in mock_fields()}
        ocr_boxes = {(f.box.xmin, f.box.ymin, f.box.xmax, f.box.ymax) for f in result.fields}
        assert not (ocr_boxes & mock_boxes)
        # Transcription confidence is folded in, so nothing claims mock-level
        # certainty about a rasterized read.
        assert all(f.confidence <= 0.96 for f in result.fields)


def test_tool_definition_is_exported_to_the_agent_schema():
    if not _HAVE_PYDANTIC:
        return _skip("needs pydantic")
    from agent.tools import get_tool_definitions

    definition = next(
        (d for d in get_tool_definitions() if d["name"] == "swarn_doc_inspect"), None)
    assert definition is not None
    assert "path" in definition["input_schema"]["properties"]
    assert definition["input_schema"]["required"] == ["path"]
