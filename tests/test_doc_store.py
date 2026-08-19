"""
Tests for swarn/capabilities/doc_store.py — parse once, ask many times.

The property that matters most here is not "the cache works" but "the cache is
a complete substitute for the source". A store that answered questions but lost
coordinates, or served text from a superseded revision of a file, would be
worse than no cache at all — so most of these assert on what SURVIVES the round
trip and on when the store is deliberately bypassed.
"""

import json
import os
import tempfile


def _have(*mods) -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in mods)


_HAVE_DEPS = _have("pydantic", "PIL", "pdfplumber")
_HAVE_PDF = _HAVE_DEPS and _have("reportlab")

if _HAVE_DEPS:
    import swarn.capabilities.doc_intelligence as di
    from swarn.capabilities.doc_intelligence import DocumentInspector, DocumentIntelligenceError
    from swarn.capabilities.doc_qa import ask_document
    from swarn.capabilities.doc_store import (
        SCHEMA_VERSION, StoredDocument, document_id_for, file_sha256, get_or_ingest,
        ingest_document, list_documents, load_document, load_for_file, save_document,
        store_path,
    )


def _skip(reason: str) -> bool:
    print(f"    (skipped: {reason})")
    return True


def _financials_pdf(dest: str, fy24_revenue: str = "148200") -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                     Spacer, Table, TableStyle)

    ss = getSampleStyleSheet()
    table = Table([["Metric", "FY23", "FY24"],
                   ["Revenue", "120100", fy24_revenue],
                   ["Operating cost", "91400", "102300"]], colWidths=[150, 90, 90])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black)]))

    SimpleDocTemplate(dest, pagesize=A4).build([
        Paragraph("Northwind Analytics — Annual Report", ss["Title"]),
        PageBreak(),
        Paragraph("Segment Notes", ss["Heading1"]),
        PageBreak(),
        Paragraph("Financial Summary (INR thousands)", ss["Heading1"]),
        Spacer(1, 12), table,
    ])


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompt = None

    def complete(self, system, prompt, **kwargs):
        self.prompt = prompt
        return json.dumps(self.payload)


REVENUE_PAYLOAD = {
    "found": True,
    "answer": "Revenue increased by 23.4% from FY23 to FY24.",
    "computation": "(148200 - 120100) / 120100 = 23.4%",
    "evidence": [
        {"line_id": "p3:L3", "label": "FY23 revenue", "quote": "120100"},
        {"line_id": "p3:L3", "label": "FY24 revenue", "quote": "148200"},
    ],
}


class _CountingInspector(DocumentInspector):
    """Records every text-extraction call, so 'did this re-parse?' is a fact
    rather than an inference from timing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractions = []

    def page_words(self, file_path, page_number=1, backend=None):
        self.extractions.append(page_number)
        return super().page_words(file_path, page_number, backend)


# ─────────────────────────────── 1. ingestion


def test_ingest_produces_the_documented_schema():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        document = ingest_document(pdf, artifacts_dir=tmp)

        assert document.schema_version == SCHEMA_VERSION
        assert document.document_name == "financials.pdf"
        assert document.page_count == 3
        assert document.backend == "text"
        assert document.source_sha256 == file_sha256(pdf)
        assert document.ingested_at.endswith("+00:00")      # UTC, unambiguous

        page = document.page(3)
        assert page.width > 0 and page.height > 0
        assert any("Revenue | 120100 | 148200" == line.text for line in page.lines)
        assert all(line.line_id.startswith("p3:L") for line in page.lines)
        assert document.n_words() > 0


def test_ingest_writes_json_to_the_documents_directory():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        document = ingest_document(pdf, artifacts_dir=tmp)

        path = store_path(document.document_id, tmp)
        assert path.exists()
        assert path.parent.name == "documents"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["document_id"] == document.document_id
        assert payload["pages"][0]["lines"][0]["words"][0]["text"]


def test_tables_are_captured_when_the_extractor_finds_them():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        document = ingest_document(pdf, artifacts_dir=tmp)

        tables = [table for page in document.pages for table in page.tables]
        assert tables, "the ruled financial table should be detected"
        # Tables are now rows of CELLS, each with its own box.
        assert any("120100" in " ".join(sum(table.text_rows(), [])) for table in tables)
        cells = [cell for table in tables for row in table.rows for cell in row.cells]
        assert any(cell.text == "120100" and len(cell.bbox) == 4 for cell in cells)


# ─────────────────────────────── 2/3. round trip fidelity


def test_stored_document_loads_back_identically():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        original = ingest_document(pdf, artifacts_dir=tmp)
        reloaded = load_document(original.document_id, tmp)

        assert reloaded is not None
        assert reloaded.model_dump(mode="json") == original.model_dump(mode="json")


def test_word_coordinates_survive_serialization_exactly():
    # Coordinates ARE the feature. A float that drifts through JSON would move
    # every evidence box on every future answer.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        original = ingest_document(pdf, artifacts_dir=tmp)
        reloaded = load_document(original.document_id, tmp)

        before = [w for p in original.pages for l in p.lines for w in l.words]
        after = [w for p in reloaded.pages for l in p.lines for w in l.words]
        assert len(before) == len(after) and before

        for a, b in zip(before, after):
            assert (a.text, a.left, a.top, a.width, a.height) == \
                   (b.text, b.left, b.top, b.width, b.height)
            assert a.confidence == b.confidence
        # Page dimensions too — normalization is meaningless without them.
        assert [(p.width, p.height) for p in original.pages] == \
               [(p.width, p.height) for p in reloaded.pages]


def test_line_ids_and_transcript_are_stable_across_reloads():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        original = ingest_document(pdf, artifacts_dir=tmp)
        reloaded = load_document(original.document_id, tmp)

        for page in original.pages:
            assert page.render() == reloaded.page(page.page_number).render()
        # A citation resolves to the same words either way.
        assert original.page(3).find("p3:L3") == reloaded.page(3).find("p3:L3")


def test_unreadable_or_outdated_store_returns_none_rather_than_raising():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")

    with tempfile.TemporaryDirectory() as tmp:
        path = store_path("broken-000000000000", tmp)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert load_document("broken-000000000000", tmp) is None

        stale = store_path("stale-000000000000", tmp)
        stale.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 99}), encoding="utf-8")
        assert load_document("stale-000000000000", tmp) is None

        assert load_document("never-ingested-abc", tmp) is None


# ─────────────────────────────── identity


def test_document_id_follows_content_not_path():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        first = os.path.join(tmp, "a.pdf")
        copy = os.path.join(tmp, "renamed_copy.pdf")
        _financials_pdf(first)
        with open(first, "rb") as src, open(copy, "wb") as dst:
            dst.write(src.read())

        # Same bytes, different name: the hash half matches, so the same
        # content is recognised regardless of where it came from.
        assert document_id_for(first).split("-")[-1] == document_id_for(copy).split("-")[-1]

        edited = os.path.join(tmp, "edited.pdf")
        _financials_pdf(edited, fy24_revenue="999999")
        assert document_id_for(edited) != document_id_for(first)


def test_an_edited_document_is_re_ingested_not_served_stale():
    # The failure this prevents: confident citations into a revision of the
    # file that no longer exists.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        first, ingested = get_or_ingest(pdf, artifacts_dir=tmp)
        assert ingested is True

        again, ingested = get_or_ingest(pdf, artifacts_dir=tmp)
        assert ingested is False and again.document_id == first.document_id

        _financials_pdf(pdf, fy24_revenue="999999")          # same path, new content
        after, ingested = get_or_ingest(pdf, artifacts_dir=tmp)
        assert ingested is True
        assert after.document_id != first.document_id
        assert "Revenue | 120100 | 999999" in after.page(3).render()


# ─────────────────────────────── 5/6. ask reuses the store


def test_ask_ingests_once_then_never_re_extracts():
    # The headline requirement. Counting extraction calls, not timing.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        inspector = _CountingInspector(artifacts_dir=tmp)

        first = ask_document(pdf, "revenue?", client=FakeLLM(REVENUE_PAYLOAD),
                             annotate=False, inspector=inspector)
        assert first.from_store is False               # ingested on this call
        assert len(inspector.extractions) == 3         # one per page, once

        inspector.extractions.clear()
        for _ in range(3):
            result = ask_document(pdf, "revenue again?", client=FakeLLM(REVENUE_PAYLOAD),
                                  annotate=False, inspector=inspector)
            assert result.from_store is True
            assert result.document_id == first.document_id
        assert inspector.extractions == [], "the PDF must not be re-parsed"


def test_ask_after_explicit_ingest_does_no_extraction_at_all():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        ingest_document(pdf, artifacts_dir=tmp)

        inspector = _CountingInspector(artifacts_dir=tmp)
        result = ask_document(pdf, "revenue?", client=FakeLLM(REVENUE_PAYLOAD),
                              annotate=False, inspector=inspector)
        assert inspector.extractions == []
        assert result.from_store is True


def test_the_transcript_sent_to_the_model_comes_from_the_store():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        ingest_document(pdf, artifacts_dir=tmp)

        model = FakeLLM(REVENUE_PAYLOAD)
        ask_document(pdf, "revenue?", client=model, annotate=False,
                     inspector=DocumentInspector(artifacts_dir=tmp))
        assert "[p3:L3] Revenue | 120100 | 148200" in model.prompt


# ─────────────────────────────── 7/8/9. existing behaviour preserved


def test_revenue_answer_evidence_and_arithmetic_still_work_from_the_store():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        ingest_document(pdf, artifacts_dir=tmp)

        result = ask_document(pdf, "percentage increase in revenue?",
                              client=FakeLLM(REVENUE_PAYLOAD), annotate=False,
                              inspector=DocumentInspector(artifacts_dir=tmp))

        assert result.found is True
        assert result.computation_check == "verified"
        assert len(result.evidence) == 2
        assert all(span.verified for span in result.evidence)

        fy23, fy24 = result.evidence
        assert (fy23.quote, fy24.quote) == ("120100", "148200")
        # Boxes are still per-cell, from the stored word geometry.
        assert fy23.box.xmax <= fy24.box.xmin
        assert fy23.box.ymin == fy24.box.ymin
        for span in result.evidence:
            assert 0.0 <= span.box.xmin < span.box.xmax <= 1.0
            assert span.box.area < 0.05


def test_fabricated_quote_is_still_caught_when_answering_from_the_store():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        ingest_document(pdf, artifacts_dir=tmp)

        result = ask_document(pdf, "revenue?", annotate=False,
                              inspector=DocumentInspector(artifacts_dir=tmp),
                              client=FakeLLM({
                                  "found": True, "answer": "Revenue was 999999.",
                                  "computation": "",
                                  "evidence": [{"line_id": "p3:L3", "label": "revenue",
                                                "quote": "999999"}]}))
        assert len(result.unverified) == 1
        assert result.found is False


def test_evidence_image_still_renders_from_the_stored_document():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        ingest_document(pdf, artifacts_dir=tmp)

        result = ask_document(pdf, "revenue?", client=FakeLLM(REVENUE_PAYLOAD),
                              inspector=DocumentInspector(artifacts_dir=tmp))
        assert len(result.annotated_image_paths) == 1
        path = result.annotated_image_paths[0]
        assert os.path.basename(path) == "financials_p3_evidence.png"
        with Image.open(path) as img:
            assert img.height > img.width


def test_render_pages_makes_the_store_self_contained():
    # With cached rasters, evidence images can be drawn with the source file
    # gone — the only part of the pipeline that otherwise still needs it.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        document = ingest_document(pdf, artifacts_dir=tmp, render_pages=True)
        assert all(page.image_path and os.path.exists(page.image_path)
                   for page in document.pages)

        from swarn.capabilities.doc_qa import _page_image
        inspector = DocumentInspector(artifacts_dir=tmp)
        os.remove(pdf)                                   # source is gone
        image = _page_image(inspector, pdf, 3, document)
        assert image.size[0] > 0


# ─────────────────────────────── 10. missing / not-ingested behaviour


def test_missing_file_is_a_clear_error_from_every_entry_point():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")

    with tempfile.TemporaryDirectory() as tmp:
        for call in (lambda: ingest_document("/definitely/not/here.pdf", artifacts_dir=tmp),
                     lambda: load_for_file("/definitely/not/here.pdf", tmp),
                     lambda: get_or_ingest("/definitely/not/here.pdf", artifacts_dir=tmp)):
            try:
                call()
            except DocumentIntelligenceError as exc:
                assert "not found" in str(exc)
                continue
            raise AssertionError("a missing file must raise a clear error")


def test_not_yet_ingested_document_returns_none_and_then_auto_ingests():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)

        assert load_for_file(pdf, tmp) is None           # nothing stored yet
        announced = []
        document, ingested = get_or_ingest(
            pdf, artifacts_dir=tmp, on_ingest=announced.append)
        assert ingested is True and announced == [pdf]
        assert load_for_file(pdf, tmp).document_id == document.document_id


def test_listing_reports_stored_documents():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        assert list_documents(tmp) == []
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        document = ingest_document(pdf, artifacts_dir=tmp)

        listed = list_documents(tmp)
        assert len(listed) == 1
        assert listed[0]["document_id"] == document.document_id
        assert listed[0]["page_count"] == 3
        assert listed[0]["backend"] == "text"


# ─────────────────────────────── 4. CLI


def test_cli_ingest_and_list():
    if not (_HAVE_PDF and _have("typer")):
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab + typer")
    from typer.testing import CliRunner

    import swarn.capabilities.doc_store as store
    from agent.cli import app

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)

        original = store.ARTIFACTS_DIR
        store.ARTIFACTS_DIR = tmp          # keep the suite out of the repo's artifacts/
        try:
            runner = CliRunner()
            result = runner.invoke(app, ["ingest", pdf])
            assert result.exit_code == 0, result.output
            assert "ingested financials.pdf" in result.output
            assert "document_id" in result.output
            assert "documents" in result.output          # reports where it stored it

            # A second ingest recognises the existing entry instead of re-parsing.
            again = runner.invoke(app, ["ingest", pdf])
            assert again.exit_code == 0
            assert "already ingested" in again.output

            listing = runner.invoke(app, ["ingest", "--list"])
            assert listing.exit_code == 0
            assert "financials.pdf" in listing.output

            missing = runner.invoke(app, ["ingest", "/definitely/not/here.pdf"])
            assert missing.exit_code == 1
        finally:
            store.ARTIFACTS_DIR = original


def test_cli_ask_reports_the_document_it_answered_from():
    if not (_HAVE_PDF and _have("typer")):
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab + typer")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        ingest_document(pdf, artifacts_dir=tmp)

        result = ask_document(pdf, "revenue?", client=FakeLLM(REVENUE_PAYLOAD),
                              annotate=False,
                              inspector=DocumentInspector(artifacts_dir=tmp))
        summary = result.summary()
        assert "stored" in summary
        assert result.document_id in summary


def test_agent_tool_is_registered_and_reports_the_store():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")
    from agent.tools import TOOL_REGISTRY, get_tool_definitions

    assert "swarn_doc_ingest" in TOOL_REGISTRY
    definition = next(d for d in get_tool_definitions() if d["name"] == "swarn_doc_ingest")
    assert definition["input_schema"]["required"] == ["path"]

    from agent.tools import swarn_doc_ingest
    assert swarn_doc_ingest("/definitely/not/here.pdf").startswith("Error:")
