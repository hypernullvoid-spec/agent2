"""
Tests for swarn/capabilities/doc_qa.py — grounded document Q&A (`swarn ask`).

Every test here injects a fake LLM client, so the suite never touches the
network and the "model" can be made to misbehave on demand — which is the
point: the interesting behaviour of this module is what it does when the model
fabricates a citation or gets the arithmetic wrong, and that cannot be tested
with a well-behaved real model.

The document fixture is a small reportlab PDF with a known financial table, so
the expected answers, quotes, and coordinates are all knowable in advance.
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
    from swarn.capabilities.doc_intelligence import DocumentInspector
    from swarn.capabilities.doc_qa import (
        DocumentQAError, PageTranscript, ask_document, build_transcript,
        resolve_evidence, safe_eval, select_pages, verify_computation,
    )


def _skip(reason: str) -> bool:
    print(f"    (skipped: {reason})")
    return True


class FakeLLM:
    """Returns a scripted payload, and records the prompt it was given so tests
    can assert on what the model was actually allowed to see."""

    def __init__(self, payload, raw: str = None):
        self.payload = payload
        self.raw = raw
        self.system = None
        self.prompt = None

    def complete(self, system, prompt, **kwargs):
        self.system, self.prompt = system, prompt
        return self.raw if self.raw is not None else json.dumps(self.payload)


def _financials_pdf(dest: str) -> None:
    """Page 1 title, page 2 filler, page 3 the table the questions are about."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                     Spacer, Table, TableStyle)

    ss = getSampleStyleSheet()
    table = Table([["Metric", "FY23", "FY24"],
                   ["Revenue", "120100", "148200"],
                   ["Operating cost", "91400", "102300"],
                   ["Headcount", "240", "268"]], colWidths=[150, 90, 90])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black)]))

    SimpleDocTemplate(dest, pagesize=A4).build([
        Paragraph("Northwind Analytics — Annual Report", ss["Title"]),
        PageBreak(),
        Paragraph("Segment Notes", ss["Heading1"]),
        Paragraph("Cloud services grew fastest.", ss["BodyText"]),
        PageBreak(),
        Paragraph("Financial Summary (INR thousands)", ss["Heading1"]),
        Spacer(1, 12), table,
    ])


GOOD_PAYLOAD = {
    "found": True,
    "answer": "Revenue increased by 23.4% from FY23 to FY24.",
    "computation": "(148200 - 120100) / 120100 = 23.4%",
    "evidence": [
        {"line_id": "p3:L3", "label": "FY23 revenue", "quote": "120100"},
        {"line_id": "p3:L3", "label": "FY24 revenue", "quote": "148200"},
    ],
}


# ─────────────────────────────── safe_eval


def test_safe_eval_computes_arithmetic():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    assert safe_eval("(148200 - 120100) / 120100 * 100") == 23.397169025811824
    assert safe_eval("-2 + 3 * 4") == 10.0


def test_safe_eval_refuses_anything_that_is_not_arithmetic():
    # This string comes from an LLM. Treating it as code would be an arbitrary
    # execution sink reachable from any document plus any question.
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    for hostile in ("__import__('os').system('id')", "open('/etc/passwd').read()",
                    "x + 1", "[].__class__", "(1).__class__.__bases__"):
        try:
            safe_eval(hostile)
        except (ValueError, SyntaxError, TypeError):
            continue
        raise AssertionError(f"safe_eval must not evaluate {hostile!r}")


# ─────────────────────────────── computation verification


def test_verified_and_mismatched_arithmetic():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    assert verify_computation("(148200 - 120100) / 120100 = 23.4%") == "verified"
    # Already-scaled left side must not be scaled again — an earlier version
    # reported this correct computation as "2,340%, not 23.4%".
    assert verify_computation("(148200 - 120100) / 120100 * 100 = 23.4%") == "verified"
    assert verify_computation("148200 - 120100 = 28100") == "verified"
    assert verify_computation(
        "(148200 - 120100) / 120100 * 100 = 31").startswith("MISMATCH")


def test_chained_equalities_are_accepted():
    # "expr = ratio = percentage" — the same quantity in two units.
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    assert verify_computation(
        "(148200 - 120100) / 120100 = 0.23397 = 23.4%") == "verified"


def test_compound_computation_checks_every_part():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    good = ("(148200 - 102300) / 148200 = 0.3097 (30.97%); "
            "(120100 - 91400) / 120100 = 0.2390 (23.90%)")
    assert verify_computation(good) == "verified"
    bad = "(148200 - 102300) / 148200 = 0.99; (120100 - 91400) / 120100 = 0.2390"
    assert verify_computation(bad).startswith("MISMATCH")


def test_uncheckable_computation_is_silent_not_an_error():
    # A lookup answer needs no arithmetic; claiming "unverified" would be noise.
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    for nothing in ("", "revenue grew a lot", "see table", "__import__('os') = 1"):
        assert verify_computation(nothing) == ""


# ─────────────────────────────── transcript


def test_transcript_keeps_table_rows_intact_with_column_markers():
    # Splitting each cell onto its own line would destroy the row association
    # the model needs to tell FY23 from FY24.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        transcripts, backend = build_transcript(DocumentInspector(artifacts_dir=tmp), pdf, [3])

        assert backend == "text"
        rendered = transcripts[3].render()
        assert "Revenue | 120100 | 148200" in rendered
        assert "Operating cost | 91400 | 102300" in rendered
        # Line ids are stable and addressable.
        assert "[p3:L3] Revenue" in rendered


def test_page_selection_prefers_pages_matching_the_question():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        transcripts, _ = build_transcript(
            DocumentInspector(artifacts_dir=tmp), pdf, [1, 2, 3])

        # Whole document fits comfortably: everything is searched.
        assert select_pages(transcripts, "revenue") == [1, 2, 3]
        # Under a tight budget, the revenue page wins over the title page.
        assert select_pages(transcripts, "revenue", budget=120) == [3]


# ─────────────────────────────── evidence resolution (the anti-fabrication core)


def test_good_citations_resolve_to_tight_boxes():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        model = FakeLLM(GOOD_PAYLOAD)
        result = ask_document(pdf, "percentage increase in revenue?",
                              client=model, annotate=False,
                              inspector=DocumentInspector(artifacts_dir=tmp))

        assert result.found is True
        assert len(result.evidence) == 2
        assert all(span.verified for span in result.evidence)
        assert result.computation_check == "verified"

        fy23, fy24 = result.evidence
        assert (fy23.quote, fy24.quote) == ("120100", "148200")
        # Each box is the CELL, not the whole row: they must not overlap, and
        # neither may span the row label.
        assert fy23.box.xmax <= fy24.box.xmin
        assert fy23.box.xmin > 0.3          # past the "Revenue" label column
        assert fy23.box.ymin == fy24.box.ymin     # same row


def test_a_quote_not_on_the_cited_line_is_marked_unverified():
    # The central defence: the model cites a real line but invents the text.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        result = ask_document(pdf, "revenue?", annotate=False,
                              inspector=DocumentInspector(artifacts_dir=tmp),
                              client=FakeLLM({
                                  "found": True, "answer": "Revenue was 999999.",
                                  "computation": "",
                                  "evidence": [{"line_id": "p3:L3",
                                                "label": "revenue",
                                                "quote": "999999"}]}))

        assert len(result.unverified) == 1
        assert result.unverified[0].quote == "999999"
        # And an answer with nothing verified behind it is not reported as found.
        assert result.found is False
        assert "unsupported" in result.answer


def test_citation_of_a_nonexistent_line_is_dropped_entirely():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        result = ask_document(pdf, "revenue?", annotate=False,
                              inspector=DocumentInspector(artifacts_dir=tmp),
                              client=FakeLLM({
                                  "found": True, "answer": "Revenue was 500.",
                                  "computation": "",
                                  "evidence": [
                                      {"line_id": "p9:L99", "label": "x", "quote": "500"},
                                      {"line_id": "not-an-id", "label": "y", "quote": "500"},
                                  ]}))
        assert result.evidence == []
        assert result.found is False


def test_bracketed_line_ids_are_accepted():
    # The transcript displays "[p3:L3] ...", so models cite the bracketed form
    # constantly. Rejecting it dropped every citation from a correct answer.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        bracketed = json.loads(json.dumps(GOOD_PAYLOAD).replace('"p3:L3"', '"[p3:L3]"'))
        result = ask_document(pdf, "revenue?", client=FakeLLM(bracketed),
                              annotate=False,
                              inspector=DocumentInspector(artifacts_dir=tmp))
        assert len(result.evidence) == 2
        assert all(span.verified for span in result.evidence)


def test_whole_row_quote_still_verifies_despite_column_markers():
    # "Revenue | 120100 | 148200" quotes the renderer's own pipes, which appear
    # in no word on the page. Failing that would call a correct citation a lie.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        result = ask_document(pdf, "revenue?", annotate=False,
                              inspector=DocumentInspector(artifacts_dir=tmp),
                              client=FakeLLM({
                                  "found": True, "answer": "Revenue rose.",
                                  "computation": "",
                                  "evidence": [{"line_id": "p3:L3", "label": "row",
                                                "quote": "Revenue | 120100 | 148200"}]}))
        assert result.evidence[0].verified is True


def test_evidence_boxes_are_real_page_coordinates():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        result = ask_document(pdf, "revenue?", client=FakeLLM(GOOD_PAYLOAD),
                              annotate=False,
                              inspector=DocumentInspector(artifacts_dir=tmp))
        for span in result.evidence:
            box = span.box
            assert 0.0 <= box.xmin < box.xmax <= 1.0
            assert 0.0 <= box.ymin < box.ymax <= 1.0
            # A cell box is a small fraction of the page, not the whole thing.
            assert box.area < 0.05


# ─────────────────────────────── model-level failures


def test_found_false_is_reported_not_papered_over():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        result = ask_document(pdf, "what is the CEO's salary?", annotate=False,
                              inspector=DocumentInspector(artifacts_dir=tmp),
                              client=FakeLLM({
                                  "found": False,
                                  "answer": "The document does not state a salary.",
                                  "computation": "", "evidence": []}))
        assert result.found is False
        assert "does not state" in result.answer
        assert result.evidence == []


def test_non_json_model_reply_is_an_actionable_error():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        try:
            ask_document(pdf, "revenue?", annotate=False,
                         inspector=DocumentInspector(artifacts_dir=tmp),
                         client=FakeLLM(None, raw="Let me think about this at length"))
        except DocumentQAError as exc:
            assert "did not return usable JSON" in str(exc)
            assert "SWARN_DEPLOYED_MODEL" in str(exc)   # names the fix
            return
        raise AssertionError("a prose reply must raise a clear error")


def test_llm_failure_is_reported_never_answered_around():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    class Broken:
        def complete(self, *a, **kw):
            raise RuntimeError("connection refused")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        try:
            ask_document(pdf, "revenue?", client=Broken(), annotate=False,
                         inspector=DocumentInspector(artifacts_dir=tmp))
        except DocumentQAError as exc:
            assert "connection refused" in str(exc)
            return
        raise AssertionError("an LLM failure must surface, not produce an answer")


def test_the_model_only_ever_sees_the_transcript():
    # It is handed extracted words, not the file — so it cannot cite a page it
    # was never shown, and page restriction is a real restriction.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        model = FakeLLM(GOOD_PAYLOAD)
        ask_document(pdf, "revenue?", pages=[3], client=model, annotate=False,
                     inspector=DocumentInspector(artifacts_dir=tmp))

        assert "[p3:L3] Revenue | 120100 | 148200" in model.prompt
        assert "Northwind Analytics" not in model.prompt      # page 1 was excluded
        assert "JSON" in model.system


def test_empty_question_and_bad_page_are_rejected():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        inspector = DocumentInspector(artifacts_dir=tmp)
        for kwargs, expected in ((dict(question="  "), "empty"),
                                  (dict(question="revenue?", pages=[99]), "out of range")):
            try:
                ask_document(pdf, client=FakeLLM(GOOD_PAYLOAD), annotate=False,
                             inspector=inspector, **kwargs)
            except DocumentQAError as exc:
                assert expected in str(exc)
                continue
            raise AssertionError(f"expected an error mentioning {expected!r}")


# ─────────────────────────────── annotation + wiring


def test_evidence_image_is_written_from_the_real_page():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        result = ask_document(pdf, "revenue?", client=FakeLLM(GOOD_PAYLOAD),
                              inspector=DocumentInspector(artifacts_dir=tmp))

        assert len(result.annotated_image_paths) == 1
        path = result.annotated_image_paths[0]
        assert os.path.basename(path) == "financials_p3_evidence.png"
        assert os.path.exists(path)
        with Image.open(path) as img:
            assert img.height > img.width          # A4 portrait, the real page


def test_answer_question_wrapper_and_agent_tool():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")
    from agent.runtime.tools import TOOL_REGISTRY, get_tool_definitions

    assert "swarn_doc_ask" in TOOL_REGISTRY
    definition = next(d for d in get_tool_definitions() if d["name"] == "swarn_doc_ask")
    assert set(definition["input_schema"]["required"]) == {"path", "question"}

    from swarn.capabilities.doc_qa import answer_question
    payload = answer_question("/definitely/not/here.pdf", "revenue?")
    assert "error" in payload and "not found" in payload["error"]


def test_summary_flags_unverified_evidence_and_bad_arithmetic():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "financials.pdf")
        _financials_pdf(pdf)
        result = ask_document(pdf, "revenue?", annotate=False,
                              inspector=DocumentInspector(artifacts_dir=tmp),
                              client=FakeLLM({
                                  "found": True, "answer": "Revenue rose 31%.",
                                  "computation": "(148200 - 120100) / 120100 = 31%",
                                  "evidence": [
                                      {"line_id": "p3:L3", "label": "FY23", "quote": "120100"},
                                      {"line_id": "p3:L3", "label": "bogus", "quote": "777"},
                                  ]}))
        summary = result.summary()
        assert "NOT FOUND IN DOCUMENT" in summary
        assert "MISMATCH" in summary
        assert len(result.unverified) == 1
