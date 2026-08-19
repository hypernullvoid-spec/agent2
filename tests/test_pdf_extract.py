"""
Tests for PDF → structured data extraction (extract_pdf_structured).

Split deliberately into two groups:

  * The _table_to_structured / _clean_cell tests below run ALWAYS — they
    exercise the grid-normalization logic on plain Python lists, so they
    need neither reportlab nor pdfplumber and keep run_tests.py's
    zero-dependency promise intact.
  * The end-to-end tests build a real PDF with reportlab and read it back
    with pdfplumber. Both are optional deps, so those tests SKIP (print
    and return) rather than fail when either is missing — a machine
    without them shouldn't report a red suite for a feature it can't run.
"""

import json
import os
import tempfile

from agent.multimodal_rag import MultiModalIndexer


def _have(*mods) -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in mods)


def _make_pdf(dest: str) -> None:
    """A one-page invoice: prose + a GRID-ruled table pdfplumber can detect."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    data = [
        ["Line", "SKU", "Description", "Qty", "Amount"],
        ["1", "CMP-M4", "Compute node", "12", "100800"],
        ["2", "STO-SSD", "Block storage", "40", "50000"],
        ["", "", "Total due", "", "150800"],
    ]
    table = Table(data, colWidths=[40, 70, 180, 40, 80])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black)]))

    SimpleDocTemplate(dest, pagesize=letter).build([
        Paragraph("Invoice INV-2024-0917 for ESDS", styles["Title"]),
        Spacer(1, 12),
        table,
    ])


# ─────────────────────────────── always-run: grid normalization


def test_table_to_structured_uses_first_row_as_header():
    parsed = MultiModalIndexer._table_to_structured(
        [["Name", "Qty"], ["widget", "3"], ["bolt", "7"]], index=1)
    assert parsed["header"] == ["Name", "Qty"]
    assert parsed["rows"] == [["widget", "3"], ["bolt", "7"]]
    assert parsed["records"] == [{"Name": "widget", "Qty": "3"},
                                  {"Name": "bolt", "Qty": "7"}]
    assert parsed["n_rows"] == 2 and parsed["n_cols"] == 2


def test_blank_header_cell_keeps_every_row_and_emits_no_records():
    # A header with a hole can't key a dict without silently losing a
    # column, so the first row must stay in `rows` as data.
    parsed = MultiModalIndexer._table_to_structured(
        [["Name", ""], ["widget", "3"]], index=1)
    assert parsed["header"] is None
    assert parsed["rows"] == [["Name", ""], ["widget", "3"]]
    assert "records" not in parsed
    assert parsed["n_rows"] == 2


def test_duplicate_header_names_are_rejected():
    # zip() into a dict would collapse the two "Qty" columns into one.
    parsed = MultiModalIndexer._table_to_structured(
        [["Qty", "Qty"], ["1", "2"]], index=1)
    assert parsed["header"] is None
    assert "records" not in parsed


def test_short_rows_are_padded_to_full_width():
    parsed = MultiModalIndexer._table_to_structured(
        [["A", "B", "C"], ["1"]], index=1)
    assert parsed["n_cols"] == 3
    assert parsed["rows"] == [["1", "", ""]]


def test_none_cells_and_wrapped_text_are_normalized():
    parsed = MultiModalIndexer._table_to_structured(
        [["A", "B"], [None, "two\nlines"]], index=1)
    assert parsed["rows"] == [["", "two lines"]]


def test_fully_empty_table_returns_none():
    assert MultiModalIndexer._table_to_structured([[None, ""], ["", None]], 1) is None
    assert MultiModalIndexer._table_to_structured([], 1) is None


def test_missing_file_returns_error_dict_not_raise():
    result = MultiModalIndexer().extract_pdf_structured("/definitely/not/here.pdf")
    assert "error" in result and "not found" in result["error"]


# ─────────────────────────────── end-to-end: real PDF (optional deps)


def test_extract_real_pdf_end_to_end():
    if not _have("reportlab", "pdfplumber"):
        print("    (skipped: needs reportlab + pdfplumber)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "invoice.pdf")
        _make_pdf(pdf)

        result = MultiModalIndexer().extract_pdf_structured(pdf)
        assert "error" not in result, result
        assert result["n_pages"] == 1
        assert result["n_tables"] == 1

        page = result["pages"][0]
        assert page["page"] == 1
        assert "INV-2024-0917" in page["text"]      # prose survives alongside tables

        table = page["tables"][0]
        assert table["header"] == ["Line", "SKU", "Description", "Qty", "Amount"]
        assert table["records"][0]["SKU"] == "CMP-M4"
        assert table["records"][0]["Amount"] == "100800"
        assert table["records"][-1]["Description"] == "Total due"
        # The structure is the point: rows stay rows, not a flattened blob.
        assert all(len(r) == 5 for r in table["rows"])


def test_tool_wrapper_returns_json_and_honors_filters():
    if not _have("reportlab", "pdfplumber"):
        print("    (skipped: needs reportlab + pdfplumber)")
        return
    from agent.tools import extract_pdf_structured

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "invoice.pdf")
        _make_pdf(pdf)

        data = json.loads(extract_pdf_structured(pdf))
        assert data["pages"][0]["text"]

        tables_only = json.loads(extract_pdf_structured(pdf, tables_only=True))
        assert "text" not in tables_only["pages"][0]
        assert tables_only["pages"][0]["tables"]

        # Out-of-range page is an error string, not an exception or an
        # empty-but-successful result.
        assert extract_pdf_structured(pdf, page=99).startswith("Error:")


def test_tool_wrapper_reports_missing_file_as_error_string():
    from agent.tools import extract_pdf_structured
    assert extract_pdf_structured("/definitely/not/here.pdf").startswith("Error:")


# ────────────────── whole-document structuring (extract_pdf_document)


def _make_report_pdf(dest: str) -> None:
    """Title, h1/h2 headings, key-value lines, paragraphs, a bulleted list,
    a numbered list, and a ruled table across two pages."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, PageBreak)

    ss = getSampleStyleSheet()
    body = ss["BodyText"]
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12)

    table = Table([["Service", "Q2 Cost", "Q3 Cost"],
                    ["Compute", "412000", "398500"],
                    ["Storage", "180400", "141000"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black)]))

    SimpleDocTemplate(dest, pagesize=letter).build([
        Paragraph("Quarterly Infrastructure Review", ss["Title"]),
        Paragraph("Document Details", h1),
        Paragraph("Report ID: QIR-2024-Q3", body),
        Paragraph("Status: Approved", body),
        Paragraph("Executive Summary", h1),
        Paragraph("Cluster utilisation rose while unit cost per workload fell. "
                   "The storage migration finished ahead of plan.", body),
        Paragraph("Key Findings", h2),
        Paragraph("• Compute utilisation averaged 71%", body),
        Paragraph("• Egress charges fell by 22%", body),
        Paragraph("Recommendations", h2),
        Paragraph("1. Promote the standby control plane", body),
        Paragraph("2. Retire two idle nodes", body),
        PageBreak(),
        Paragraph("Cost Breakdown", h1),
        Spacer(1, 8),
        table,
    ])


def _blocks(doc, kind):
    return [b for s in doc["sections"] for b in s["blocks"] if b["type"] == kind]


def test_document_recovers_headings_sections_and_levels():
    if not _have("reportlab", "pdfplumber"):
        print("    (skipped: needs reportlab + pdfplumber)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "report.pdf")
        _make_report_pdf(pdf)
        doc = MultiModalIndexer().extract_pdf_document(pdf)
        assert "error" not in doc, doc

        assert doc["title"] == "Quarterly Infrastructure Review"
        headings = [s["heading"] for s in doc["sections"]]
        for expected in ["Document Details", "Executive Summary", "Key Findings",
                         "Recommendations", "Cost Breakdown"]:
            assert expected in headings, headings

        # h2s must nest deeper than the h1s around them.
        levels = {s["heading"]: s["level"] for s in doc["sections"]}
        assert levels["Key Findings"] > levels["Executive Summary"]
        assert levels["Cost Breakdown"] == levels["Executive Summary"]


def test_document_types_lists_paragraphs_and_key_values():
    if not _have("reportlab", "pdfplumber"):
        print("    (skipped: needs reportlab + pdfplumber)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "report.pdf")
        _make_report_pdf(pdf)
        doc = MultiModalIndexer().extract_pdf_document(pdf)

        # Adjacent bullets form ONE list, with markers stripped — spacing
        # between items must not split them into separate blocks.
        lists = _blocks(doc, "list")
        bullets = [l for l in lists if not l["ordered"]]
        numbered = [l for l in lists if l["ordered"]]
        assert len(bullets) == 1 and len(bullets[0]["items"]) == 2
        assert bullets[0]["items"][0].startswith("Compute utilisation")
        assert len(numbered) == 1 and len(numbered[0]["items"]) == 2
        assert numbered[0]["items"][0] == "Promote the standby control plane"

        kvs = _blocks(doc, "key_values")
        assert any(b["fields"].get("Report ID") == "QIR-2024-Q3" for b in kvs)
        assert any(b["fields"].get("Status") == "Approved" for b in kvs)

        # Wrapped body lines rejoin into a single paragraph.
        paras = _blocks(doc, "paragraph")
        assert any("finished ahead of plan" in p["text"] and
                    "Cluster utilisation rose" in p["text"] for p in paras)


def test_document_field_map_and_table_are_populated():
    if not _have("reportlab", "pdfplumber"):
        print("    (skipped: needs reportlab + pdfplumber)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "report.pdf")
        _make_report_pdf(pdf)
        doc = MultiModalIndexer().extract_pdf_document(pdf)

        assert doc["fields"]["Report ID"] == "QIR-2024-Q3"
        assert doc["fields"]["Status"] == "Approved"

        tables = _blocks(doc, "table")
        assert len(tables) == 1
        assert tables[0]["header"] == ["Service", "Q2 Cost", "Q3 Cost"]
        assert tables[0]["page"] == 2
        assert doc["counts"]["tables"] == 1


def test_table_text_is_not_duplicated_into_paragraphs():
    if not _have("reportlab", "pdfplumber"):
        print("    (skipped: needs reportlab + pdfplumber)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "report.pdf")
        _make_report_pdf(pdf)
        doc = MultiModalIndexer().extract_pdf_document(pdf)

        # Table cells are excluded from the text layer by bounding box, so
        # a cell value must not resurface as prose.
        prose = " ".join(p["text"] for p in _blocks(doc, "paragraph"))
        assert "398500" not in prose
        assert "Compute 412000" not in prose


def test_prose_embedded_fields_are_mined():
    # "Label: value" pairs inside a flowing sentence are where invoice-style
    # data usually lives — line-based detection alone would miss them.
    mined = MultiModalIndexer._mine_prose_fields(
        "Billed to: ESDS Ltd. Invoice date: 2024-09-17. Terms: Net 30.")
    assert mined["Invoice date"] == "2024-09-17"
    assert mined["Terms"] == "Net 30"


def test_prose_without_fields_yields_nothing():
    # A colon inside ordinary prose must not manufacture a field.
    assert MultiModalIndexer._mine_prose_fields(
        "The team met on Tuesday and agreed to proceed.") == {}
    assert MultiModalIndexer._mine_prose_fields(
        "There was one conclusion: the rollout should continue through the "
        "next quarter. Nobody objected to that plan.") == {}


def test_cid_bullet_glyph_is_treated_as_a_list_marker():
    # An unmapped bullet glyph extracts as the literal "(cid:127)". Without
    # special-casing it, the line parses as a field named "(cid".
    assert MultiModalIndexer._line_kind("(cid:127) Compute utilisation rose") == "list"
    assert MultiModalIndexer._kv_match("(cid:127) Compute utilisation rose") is None


def test_kv_match_rejects_prose_and_junk_labels():
    assert MultiModalIndexer._kv_match("Status: Approved") == ("Status", "Approved")
    # Value that runs into a new sentence is prose, not a field.
    assert MultiModalIndexer._kv_match(
        "Note: the migration finished early. Nobody objected.") is None
    # Label too wordy to be a field name.
    assert MultiModalIndexer._kv_match(
        "One thing that everybody on the team agreed about: proceed") is None
    assert MultiModalIndexer._kv_match("Status:") is None


def test_document_reports_missing_file_as_error():
    result = MultiModalIndexer().extract_pdf_document("/definitely/not/here.pdf")
    assert "error" in result and "not found" in result["error"]


def test_document_tool_wrapper_json_and_page_filter():
    if not _have("reportlab", "pdfplumber"):
        print("    (skipped: needs reportlab + pdfplumber)")
        return
    from agent.tools import extract_pdf_document

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "report.pdf")
        _make_report_pdf(pdf)

        doc = json.loads(extract_pdf_document(pdf))
        assert doc["counts"]["sections"] >= 5

        page2 = json.loads(extract_pdf_document(pdf, page=2))
        pages_present = {b["page"] for s in page2["sections"] for b in s["blocks"]}
        assert pages_present == {2}
        assert page2["counts"]["tables"] == 1

        assert extract_pdf_document(pdf, page=99).startswith("Error:")
