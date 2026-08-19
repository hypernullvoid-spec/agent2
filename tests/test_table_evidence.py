"""
Tests for table-aware evidence resolution.

The bug these cover: the resolver was line-centric, so a citation a model
naturally writes for a table row — "<trigger> CC 5", which interleaves a cell,
a COLUMN HEADER, and a cell from another column — matched no single extracted
line and was reported `NOT FOUND ON THAT LINE`. The citation was correct; the
resolver's unit of search was wrong.

The risk in fixing that is over-correcting into fuzzy matching, which would
manufacture citations instead of catching them. So roughly half of these tests
assert on what must STILL be rejected: a real row with the wrong value, a real
trigger paired with a different row's number, an invented row, and a real row
with an invented claim appended.
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
        _consume_row, _tokens, _token_span, ask_document, resolve_evidence,
    )
    from swarn.capabilities.doc_store import (
        StoredCell, StoredRow, StoredTable, ingest_document,
    )


def _skip(reason: str) -> bool:
    print(f"    (skipped: {reason})")
    return True


# ─────────────────────────────── fixtures


def _bonus_table_pdf(dest: str) -> None:
    """A three-column table shaped like the one that exposed the bug: a long
    wrapping trigger column, a short numeric column, and a long description —
    plus ordinary prose above it, so the line path is exercised too."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                     TableStyle)

    ss = getSampleStyleSheet()
    body = ss["BodyText"]
    rows = [
        ["Earning Trigger", "CC", "How to Earn It"],
        [Paragraph("Fastest Quiz Responder Bonus", body), "5",
         Paragraph("First team to submit a correct quiz answer each day.", body)],
        [Paragraph("Cross-Functional Team Bonus (3+ departments)", body), "25",
         Paragraph("Form a team spanning three or more departments.", body)],
        [Paragraph("Night Owl Check-In (after 11 PM, max 3)", body), "10",
         Paragraph("Team posts a check-in after 11 PM during the sprint.", body)],
    ]
    table = Table(rows, colWidths=[170, 40, 220])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                                ("VALIGN", (0, 0), (-1, -1), "TOP")]))

    SimpleDocTemplate(dest, pagesize=A4).build([
        Paragraph("Gamification — CloudCoins and Bonuses", ss["Heading1"]),
        Paragraph("Teams accumulate CloudCoins throughout the sprint by "
                  "completing the triggers listed below.", body),
        Spacer(1, 14), table,
    ])


class FakeLLM:
    def __init__(self, evidence, answer="Bonuses are listed in the table."):
        self.evidence = evidence
        self.answer = answer

    def complete(self, system, prompt, **kwargs):
        return json.dumps({"found": True, "answer": self.answer,
                           "computation": "", "evidence": self.evidence})


def _ask(pdf, tmp, evidence):
    return ask_document(pdf, "what bonuses are available?",
                        client=FakeLLM(evidence), annotate=False,
                        inspector=DocumentInspector(artifacts_dir=tmp))


def _cite(quote, line_id="p1:L4", label="bonus"):
    return [{"line_id": line_id, "label": label, "quote": quote}]


def _synthetic_table() -> StoredTable:
    """A table built by hand, so the matcher can be tested without a PDF."""
    def cell(row, col, column, text, box):
        return StoredCell(column=column, row_index=row, col_index=col,
                          text=text, bbox=box)
    return StoredTable(index=1, page=1, bbox=[10, 10, 500, 200],
                        headers=["Earning Trigger", "CC", "How to Earn It"],
                        rows=[
        StoredRow(index=0, cells=[cell(0, 0, "Earning Trigger", "Earning Trigger", [10, 10, 200, 24]),
                                   cell(0, 1, "CC", "CC", [210, 10, 250, 24]),
                                   cell(0, 2, "How to Earn It", "How to Earn It", [260, 10, 500, 24])]),
        StoredRow(index=1, cells=[cell(1, 0, "Earning Trigger", "Fastest Quiz Responder Bonus", [10, 30, 200, 44]),
                                   cell(1, 1, "CC", "5", [210, 30, 250, 44]),
                                   cell(1, 2, "How to Earn It", "First team to submit a correct quiz answer each day.", [260, 30, 500, 60])]),
        StoredRow(index=2, cells=[cell(2, 0, "Earning Trigger", "Cross-Functional Team Bonus (3+ departments)", [10, 70, 200, 100]),
                                   cell(2, 1, "CC", "25", [210, 70, 250, 84]),
                                   cell(2, 2, "How to Earn It", "Form a team spanning three or more departments.", [260, 70, 500, 100])]),
    ])


def _match(quote, table=None):
    table = table or _synthetic_table()
    headers = [_tokens(h) for h in table.headers]
    for row in table.rows:
        cells = _consume_row(_tokens(quote), row, headers)
        if cells:
            return row, cells
    return None, None


# ─────────────────────────────── 1. existing prose behaviour is untouched


def test_ordinary_prose_still_resolves_on_its_own_line():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        document = ingest_document(pdf, artifacts_dir=tmp)

        line = next(l for p in document.pages for l in p.lines
                    if "accumulate CloudCoins" in l.text)
        result = _ask(pdf, tmp, _cite("Teams accumulate CloudCoins", line.line_id))

        span = result.evidence[0]
        assert span.verified is True
        assert span.strategy == "line", "prose must stay on the fast path"
        assert span.table is None


def test_a_single_table_cell_quoted_alone_resolves():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        result = _ask(pdf, tmp, _cite("Fastest Quiz Responder Bonus"))
        span = result.evidence[0]
        assert span.verified is True
        assert span.strategy in ("line", "multiline", "table")


# ─────────────────────────────── 3/4. multi-column evidence


def test_evidence_combining_two_columns_via_a_header_label():
    # THE regression: "<trigger> CC 5" is cell + column header + cell. It is on
    # no single line, and used to be reported NOT FOUND.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        result = _ask(pdf, tmp, _cite("Fastest Quiz Responder Bonus CC 5"))

        span = result.evidence[0]
        assert span.verified is True, "multi-column table evidence must resolve"
        assert span.strategy == "table"
        assert span.table["columns"] == ["Earning Trigger", "CC"]
        assert [cell["text"] for cell in span.table["cells"]] == \
               ["Fastest Quiz Responder Bonus", "5"]


def test_trigger_plus_numeric_cell_across_a_wrapped_cell():
    # The trigger cell wraps ("Cross-Functional Team Bonus (3+ departments)")
    # while the quote names only its opening words.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        result = _ask(pdf, tmp, _cite("Cross-Functional Team Bonus CC 25"))

        span = result.evidence[0]
        assert span.verified is True
        assert span.strategy == "table"
        assert "25" in [cell["text"] for cell in span.table["cells"]]


def test_multi_line_cell_text_is_matched_as_one_cell():
    # A cell whose text wraps must behave as a single value, not two halves.
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    row, cells = _match("Cross-Functional Team Bonus (3+ departments) CC 25")
    assert row is not None and row.index == 2
    assert [cell.text for cell in cells][0].startswith("Cross-Functional")


def test_multi_column_and_multi_line_row_together():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    row, cells = _match(
        "Cross-Functional Team Bonus CC 25 How to Earn It "
        "Form a team spanning three or more departments.")
    assert row is not None and row.index == 2
    assert len(cells) == 3, "all three columns should be matched"


def test_quote_lifted_from_the_middle_of_one_long_cell():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    row, cells = _match("submit a correct quiz answer")
    assert row is not None and row.index == 1
    assert cells[0].column == "How to Earn It"


# ─────────────────────────────── 7. fabrications still rejected


def test_genuinely_missing_evidence_is_still_not_found():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        result = _ask(pdf, tmp, _cite("Blockchain Integration Bonus CC 50"))
        span = result.evidence[0]
        assert span.verified is False
        assert span.strategy == "none"
        assert result.found is False           # nothing verified behind the answer


def test_real_row_with_a_wrong_value_is_rejected():
    # The dangerous near-miss: everything true except the number.
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    assert _match("Fastest Quiz Responder Bonus CC 500") == (None, None)


def test_values_from_different_rows_cannot_be_combined():
    # A real trigger paired with a real number from ANOTHER row must not pass:
    # cells have to come from one row for the row's claim to hold.
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    assert _match("Fastest Quiz Responder Bonus CC 25") == (None, None)


def test_invented_trigger_with_a_real_value_is_rejected():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    assert _match("Slowest Quiz Responder Bonus CC 5") == (None, None)


def test_a_real_row_with_an_invented_claim_appended_is_rejected():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    assert _match("Fastest Quiz Responder Bonus CC 5 guaranteed for every team") == (None, None)


def test_header_words_alone_cannot_carry_a_match():
    # Column names are not data; a string built only from them must not
    # validate against a data row.
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    table = _synthetic_table()
    headers = [_tokens(h) for h in table.headers]
    data_row = table.rows[1]
    assert _consume_row(_tokens("Earning Trigger CC How to Earn It"),
                        data_row, headers) is None


def test_a_one_token_quote_does_not_reach_the_table_strategy():
    # "5" would match a numeric cell in almost any table; too weak to identify
    # a row, so it stays on the line strategies.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        # Cited against a prose line that does not contain it.
        result = _ask(pdf, tmp, _cite("99999", "p1:L2"))
        assert result.evidence[0].strategy != "table"


# ─────────────────────────────── 8. bounding boxes


def test_bounding_box_is_the_union_of_the_matched_cells():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        document = ingest_document(pdf, artifacts_dir=tmp)
        page = document.page(1)
        width, height = page.page_size

        result = _ask(pdf, tmp, _cite("Fastest Quiz Responder Bonus CC 5"))
        span = result.evidence[0]
        boxes = [cell["bbox"] for cell in span.table["cells"]]

        expected = (min(b[0] for b in boxes) / width, min(b[1] for b in boxes) / height,
                    max(b[2] for b in boxes) / width, max(b[3] for b in boxes) / height)
        for got, want in zip((span.box.xmin, span.box.ymin, span.box.xmax, span.box.ymax),
                             expected):
            assert abs(got - want) < 1e-6

        # The union must be wider than either cell alone — it spans two columns.
        trigger_width = (boxes[0][2] - boxes[0][0]) / width
        assert span.box.xmax - span.box.xmin > trigger_width
        assert 0.0 <= span.box.xmin < span.box.xmax <= 1.0
        assert 0.0 <= span.box.ymin < span.box.ymax <= 1.0


def test_cells_carry_their_own_boxes_in_the_stored_document():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        document = ingest_document(pdf, artifacts_dir=tmp)

        table = next(t for p in document.pages for t in p.tables)
        assert table.headers == ["Earning Trigger", "CC", "How to Earn It"]
        assert table.n_cols == 3 and table.n_rows >= 4

        for row in table.rows:
            for cell in row.cells:
                assert len(cell.bbox) == 4
                assert cell.bbox[0] < cell.bbox[2] and cell.bbox[1] < cell.bbox[3]
                assert cell.column in table.headers

        # Multi-line cells were rejoined rather than split across rows.
        triggers = [row.cells[0].text for row in table.rows]
        assert any("Cross-Functional Team Bonus" in t and "departments" in t
                   for t in triggers)


def test_records_view_pairs_columns_with_values():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        document = ingest_document(pdf, artifacts_dir=tmp)
        table = next(t for p in document.pages for t in p.tables)

        records = table.records()
        assert records and records[0]["Earning Trigger"] == "Fastest Quiz Responder Bonus"
        assert records[0]["CC"] == "5"


# ─────────────────────────────── 9/10. identity and output compatibility


def test_document_identity_is_unaffected_by_table_extraction():
    # SHA-256 is of the FILE. Richer parsing must not change it, or every
    # stored document would be orphaned by an extraction change.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")
    from swarn.capabilities.doc_store import document_id_for, file_sha256

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        before = file_sha256(pdf)
        document = ingest_document(pdf, artifacts_dir=tmp)

        assert document.source_sha256 == before == file_sha256(pdf)
        assert document.document_id == document_id_for(pdf)
        assert document.document_id.endswith(before[:12])


def test_json_output_shape_stays_backward_compatible():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        result = _ask(pdf, tmp, _cite("Fastest Quiz Responder Bonus CC 5"))
        payload = json.loads(result.to_json())

        # Every pre-existing key is still present and typed as before.
        for key in ("question", "document_name", "answer", "found", "computation",
                    "computation_check", "evidence", "annotated_image_paths",
                    "pages_searched", "page_count", "backend", "document_id"):
            assert key in payload, key
        span = payload["evidence"][0]
        for key in ("page_number", "line_id", "label", "quote", "box", "verified"):
            assert key in span, key
        assert set(span["box"]) == {"xmin", "ymin", "xmax", "ymax"}
        # New keys are additive.
        assert span["strategy"] == "table"
        assert span["table"]["row_index"] >= 0


def test_summary_line_still_flags_unverified_evidence():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "bonuses.pdf")
        _bonus_table_pdf(pdf)
        good = _ask(pdf, tmp, _cite("Fastest Quiz Responder Bonus CC 5"))
        bad = _ask(pdf, tmp, _cite("Blockchain Integration Bonus CC 50"))

        assert "NOT FOUND" not in good.summary()
        assert "NOT FOUND" in bad.summary()


# ─────────────────────────────── matcher internals


def test_tokenization_absorbs_only_formatting_differences():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    same = _tokens("Cross-Functional  Team\nBonus (3+ departments)")
    assert same == _tokens("cross functional team bonus 3 departments")
    # Word identity is preserved: a different word is a different token list.
    assert _tokens("Fastest Quiz") != _tokens("Slowest Quiz")


def test_token_span_finds_contiguous_runs_only():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    assert _token_span(["a", "b", "c", "d"], ["b", "c"]) == 1
    assert _token_span(["a", "b", "c", "d"], ["a", "c"]) == -1   # not contiguous
    assert _token_span(["a"], ["a", "b"]) == -1                  # longer than haystack


def test_resolver_tolerates_missing_or_malformed_evidence():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    assert resolve_evidence(None, {}) == []
    assert resolve_evidence("not a list", {}) == []
    assert resolve_evidence([{"line_id": "nonsense"}], {}) == []
