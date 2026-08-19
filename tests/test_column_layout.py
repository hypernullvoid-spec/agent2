"""
Tests for column-aware transcript construction.

The bug these cover: `_lines_for_page` joined every horizontal band's column
segments with " | ", so a two-column slide produced

    [p1:L11] TEAM MEMBERS | Jaimin Nalin Desai

where the caption is in the LEFT column and the name is a value in the RIGHT
one, sitting under its own "MENTOR NAME" caption. Asked to list team members, a
model answered "Jaimin Nalin Desai" — correctly, given what it was shown. The
pipe asserted a relationship the document never made.

Two properties are under test. Independent page columns must not be joined, and
a multi-column page must be emitted in reading order so a caption stays adjacent
to its own value. Both must hold WITHOUT breaking table rows, where the pipes
are load-bearing and the cells really do belong together.
"""

import os
import tempfile


def _have(*mods) -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in mods)


_HAVE_DEPS = _have("pydantic", "PIL", "pdfplumber")
_HAVE_PDF = _HAVE_DEPS and _have("reportlab")

if _HAVE_DEPS:
    from swarn.capabilities.doc_intelligence import (
        COLUMN_BAND_GAP, COLUMN_JOIN_GAP, band_of, column_bands,
    )
    from swarn.capabilities.doc_store import ingest_document


def _skip(reason: str) -> bool:
    print(f"    (skipped: {reason})")
    return True


# ─────────────────────────────── fixtures


def _two_column_slide(dest: str) -> None:
    """A landscape slide shaped like the one that exposed the bug: a left
    column of members and a right column of captioned single values, with the
    caption/value pairs deliberately at the same heights as left-column rows."""
    from reportlab.lib.pagesizes import landscape
    from reportlab.pdfgen import canvas

    width, height = 960, 540
    c = canvas.Canvas(dest, pagesize=(width, height))

    def at(x, y_from_top, text, size=11, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, height - y_from_top, text)

    # Full-width title crossing the gutter — must not defeat column detection.
    at(26, 150, "SWARAJ CLOUDFORGE", size=40, bold=True)

    # LEFT column (x=31): the caption and the real members.
    at(31, 391, "TEAM MEMBERS", size=10, bold=True)
    for offset, name in enumerate(["Anupriys", "Anupriya Raj", "Edunoori Spoorthi",
                                    "Sargam Maurya", "Tanmayee Patil", "Neha Kesharkar"]):
        at(33, 405 + offset * 14, name)

    # RIGHT column (x=620): captions each owning the value directly below.
    at(620, 57, "TEAM NAME", size=10, bold=True)
    at(620, 95, "Cloud Catalyst")
    at(620, 350, "MENTOR NAME", size=10, bold=True)
    at(620, 393, "Jaimin Nalin Desai")       # same band as "TEAM MEMBERS"
    at(614, 446, "SUBMIT BY", size=10, bold=True)
    at(620, 493, "9 August 2026, 10:00 PM")
    c.showPage()
    c.save()


def _single_column_report(dest: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    ss = getSampleStyleSheet()
    SimpleDocTemplate(dest, pagesize=A4).build([
        Paragraph("Quarterly Review", ss["Title"]),
        Paragraph("Revenue grew across every segment this quarter.", ss["BodyText"]),
        Paragraph("Costs were held flat against the prior period.", ss["BodyText"]),
    ])


def _table_report(dest: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    table = Table([["Metric", "FY23", "FY24"],
                   ["Revenue", "120100", "148200"],
                   ["Operating cost", "91400", "102300"]], colWidths=[150, 90, 90])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black)]))
    SimpleDocTemplate(dest, pagesize=A4).build([
        Paragraph("Financial Summary", ss["Heading1"]), Spacer(1, 12), table,
    ])


def _borderless_table(dest: str) -> None:
    """A two-column table with NO ruling — invisible to find_tables()."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table

    SimpleDocTemplate(dest, pagesize=A4).build([
        Table([["Date", "Activity"],
               ["24 Jul", "Idea submission closes"],
               ["01 Aug", "Shortlist announced"]], colWidths=[80, 260]),
    ])


def _two_tables_side_by_side(dest: str) -> None:
    """Two ruled tables in adjacent page columns, rows at the same heights."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape
    from reportlab.platypus import Table, TableStyle
    from reportlab.pdfgen import canvas

    width, height = 960, 540
    c = canvas.Canvas(dest, pagesize=(width, height))
    style = TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black)])

    left = Table([["Lheader", "Lvalue"], ["Lrow1", "1"], ["Lrow2", "2"], ["Lrow3", "3"]],
                 colWidths=[110, 60])
    right = Table([["Rheader", "Rvalue"], ["Rrow1", "9"], ["Rrow2", "8"], ["Rrow3", "7"]],
                  colWidths=[110, 60])
    for table, x in ((left, 40), (right, 640)):
        table.setStyle(style)
        table.wrapOn(c, width, height)
        table.drawOn(c, x, height - 260)
    c.showPage()
    c.save()


def _lines(pdf, tmp, page=1):
    return ingest_document(pdf, artifacts_dir=tmp).page(page).lines


def _text_of(lines):
    return [line.text for line in lines]


def _index_of(lines, needle):
    return next(i for i, line in enumerate(lines) if needle in line.text)


# ─────────────────────────────── band detection


def test_a_wide_gutter_splits_into_bands():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    # The measured two-column slide: columns at x≈3% and x≈64% of the page.
    assert len(column_bands([26, 31, 33, 33, 614, 620, 620], page_width=960)) == 2
    # A narrow split stays one band.
    assert len(column_bands([51, 100, 140, 190], page_width=595)) == 1


def test_neither_join_signal_is_sufficient_alone():
    # Why `_same_row` consults BOTH a detected table and distance. Measured on
    # two real documents:
    #   * a contacts table's columns sit 43% of page width apart — further than
    #     some page gutters, so distance alone would shred its rows
    #   * a BORDERLESS table is invisible to find_tables(), so detection alone
    #     shredded three of them (`Date | Activity` became two lines)
    # Each signal covers the other's blind spot; removing either reintroduces a
    # regression that was observed, not hypothesised.
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")

    page_width = 595
    wide_table_gap = 0.43 * page_width
    borderless_cell_gap = 0.05 * page_width
    page_gutter = 0.54 * page_width

    assert wide_table_gap > COLUMN_JOIN_GAP * page_width, \
        "distance alone would wrongly split a wide table"
    assert borderless_cell_gap < COLUMN_JOIN_GAP * page_width, \
        "distance must rescue borderless tables, which detection cannot see"
    assert page_gutter > COLUMN_JOIN_GAP * page_width, \
        "a real page gutter must still split"


def test_band_detection_edges():
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    assert column_bands([], page_width=960) == []
    assert len(column_bands([100], page_width=960)) == 1
    # Exactly at the threshold counts as a split.
    exact = COLUMN_BAND_GAP * 1000
    assert len(column_bands([0, exact], page_width=1000)) == 2
    assert len(column_bands([0, exact - 1], page_width=1000)) == 1


def test_band_of_places_a_full_width_element_by_its_left_edge():
    # The title spans the gutter; it belongs to the column it starts in.
    if not _HAVE_DEPS:
        return _skip("needs pydantic + Pillow + pdfplumber")
    bands = column_bands([26, 33, 614, 620], page_width=960)
    assert band_of(26, bands) == 0
    assert band_of(620, bands) == 1
    assert band_of(-5, bands) in (0, len(bands) - 1)      # never raises


# ─────────────────────────────── two-column pages


def test_independent_page_columns_are_not_joined():
    # THE regression: "TEAM MEMBERS | Jaimin Nalin Desai" must not exist.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "slide.pdf")
        _two_column_slide(pdf)
        texts = _text_of(_lines(pdf, tmp))

        assert not any("|" in text for text in texts), \
            f"no page-column line may be joined: {[t for t in texts if '|' in t]}"
        assert not any("TEAM MEMBERS" in t and "Jaimin" in t for t in texts)
        assert "TEAM MEMBERS" in texts and "Jaimin Nalin Desai" in texts


def test_a_caption_is_adjacent_to_its_own_value_after_reordering():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "slide.pdf")
        _two_column_slide(pdf)
        lines = _lines(pdf, tmp)

        # MENTOR NAME is immediately followed by the mentor.
        assert _index_of(lines, "Jaimin Nalin Desai") == _index_of(lines, "MENTOR NAME") + 1
        # TEAM MEMBERS is immediately followed by the first member, and every
        # member precedes the right column entirely.
        members_at = _index_of(lines, "TEAM MEMBERS")
        assert _index_of(lines, "Anupriys") == members_at + 1
        assert _index_of(lines, "Neha Kesharkar") < _index_of(lines, "MENTOR NAME")


def test_columns_are_emitted_left_to_right_and_labelled():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "slide.pdf")
        _two_column_slide(pdf)
        lines = _lines(pdf, tmp)

        columns = [line.column for line in lines]
        assert set(columns) == {0, 1}
        assert columns == sorted(columns), "each column is emitted in full before the next"
        assert lines[_index_of(lines, "TEAM MEMBERS")].column == 0
        assert lines[_index_of(lines, "MENTOR NAME")].column == 1


def test_the_transcript_no_longer_supports_the_wrong_answer():
    # Read the rendered transcript the way the model does: the only caption
    # above "Jaimin Nalin Desai" is MENTOR NAME.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "slide.pdf")
        _two_column_slide(pdf)
        rendered = ingest_document(pdf, artifacts_dir=tmp).page(1).render()

        mentor_block = rendered[rendered.index("MENTOR NAME"):]
        assert "Jaimin Nalin Desai" in mentor_block.splitlines()[1]

        members_block = rendered[rendered.index("TEAM MEMBERS"):rendered.index("TEAM NAME")]
        assert "Jaimin" not in members_block
        for member in ("Anupriys", "Neha Kesharkar"):
            assert member in members_block


# ─────────────────────────────── no regression elsewhere


def test_table_rows_are_still_joined_with_pipes():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "report.pdf")
        _table_report(pdf)
        texts = _text_of(_lines(pdf, tmp))

        assert "Revenue | 120100 | 148200" in texts
        assert "Operating cost | 91400 | 102300" in texts


def test_single_column_pages_keep_positional_order():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "report.pdf")
        _single_column_report(pdf)
        lines = _lines(pdf, tmp)

        assert all(line.column == 0 for line in lines)
        # Top-to-bottom, as written.
        assert _index_of(lines, "Quarterly Review") < _index_of(lines, "Revenue grew")
        assert _index_of(lines, "Revenue grew") < _index_of(lines, "Costs were held")
        # Vertical order is preserved exactly.
        tops = [min(w.top for w in line.words) for line in lines]
        assert tops == sorted(tops)


def test_a_single_column_table_page_keeps_positional_order():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "report.pdf")
        _table_report(pdf)
        page = ingest_document(pdf, artifacts_dir=tmp).page(1)

        assert page.tables, "fixture should have a detected table"
        assert all(line.column == 0 for line in page.lines)
        tops = [min(w.top for w in line.words) for line in page.lines]
        assert tops == sorted(tops)


def test_borderless_table_rows_stay_joined():
    # Regression: gating the join on find_tables() alone shredded every table
    # drawn without ruling, because find_tables() cannot see them. Their cells
    # sit close together, so proximity must hold them.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "borderless.pdf")
        _borderless_table(pdf)
        page = ingest_document(pdf, artifacts_dir=tmp).page(1)

        assert not page.tables, "fixture is deliberately unruled"
        texts = _text_of(page.lines)
        assert "Date | Activity" in texts
        assert any(t.startswith("24 Jul | Idea submission") for t in texts)


def test_two_tables_side_by_side_do_not_interleave():
    # Regression: reordering used to be suppressed on any page containing a
    # table, so a page holding two SIDE-BY-SIDE tables alternated their rows —
    # the exact adjacency error this whole change exists to remove.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "twotables.pdf")
        _two_tables_side_by_side(pdf)
        page = ingest_document(pdf, artifacts_dir=tmp).page(1)

        texts = _text_of(page.lines)
        left = [i for i, t in enumerate(texts) if t.startswith("L")]
        right = [i for i, t in enumerate(texts) if t.startswith("R")]
        assert left and right
        # Each table occupies one contiguous run of lines.
        assert left == list(range(left[0], left[0] + len(left))), texts
        assert right == list(range(right[0], right[0] + len(right))), texts


def test_line_ids_stay_sequential_and_resolvable():
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "slide.pdf")
        _two_column_slide(pdf)
        page = ingest_document(pdf, artifacts_dir=tmp).page(1)

        assert [l.line_id for l in page.lines] == \
               [f"p1:L{i}" for i in range(1, len(page.lines) + 1)]
        for line in page.lines:
            found = page.find(line.line_id)
            assert found is not None and found["text"] == line.text
            assert found["words"], "every line must resolve back to its words"


def test_words_and_boxes_survive_the_reordering():
    # Reordering must move lines, never lose or corrupt their geometry.
    if not _HAVE_PDF:
        return _skip("needs pydantic + Pillow + pdfplumber + reportlab")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "slide.pdf")
        _two_column_slide(pdf)
        page = ingest_document(pdf, artifacts_dir=tmp).page(1)

        mentor = next(l for l in page.lines if "Jaimin" in l.text)
        assert [w.text for w in mentor.words] == ["Jaimin", "Nalin", "Desai"]
        # It is in the RIGHT column, so its box must be on the right of the page.
        assert min(w.left for w in mentor.words) > page.width * 0.5

        members = next(l for l in page.lines if l.text == "TEAM MEMBERS")
        assert max(w.left + w.width for w in members.words) < page.width * 0.5
