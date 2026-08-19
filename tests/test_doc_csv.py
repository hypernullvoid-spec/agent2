"""
Tests for swarn/capabilities/doc_csv.py — PDF tables → CSV files on disk.

The theme is that the text-alignment fallback always returns *something*, so
most of these are about refusing it when what it returned is not a table. A
garbled CSV is worse than an empty one: it gets loaded, and the failure is
never noticed.
"""

import csv
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


def _dataset_pdf(dest: str, pages: int = 2) -> None:
    """A borderless dataset dump — no ruling at all, columns by alignment."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(dest, pagesize=A4)
    xs = [60, 160, 260, 360]
    for page in range(pages):
        y = 780
        if page == 0:
            c.setFont("Helvetica", 10)
            for x, name in zip(xs, ["city", "year", "population", "area"]):
                c.drawString(x, y, name)
            y -= 16
        c.setFont("Helvetica", 10)
        for row in range(12):
            n = page * 12 + row
            for x, value in zip(xs, [f"town{n}", f"20{10 + n % 9}",
                                     f"{1000 + n * 7}", f"{n * 3}.5"]):
                c.drawString(x, y, value)
            y -= 14
        c.showPage()
    c.save()


def _prose_pdf(dest: str) -> None:
    """An ordinary page of sentences — the text strategy will still find a
    'grid' in it, and must not be believed."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(dest, pagesize=A4)
    c.setFont("Helvetica", 11)
    y = 780
    for line in [
        "This bill will be construed as a final notice under section 9.3 of the",
        "Electricity Supply Code 2005. Supply can be disconnected at any date",
        "on non-payment of dues. If the consumer fails to pay the entire bill",
        "by the due date a late payment surcharge shall be levied on the",
        "unpaid amount and the connection will be disconnected without any",
        "further notice being served on the consumer at the premises.",
    ]:
        c.drawString(50, y, line)
        y -= 18
    c.showPage()
    c.save()


def _ruled_pdf(dest: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

    table = Table([["Meter", "Unit", "Reading"],
                   ["AL69", "KWH", "2230.35"],
                   ["AL69", "KVAH", "2253.89"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black)]))
    SimpleDocTemplate(dest, pagesize=A4).build([table])


def _convert(path: str, **kw):
    from swarn.capabilities.doc_csv import pdf_to_csv
    return pdf_to_csv(path, **kw)


def _rows(path: str):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


# ── the borderless dataset, which is the whole reason this exists ──────────

def test_a_borderless_dataset_is_converted():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp = os.path.join(tempfile.mkdtemp(), "data.pdf")
    _dataset_pdf(tmp)
    result = _convert(tmp)
    assert result.paths, result.summary()
    assert result.n_rows >= 24


def test_pages_of_the_same_shape_become_one_file():
    # Handing back one file per page is not a conversion.
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp = os.path.join(tempfile.mkdtemp(), "data.pdf")
    _dataset_pdf(tmp, pages=3)
    result = _convert(tmp)
    assert len(result.paths) == 1
    assert result.combined


def test_a_repeated_header_is_not_written_twice():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp = os.path.join(tempfile.mkdtemp(), "data.pdf")
    _dataset_pdf(tmp, pages=3)
    result = _convert(tmp)
    rows = _rows(result.paths[0])
    assert rows.count(rows[0]) == 1


def test_every_row_has_the_same_width():
    # A ragged CSV fails to parse as a table, which is the one thing the
    # caller asked for.
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp = os.path.join(tempfile.mkdtemp(), "data.pdf")
    _dataset_pdf(tmp, pages=2)
    result = _convert(tmp)
    widths = {len(r) for r in _rows(result.paths[0])}
    assert len(widths) == 1, widths


# ── refusing what is not a table ───────────────────────────────────────────

def test_prose_is_not_reported_as_a_table():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp = os.path.join(tempfile.mkdtemp(), "prose.pdf")
    _prose_pdf(tmp)
    result = _convert(tmp)
    assert not result.paths, f"prose was converted: {result.summary()}"
    assert "No table-shaped content" in result.summary()


def test_a_grid_that_cuts_words_in_half_is_rejected():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    from swarn.capabilities.doc_csv import _cuts_words

    class _Word:
        def __init__(self, text): self.text = text

    class _Line:
        def __init__(self, words): self.words = [_Word(w) for w in words]

    class _Page:
        lines = [_Line(["Section", "9.3", "applies"])]

    intact = [["Section", "9.3"], ["applies", ""]]
    cut = [["Sectio", "n9.3"], ["appli", "es"]]
    assert _cuts_words(intact, _Page()) == 0.0
    assert _cuts_words(cut, _Page()) == 1.0


def test_a_sparse_grid_is_not_a_table():
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from swarn.capabilities.doc_csv import _looks_tabular

    dense = [["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
    sparse = [["", "", "x"], ["y", "", ""], ["", "z", ""], ["", "", "w"]]
    assert _looks_tabular(dense)
    assert not _looks_tabular(sparse)


# ── ruled tables ───────────────────────────────────────────────────────────

def test_a_ruled_table_comes_from_the_store():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp = os.path.join(tempfile.mkdtemp(), "ruled.pdf")
    _ruled_pdf(tmp)
    result = _convert(tmp)
    assert result.paths, result.summary()
    assert result.grids[0].strategy == "ruled"
    assert _rows(result.paths[0])[0] == ["Meter", "Unit", "Reading"]


# ── output layout ──────────────────────────────────────────────────────────

def test_each_pdf_gets_its_own_folder():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp_dir = tempfile.mkdtemp()
    tmp = os.path.join(tmp_dir, "data.pdf")
    _dataset_pdf(tmp)
    result = _convert(tmp)
    assert result.folder == os.path.join(tmp_dir, "data")
    for path in result.paths:
        assert os.path.dirname(path) == result.folder


def test_the_folder_carries_a_manifest():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    import json

    tmp = os.path.join(tempfile.mkdtemp(), "data.pdf")
    _dataset_pdf(tmp)
    result = _convert(tmp)
    manifest = json.load(open(os.path.join(result.folder, "tables.json")))
    assert manifest["n_tables"] == len(result.paths)
    for entry in manifest["tables"]:
        assert os.path.exists(os.path.join(result.folder, entry["file"]))
        assert entry["pages"] and entry["n_rows"] and entry["n_cols"]


def test_out_path_writes_one_flat_file_and_no_folder():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp_dir = tempfile.mkdtemp()
    tmp = os.path.join(tmp_dir, "data.pdf")
    _dataset_pdf(tmp, pages=2)
    target = os.path.join(tmp_dir, "flat.csv")
    result = _convert(tmp, out_path=target)
    assert result.paths == [target]
    assert result.folder is None


def test_out_dir_puts_the_folder_under_it():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp_dir = tempfile.mkdtemp()
    out = tempfile.mkdtemp()
    tmp = os.path.join(tmp_dir, "data.pdf")
    _dataset_pdf(tmp)
    result = _convert(tmp, out_dir=out)
    assert result.folder == os.path.join(out, "data")


def test_page_selection_is_honoured():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp = os.path.join(tempfile.mkdtemp(), "data.pdf")
    _dataset_pdf(tmp, pages=3)
    result = _convert(tmp, pages=[2])
    assert {g.page for g in result.grids} == {2}


def test_a_missing_file_is_an_error_not_a_crash():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + pydantic")
    from swarn.capabilities.doc_intelligence import DocumentIntelligenceError
    try:
        _convert("/nonexistent/nope.pdf")
    except DocumentIntelligenceError:
        return
    raise AssertionError("expected DocumentIntelligenceError")


# ── fused cells ────────────────────────────────────────────────────────────

def test_fused_cells_are_counted_but_not_split_by_default():
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from swarn.capabilities.doc_csv import _count_fused

    assert _count_fused([["124.9833yes", "no"], ["9.854167no", "no"]]) == 2
    assert _count_fused([["124.9833", "yes"]]) == 0


def test_splitting_is_refused_when_it_misaligns_rows():
    # The left half being a variable-width row index shifts halves into
    # different columns; the unsplit grid is the lesser evil.
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from swarn.capabilities.doc_csv import _split_is_aligned

    aligned = [["a", "b"], ["1", "yes"], ["2", "yes"], ["3", "no"], ["4", "no"]]
    shifted = [["a", "b", "c"], ["1", "yes", ""], ["", "2", "yes"],
               ["3", "no", ""], ["", "4", "no"]]
    assert _split_is_aligned(aligned)
    assert not _split_is_aligned(shifted)


# ── table boundaries ───────────────────────────────────────────────────────

def _two_unrelated_ruled_tables(dest: str) -> None:
    """Two ruled tables, same width, different subjects — must not merge."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import PageBreak, SimpleDocTemplate, Table, TableStyle

    style = TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black)])
    first = Table([["Date", "Milestone", "What Happens"],
                   ["21 Jul", "Portal live", "Registration begins"],
                   ["27 Jul", "Submission closes", "No late entries"]])
    second = Table([["Award", "Criteria", "Recognition"],
                    ["Best Idea", "Highest score", "Trophy"],
                    ["Peoples Choice", "Audience vote", "Certificate"]])
    first.setStyle(style)
    second.setStyle(style)
    SimpleDocTemplate(dest, pagesize=A4).build([first, PageBreak(), second])


def test_two_ruled_tables_of_equal_width_stay_separate():
    # Grouping on width alone welded five unrelated three-column tables of a
    # participant handbook into one CSV.
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    tmp = os.path.join(tempfile.mkdtemp(), "handbook.pdf")
    _two_unrelated_ruled_tables(tmp)
    result = _convert(tmp)
    assert len(result.paths) == 2, result.summary()
    headers = {tuple(_rows(path)[0]) for path in result.paths}
    assert ("Date", "Milestone", "What Happens") in headers
    assert ("Award", "Criteria", "Recognition") in headers


def test_a_ruled_table_continues_only_when_its_header_repeats():
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from swarn.capabilities.doc_csv import Grid, _continues

    head = ["Date", "Milestone", "What Happens"]
    same = Grid(rows=[head, ["a", "b", "c"]], page=1, strategy="ruled")
    repeat = Grid(rows=[head, ["d", "e", "f"]], page=2, strategy="ruled")
    other = Grid(rows=[["Award", "Criteria", "Recognition"], ["x", "y", "z"]],
                 page=2, strategy="ruled")
    assert _continues(same, repeat)
    assert not _continues(same, other)


def test_text_grids_continue_on_width_alone():
    # A dataset's later pages carry no header at all, so width is the only
    # signal there is — and the right one, since nothing asserted a boundary.
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from swarn.capabilities.doc_csv import Grid, _continues

    first = Grid(rows=[["a", "b"], ["1", "2"]], page=1, strategy="text")
    second = Grid(rows=[["3", "4"], ["5", "6"]], page=2, strategy="text")
    narrow = Grid(rows=[["7"], ["8"]], page=3, strategy="text")
    assert _continues(first, second)
    assert not _continues(second, narrow)


def test_a_ruled_and_a_text_grid_never_merge():
    if not _HAVE_DEPS:
        return _skip("needs pydantic")
    from swarn.capabilities.doc_csv import Grid, _continues

    ruled = Grid(rows=[["a", "b"], ["1", "2"]], page=1, strategy="ruled")
    text = Grid(rows=[["a", "b"], ["3", "4"]], page=2, strategy="text")
    assert not _continues(ruled, text)


# ── automatic export at ingest ─────────────────────────────────────────────

def test_ingest_writes_the_tables_as_csv():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    from swarn.capabilities.doc_store import ingest_document

    tmp_dir = tempfile.mkdtemp()
    pdf = os.path.join(tmp_dir, "handbook.pdf")
    _two_unrelated_ruled_tables(pdf)
    document = ingest_document(pdf, artifacts_dir=tmp_dir, save=False)

    assert document.tables_dir, "ingest did not export any tables"
    written = [f for f in os.listdir(document.tables_dir) if f.endswith(".csv")]
    assert len(written) == 2


def test_export_can_be_switched_off():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    from swarn.capabilities.doc_store import ingest_document

    tmp_dir = tempfile.mkdtemp()
    pdf = os.path.join(tmp_dir, "handbook.pdf")
    _two_unrelated_ruled_tables(pdf)
    document = ingest_document(pdf, artifacts_dir=tmp_dir, save=False,
                               export_csv=False)
    assert document.tables_dir == ""


def test_a_document_with_no_tables_records_no_folder():
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    from swarn.capabilities.doc_store import ingest_document

    tmp_dir = tempfile.mkdtemp()
    pdf = os.path.join(tmp_dir, "prose.pdf")
    _prose_pdf(pdf)
    document = ingest_document(pdf, artifacts_dir=tmp_dir, save=False)
    assert document.tables_dir == ""


def test_a_failing_export_never_costs_the_parse():
    # A document that cannot be tabulated is still a good stored document.
    if not _HAVE_DEPS:
        return _skip("needs pdfplumber + reportlab + pydantic")
    import swarn.capabilities.doc_csv as doc_csv
    from swarn.capabilities.doc_store import ingest_document

    tmp_dir = tempfile.mkdtemp()
    pdf = os.path.join(tmp_dir, "handbook.pdf")
    _two_unrelated_ruled_tables(pdf)

    original = doc_csv.pdf_to_csv
    doc_csv.pdf_to_csv = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        document = ingest_document(pdf, artifacts_dir=tmp_dir, save=False)
    finally:
        doc_csv.pdf_to_csv = original

    assert document.tables_dir == ""
    assert document.n_words() > 0, "the parse itself must survive"
