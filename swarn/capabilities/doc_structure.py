"""
Document tree from stored data — the bridge that lets `extract-pdf` and the
document Q&A stack share one parse.

Why this module exists
──────────────────────
`swarn extract-pdf` and `swarn ingest` used to be two independent pdfplumber
pipelines over the same file. Both opened the PDF, both grouped words into
lines, both detected tables, and neither knew the other existed. That is not
merely wasteful — it made them disagree. On a UPPCL electricity bill the
document-tree path produced:

  * `"Debitrepresentstheadditionalamountchargedtothe"` — the bill's text layer
    carries no space glyphs, and joining characters loses the word boundaries
    that the positioned words still have.
  * `"How to update Mobile/WhatsApp/Email How to verify Mobile/Email"` — two
    column headings merged, because that path had no column handling at all.
  * `"heading": "Office."` — a sentence fragment promoted to a heading.

The store already solves the first two: it keeps words as separate positioned
objects and knows which page column each line came from. So the tree is now
derived from the store rather than re-parsed, and inherits those fixes.

What this module is and is not
──────────────────────────────
It converts a `StoredDocument` into the flat, reading-order **element list**
that the tree builder in `agent/multimodal_rag.py` already consumes — lines
with their font size and weight, tables with their grids. Everything above
that (heading ranking, section assembly, block typing, field mining) is
unchanged and stays where it is; that logic is pure text handling and was
never the duplicated part.

The duplicated part was everything below: opening the file, grouping words,
finding tables, excluding table regions from the text layer. That is what this
replaces, and it replaces it with a read of data already on disk.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from swarn.capabilities.doc_store import (
    StoredDocument,
    StoredLine,
    StoredPage,
    StoredTable,
)

# A word counts as inside a table when its top-left corner falls in the table's
# box, matching the rule the old text-layer filter used. Half a point of slack
# absorbs the rounding a renderer applies to a ruled edge.
_BBOX_SLACK = 0.5


def _line_box(line: StoredLine) -> Optional[tuple]:
    """(left, top, right, bottom) over the line's words, or None if wordless."""
    if not line.words:
        return None
    return (
        min(w.left for w in line.words),
        min(w.top for w in line.words),
        max(w.left + w.width for w in line.words),
        max(w.top + w.height for w in line.words),
    )


def _in_any_table(line: StoredLine, tables: Sequence[StoredTable]) -> bool:
    """
    Is this line part of a detected table?

    Table contents must be emitted once, as a table. Left in the text layer as
    well, they appear a second time as prose whose column gaps have collapsed
    into meaningless runs ("Compute 412,000 398,500 -3.3%"). Excluding by box
    is exact; deduplicating the strings afterwards is guesswork.
    """
    box = _line_box(line)
    if box is None:
        return False
    left, top, _, _ = box
    for table in tables:
        if len(table.bbox) != 4:
            continue
        x0, y0, x1, y1 = table.bbox
        if (x0 - _BBOX_SLACK <= left <= x1 + _BBOX_SLACK
                and y0 - _BBOX_SLACK <= top <= y1 + _BBOX_SLACK):
            return True
    return False


def _table_element(table: StoredTable, index: int) -> Optional[dict]:
    """
    One StoredTable in the structured shape the document tree publishes.

    The header rule is the one `extract_pdf_structured` has always used and is
    strict on purpose: the first row becomes a header only if every cell is
    non-empty and no name repeats. A header with blanks or duplicates would
    silently collapse columns when zipped into a dict, losing data with no
    error. When it fails, `header` is null and `rows` holds EVERY row including
    the first, so nothing is dropped.
    """
    grid = [[cell.strip() for cell in row] for row in table.text_rows()]
    grid = [row for row in grid if any(row)]
    if not grid:
        return None

    n_cols = max(len(row) for row in grid)
    grid = [row + [""] * (n_cols - len(row)) for row in grid]

    head = grid[0]
    usable_header = all(h for h in head) and len(set(head)) == len(head)

    out = {
        "index":  index,
        "n_rows": len(grid) - 1 if usable_header else len(grid),
        "n_cols": n_cols,
        "header": head if usable_header else None,
        "rows":   grid[1:] if usable_header else grid,
    }
    if usable_header:
        out["records"] = [dict(zip(head, row)) for row in grid[1:]]
    return out


def _page_elements(page: StoredPage) -> List[dict]:
    """One stored page → its lines and tables, in reading order."""
    elements: List[dict] = []

    for index, table in enumerate(page.tables, start=1):
        structured = _table_element(table, index)
        if structured:
            elements.append({
                "kind": "table",
                "page": page.page_number,
                "top":  table.bbox[1] if len(table.bbox) == 4 else 0.0,
                "table": structured,
            })

    for line in page.lines:
        if _in_any_table(line, page.tables):
            continue
        text = " ".join(line.text.split())
        if not text:
            continue
        box = _line_box(line)
        sizes = [w.size for w in line.words if w.size]
        elements.append({
            "kind":   "line",
            "page":   page.page_number,
            "top":    box[1] if box else 0.0,
            "height": max(1.0, (box[3] - box[1]) if box else 1.0),
            "text":   text,
            # Largest size on the line, matching how a reader judges a heading:
            # one large word in a line of small ones still reads as emphasis.
            "size":   round(max(sizes), 2) if sizes else 0.0,
            # Every word bold, not any — a sentence with one bold term is not
            # a heading, and `all` over an empty list would call it one.
            "bold":   bool(line.words) and all(w.bold for w in line.words),
            "n_chars": len(text),
            # Carried through so a consumer can locate a block on the page.
            # The old path had no coordinates at all, which is why nothing
            # downstream could verify anything it produced.
            "box":    list(box) if box else None,
            "column": line.column,
            "line_id": line.line_id,
        })

    elements.sort(key=lambda e: (e["page"], e["top"]))
    return elements


def elements_from_stored(document: StoredDocument) -> List[dict]:
    """
    A whole `StoredDocument` → the flat reading-order element list.

    Pages are emitted in order and each page's elements sorted within it, so
    the caller sees exactly the sequence the old per-page pdfplumber walk
    produced — with the store's column ordering and row-join rules already
    applied to the lines.
    """
    elements: List[dict] = []
    for page in sorted(document.pages, key=lambda p: p.page_number):
        elements.extend(_page_elements(page))
    return elements
