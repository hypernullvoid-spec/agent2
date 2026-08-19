"""
PDF → CSV.

`extract-pdf --csv-dir` already writes a CSV per table, but only for tables
pdfplumber's default strategy can see — ruled ones. A great many PDFs that
exist *to carry a table* have no ruling at all: an R data frame printed to
PDF, a bank statement, a report appendix. On a 54-page credit-card dataset
(1,300 records) that path produced zero files, because `find_tables()` found
zero tables.

So this module tries three things in order and reports which one answered:

  1. **ruled**   — tables already detected and stored by `swarn ingest`.
                   Exact: the grid comes from the lines the document draws.
  2. **text**    — pdfplumber's text-alignment strategy, which infers columns
                   from where words line up rather than from ruling. This is
                   what recovers a borderless dataset dump, and on the credit
                   card PDF it returns a clean 9-column grid per page.
  3. **nothing** — reported as such. A garbled CSV that looks like data is
                   worse than an honest failure, because the failure is
                   noticed and the garbage is not.

Fused cells
───────────
A dump whose columns are right-aligned and touching has no gap for any
strategy to find, so two values arrive in one cell: `1yes` is the row index
and the `card` value; `124.9833yes` is the expenditure and the owner. These
are *detected* and counted always, but split only when the caller passes
`split_fused=True` — splitting at a digit/letter boundary is right for a data
dump and wrong for a document holding "COVID19" or "3M", and this module
cannot tell which it has.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from swarn.capabilities.doc_intelligence import DocumentIntelligenceError

# pdfplumber's text strategy, used when nothing is ruled. Both axes are set to
# "text": columns come from vertical alignment of words, rows from horizontal.
_TEXT_STRATEGY = {"vertical_strategy": "text", "horizontal_strategy": "text"}

# A number running straight into a word, or the reverse, with no separator.
# Only ever applied when the caller opts in — see the module docstring.
_FUSED_RE = re.compile(r"(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)")


@dataclass
class Grid:
    """One table's cells, plus how it was found."""

    rows: List[List[str]] = field(default_factory=list)
    page: int = 1
    index: int = 1
    strategy: str = "ruled"          # 'ruled' | 'text'
    n_fused: int = 0                 # cells holding two values with no separator
    split_refused: bool = False      # --split-fused asked for, but it misaligned

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)


@dataclass
class ConversionResult:
    """What was written, and what could not be."""

    paths: List[str] = field(default_factory=list)
    grids: List[Grid] = field(default_factory=list)
    folder: Optional[str] = None      # the per-PDF folder, when one was made
    combined: bool = False
    pages_with_nothing: List[int] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return sum(g.n_rows for g in self.grids)

    @property
    def n_fused(self) -> int:
        return sum(g.n_fused for g in self.grids)

    def summary(self) -> str:
        if not self.paths:
            return ("No table-shaped content found — nothing was written. "
                    "The PDF may be prose, or scanned (no text layer).")
        how = ", ".join(sorted({g.strategy for g in self.grids}))
        lines = [f"{self.n_rows} row(s) from {len(self.grids)} table(s)  [{how}]"]
        if self.folder:
            lines.append(f"  -> {self.folder}/")
            for path in self.paths:
                lines.append(f"       {Path(path).name}")
            lines.append("       tables.json")
        else:
            for path in self.paths:
                lines.append(f"  -> {path}")
        if self.pages_with_nothing:
            shown = ", ".join(str(p) for p in self.pages_with_nothing[:12])
            more = "" if len(self.pages_with_nothing) <= 12 else " ..."
            lines.append(f"  no table found on page(s): {shown}{more}")
        if any(g.split_refused for g in self.grids):
            lines.append(
                f"  NOTE: {self.n_fused} cell(s) hold two values with no separator, but "
                "splitting them shifted rows out of alignment (the left half is a "
                "variable-width row index), so the unsplit grid was kept. Split the "
                "affected column after loading instead.")
        elif self.n_fused:
            lines.append(
                f"  WARNING: {self.n_fused} cell(s) hold two values with no separator "
                "between them (touching right-aligned columns). Re-run with "
                "--split-fused to split them at the digit/letter boundary.")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# GRID EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════


def _clean(cell) -> str:
    return " ".join(str(cell or "").split())


def _normalize(rows: Sequence[Sequence]) -> List[List[str]]:
    """Clean cells, drop blank rows, pad ragged rows, drop always-empty columns.

    The empty-column drop matters for the text strategy, which reports a
    leading empty column for a page's left margin on essentially every dump.
    """
    grid = [[_clean(c) for c in row] for row in rows]
    grid = [row for row in grid if any(row)]
    if not grid:
        return []

    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]

    keep = [i for i in range(width) if any(row[i] for row in grid)]
    return [[row[i] for i in keep] for row in grid]


def _count_fused(grid: Sequence[Sequence[str]]) -> int:
    return sum(1 for row in grid for cell in row if _FUSED_RE.search(cell))


# After splitting, a column filled in fewer than this share of data rows means
# the split shifted rows relative to each other instead of separating columns.
_COLUMN_FILL_FLOOR = 0.9


def _split_fused(grid: List[List[str]]) -> List[List[str]]:
    """Split every digit/letter boundary, then re-pad to the widest row."""
    out = []
    for row in grid:
        cells: List[str] = []
        for cell in row:
            cells.extend(_FUSED_RE.split(cell) if cell else [cell])
        out.append(cells)
    width = max((len(r) for r in out), default=0)
    return [row + [""] * (width - len(row)) for row in out]


def _split_is_aligned(grid: Sequence[Sequence[str]]) -> bool:
    """
    Did splitting produce a real grid, or just shift rows past each other?

    Splitting is only safe when the fused values have a fixed width. Where the
    left half is a row index, it does not: on the credit-card dump a one-digit
    index and a two-digit index start at different x, so the column inference
    upstream had already placed them in different columns, and splitting moves
    their halves into different columns too. The result reads like a table and
    is not one — record 9's `card` value sits under record 10's `reports`.

    A genuine column is populated in nearly every data row. A column that is
    filled in only some of them is the signature of that shift, so the caller
    keeps the unsplit grid, where every value is at least in a consistent
    place.
    """
    body = [row for row in grid[1:] if any(row)]
    if len(body) < 3:
        return True                     # too little evidence to reject on
    width = max(len(row) for row in body)
    for column in range(width):
        filled = sum(1 for row in body if column < len(row) and row[column])
        if filled and filled < _COLUMN_FILL_FLOOR * len(body):
            return False
    return True


def _repair_header(grid: Grid, document) -> None:
    """
    Replace a mis-cut header row with the words the store segmented.

    The text strategy infers column boundaries from where *data* lines up, and
    a header set in a different alignment gets cut at those same boundaries —
    on the credit-card dump it produced `ag`, `e inc`, `ome s` out of "age",
    "income", "share". The store already holds that line correctly split into
    words, because it groups words rather than re-cutting a rendered string.

    Applied only when the two agree on the TEXT and disagree on the CUTS: the
    header cells and the stored words must concatenate to the same characters.
    That makes this a repair of a known-bad split, not a guess that some other
    line might be the header.
    """
    if not grid.rows:
        return
    page = document.page(grid.page)
    if page is None or not page.lines:
        return

    names = [w.text for w in page.lines[0].words]
    if not names:
        return

    def squashed(parts: Sequence[str]) -> str:
        return "".join("".join(str(p).split()) for p in parts)

    if squashed(grid.rows[0]) != squashed(names):
        return

    width = grid.n_cols
    if len(names) == width - 1:
        # A leading unnamed column — a row index, which data dumps print
        # without a header of its own.
        names = [""] + names
    elif len(names) != width:
        return
    grid.rows[0] = names


def _stored_grids(document) -> List[Grid]:
    """Ruled tables, straight from the store — no re-reading of the file."""
    grids = []
    for page in sorted(document.pages, key=lambda p: p.page_number):
        for table in page.tables:
            rows = _normalize(table.text_rows())
            if rows:
                grids.append(Grid(rows=rows, page=page.page_number,
                                  index=table.index, strategy="ruled"))
    return grids


def _text_grid(pdf_path: str, page_number: int, stored_page=None) -> Optional[Grid]:
    """One page's grid inferred from text alignment, or None."""
    try:
        import pdfplumber
    except ImportError as exc:                                       # noqa: BLE001
        raise DocumentIntelligenceError(
            f"PDF → CSV requires 'pip install pdfplumber' ({exc}).") from exc

    with pdfplumber.open(pdf_path) as pdf:
        if not 1 <= page_number <= len(pdf.pages):
            return None
        raw = pdf.pages[page_number - 1].extract_table(_TEXT_STRATEGY)

    rows = _normalize(raw or [])
    # One column is a list, not a table; one row is a stray line. Requiring
    # both keeps ordinary prose pages from being reported as data.
    if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
        return None
    if not _looks_tabular(rows):
        return None
    if stored_page is not None and _cuts_words(rows, stored_page) > _CUT_WORD_CEILING:
        return None
    return Grid(rows=rows, page=page_number, strategy="text")


# A real table's rows are mostly full. Below this, the "columns" are an artefact
# of the strategy carving up a page that has no columns.
_ROW_FILL_FLOOR = 0.6
# ...and most of its columns appear in most of its rows.
_COLUMN_PRESENCE_FLOOR = 0.6


# Share of a grid's tokens that may be text the store never saw as a word.
# Measured: a genuine data page scores 0-4% (the 4% is a mis-cut header row,
# which _repair_header then fixes); a prose page carved into columns scores 28%.
_CUT_WORD_CEILING = 0.15


def _cuts_words(rows: Sequence[Sequence[str]], stored_page) -> float:
    """
    What fraction of this grid's tokens are not words the document contains?

    The definitive signature of a false grid is a column boundary that falls
    inside a word. On the bill's statutory-notices page the text strategy
    produced `asFinalNoticeunderSectio` in one cell and `n9.3` in the next —
    "Section 9.3", cut in half. Nothing about the *shape* of that grid is
    wrong: its rows are full and its columns are consistent, so the fill tests
    above pass it. What is wrong is the text.

    The store settles it, because it grouped the page's words rather than
    re-cutting a rendered string: a token that is not one of those words is
    text this grid invented. Fragmentation is measured rather than forbidden
    outright — a stray mismatch is normal, a quarter of the page is not.
    """
    words = {w.text for line in stored_page.lines for w in line.words}
    tokens = [t for row in rows for cell in row if cell for t in cell.split()]
    if not tokens:
        return 1.0
    return sum(1 for t in tokens if t not in words) / len(tokens)


def _looks_tabular(rows: Sequence[Sequence[str]]) -> bool:
    """
    Is this a table, or prose that the text strategy carved into a grid?

    The strategy always returns *something*: run it on an invoice and it slices
    the address block at whatever x-positions the rest of the page suggests,
    producing cells like `MAKEMYTRIP (I` | `NDIA) PRIVATE` — a grid of word
    fragments that reads, to anything downstream, exactly like data. That is
    the worst possible output, because a garbled CSV is used and an empty one
    is investigated.

    Two properties separate the cases, and prose fails both. A table's rows are
    mostly full, where a page of text leaves most of each row's cells empty;
    and a table's columns appear in most of its rows, where prose produces
    columns that exist on a handful of lines and nowhere else.
    """
    body = [row for row in rows if any(row)]
    if len(body) < 3:
        return False
    width = max(len(row) for row in body)

    fills = sorted(sum(1 for cell in row if cell) / width for row in body)
    median_fill = fills[len(fills) // 2]
    if median_fill < _ROW_FILL_FLOOR:
        return False

    present = sum(
        1 for column in range(width)
        if sum(1 for row in body if column < len(row) and row[column])
        >= _COLUMN_PRESENCE_FLOOR * len(body)
    )
    return present >= _COLUMN_PRESENCE_FLOOR * width


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSION
# ══════════════════════════════════════════════════════════════════════════════


def _group_by_shape(grids: Sequence[Grid]) -> List[List[Grid]]:
    """
    Split the page grids into tables, on every change of column count.

    A dataset printed across pages is one table and must land in one file —
    handing back 54 files to concatenate is not a conversion. But a PDF is not
    obliged to hold only one table: the credit-card dump is an R console
    session whose first 27 pages are a 9-column data frame and whose page 30
    is a 3-column one. Grouping CONSECUTIVE pages of equal width keeps the
    first case whole without merging the second into nonsense.
    """
    groups: List[List[Grid]] = []
    for grid in grids:
        if groups and _continues(groups[-1][-1], grid):
            groups[-1].append(grid)
        else:
            groups.append([grid])
    return groups


def _continues(previous: Grid, current: Grid) -> bool:
    """
    Is `current` the same table as `previous`, carried onto the next page?

    Equal width is necessary and nowhere near sufficient. A participant
    handbook held 18 ruled tables of which most were three columns wide —
    "Earning Trigger / CC / How to Earn It", "Time / Section / What to Cover",
    "Award / Criteria / Recognition" — and grouping on width alone welded five
    unrelated tables into one CSV. What separates the cases is who drew the
    boundary:

      * A **ruled** table is the document's own assertion that a table starts
        here and ends there. Overriding it needs positive evidence, and the
        only evidence that a ruled table continues is that the next one
        repeats its header. Different headers mean different tables.
      * A **text**-strategy grid asserts nothing: it is one page's worth of
        inferred columns, and a dataset spanning 26 pages produces 26 of them
        with no header after the first. Width is the only signal available and
        it is the right one.
    """
    if previous.n_cols != current.n_cols:
        return False
    if previous.strategy != current.strategy:
        return False
    if previous.strategy == "text":
        return True
    return bool(previous.rows and current.rows and previous.rows[0] == current.rows[0])


def _merge(group: Sequence[Grid]) -> List[List[str]]:
    """One group's rows, with each page's repeated header dropped after the first."""
    rows = list(group[0].rows)
    header = rows[0] if rows else None
    for grid in group[1:]:
        rows.extend(row for row in grid.rows if row != header)
    return rows


def _write(path: Path, rows: Sequence[Sequence[str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return str(path)


def pdf_to_csv(
    file_path: str,
    out_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    pages: Optional[Sequence[int]] = None,
    split_fused: bool = False,
    document=None,
) -> ConversionResult:
    """
    Convert `file_path` to CSV on disk and report what happened.

    One combined file is written when every page yields the same number of
    columns — a dataset split across pages is one table to whoever asked for
    it, and handing them 54 files to concatenate is not a conversion. Tables
    of differing shapes are written one file each.

    `out_path` forces a single file; `out_dir` forces one file per table.
    Neither: the source's own name and directory.
    """
    source = Path(file_path).expanduser().resolve()
    if not source.exists():
        raise DocumentIntelligenceError(f"file not found: {file_path}")

    from swarn.capabilities.doc_store import get_or_ingest

    if document is None:
        document, _ = get_or_ingest(str(source))

    wanted = set(pages) if pages else None
    result = ConversionResult()

    grids = [g for g in _stored_grids(document)
             if wanted is None or g.page in wanted]
    covered = {g.page for g in grids}

    # Text strategy only for pages that produced no ruled table — a page that
    # already gave an exact grid should not be second-guessed by an inferred one.
    for page in sorted(p.page_number for p in document.pages):
        if page in covered or (wanted is not None and page not in wanted):
            continue
        grid = _text_grid(str(source), page, document.page(page))
        if grid:
            grids.append(grid)
        else:
            result.pages_with_nothing.append(page)

    grids.sort(key=lambda g: (g.page, g.index))
    for grid in grids:
        grid.n_fused = _count_fused(grid.rows)
        if grid.strategy == "text":
            _repair_header(grid, document)
    result.grids = grids

    if not grids:
        return result

    groups = _group_by_shape(grids)

    # The split decision is taken per GROUP, not per page: a group becomes one
    # file, and splitting some of its pages and not others would leave that
    # file ragged — worse than not splitting at all.
    if split_fused:
        for group in groups:
            if not any(g.n_fused for g in group):
                continue
            candidates = [_normalize(_split_fused(g.rows)) for g in group]
            widths = {max((len(r) for r in c), default=0) for c in candidates}
            if len(widths) == 1 and all(_split_is_aligned(c) for c in candidates):
                for grid, rows in zip(group, candidates):
                    grid.rows, grid.n_fused = rows, 0
            else:
                for grid in group:
                    grid.split_refused = bool(grid.n_fused)

    if out_path:
        target = Path(out_path).expanduser()
        result.paths.append(_write(target, [r for g in groups for r in _merge(g)]))
        result.combined = len(grids) > 1
        return result

    # One folder per PDF, named after it. A document with several tables
    # produces several files, and leaving them loose in whatever directory the
    # PDF happened to sit in mixes them with the tables of every other document
    # converted there. The folder is the unit that corresponds to the source.
    base = Path(out_dir).expanduser() if out_dir else source.parent
    folder = base / source.stem
    result.folder = str(folder)

    for number, group in enumerate(groups, start=1):
        pages = (f"p{group[0].page}" if len(group) == 1
                 else f"p{group[0].page}-{group[-1].page}")
        name = f"table_{number}_{pages}.csv"
        result.paths.append(_write(folder / name, _merge(group)))

    # A manifest, so the folder says what its files are without them having to
    # be opened: which pages each came from, its shape, and how it was found.
    manifest = {
        "source":   str(source),
        "n_tables": len(groups),
        "tables": [
            {
                "file":     Path(path).name,
                "pages":    [g.page for g in group],
                "n_rows":   sum(g.n_rows for g in group),
                "n_cols":   group[0].n_cols,
                "strategy": group[0].strategy,
                "header":   _merge(group)[0] if _merge(group) else [],
            }
            for path, group in zip(result.paths, groups)
        ],
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tables.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    result.combined = any(len(g) > 1 for g in groups)
    return result
