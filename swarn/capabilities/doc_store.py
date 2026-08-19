"""
Persistent structured document store
====================================

Parse a document once; answer questions about it many times.

    swarn ingest report.pdf          # PDF → structured JSON, once
    swarn ask "..." report.pdf       # loads the JSON, no re-parsing
    swarn ask "..." report.pdf       # ...and again

Before this existed, every `swarn ask` re-ran the whole extraction: open the
PDF, pull word geometry off every page (or OCR every page, which is far worse —
tesseract on a 6-page deck is seconds per run), rebuild the lines, throw it all
away after one answer. The extraction is deterministic, so that work produced
byte-identical output every time. Doing it once and keeping the result is the
obvious win, and it becomes a large one for OCR documents, where the cost is
seconds per page rather than milliseconds.

What is stored, and why that specific set
──────────────────────────────────────────
The stored form has to be a complete substitute for re-reading the source, or
the cache is a liability: an answer grounded in stale or lossy data is worse
than one that took longer to produce. So the schema keeps everything the
evidence path consumes — not a summary of it:

    word text + left/top/width/height   the bounding boxes ARE the feature; a
                                        store that dropped coordinates could
                                        answer questions but never show where
    per-word confidence                 OCR's own certainty, which feeds every
                                        field's score downstream
    line_id + line text                 the citation anchor ("p3:L5"). Lines
                                        are grouped ONCE, at ingest, so an id
                                        means the same thing forever
    page width/height                   coordinates are normalized against
                                        these; without them the numbers are
                                        meaningless
    backend                             whether these words were read from a
                                        text layer or transcribed by OCR
    source sha256                       identity, see below

A stored document is therefore enough to rebuild the exact transcript the
model sees and to resolve every citation back to a box, with the original file
never opened.

Identity is content, not path
──────────────────────────────
`document_id` is `<stem>-<sha256(content)[:12]>`. Two consequences, both
wanted: the same file ingested from two different paths hits one cache entry,
and an EDITED file gets a different id, so it misses the cache and is
re-ingested automatically. Keying on the path instead would serve stale text
for a document that had been revised — silently, and with confident citations
into a version that no longer exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from swarn.capabilities.doc_intelligence import (
    ARTIFACTS_DIR,
    DocumentInspector,
    DocumentIntelligenceError,
    _line_text_and_spans,
    _split_columns,
    COLUMN_JOIN_GAP,
    PROSE_SEGMENT_WORDS,
    band_of,
    column_bands,
    group_lines,
)

# Bumped when the stored shape changes in a way older files cannot satisfy.
# A stored document from an older version is re-ingested rather than adapted:
# extraction is cheap and deterministic, so migrating a cache is all risk and
# no benefit.
SCHEMA_VERSION = 5

# Documents live under the same artifacts root the annotated images use, so a
# single SWARN_ARTIFACTS_DIR relocates everything this capability writes.
DOCUMENTS_SUBDIR = "documents"

# Rasterized pages, when `swarn ingest --render-pages` is used. Kept beside the
# JSON so the store can be fully self-contained.
PAGES_SUBDIR = "pages"
# Per-document folders of extracted tables, one CSV each.
TABLES_SUBDIR = "tables"


def documents_dir(artifacts_dir: Optional[Union[str, Path]] = None) -> Path:
    return Path(artifacts_dir or ARTIFACTS_DIR) / DOCUMENTS_SUBDIR


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════


class StoredWord(BaseModel):
    """One word with its box, in PAGE coordinates (PDF points, or pixels for a
    rasterized OCR page — `StoredPage.width/height` says which scale)."""

    model_config = ConfigDict(extra="ignore")

    text: str
    left: float
    top: float
    width: float
    height: float
    confidence: float = 1.0
    size: Optional[float] = None      # font size when known; OCR uses height
    # Emphasis, when the source says so. A PDF marks a heading with size OR
    # weight; without this, a document that uses weight alone structures into
    # nothing. OCR cannot report it, so it stays False there rather than being
    # guessed from stroke thickness.
    bold: bool = False

    def as_positioned(self) -> dict:
        """The plain-dict shape the extraction and evidence helpers in
        doc_intelligence operate on. One conversion point, so the stored
        schema and the in-memory shape can differ without either leaking."""
        return {
            "text": self.text, "left": self.left, "top": self.top,
            "width": self.width, "height": self.height,
            "conf": self.confidence, "size": self.size, "bold": self.bold,
        }

    @classmethod
    def from_positioned(cls, word: dict) -> "StoredWord":
        return cls(
            text=word["text"], left=word["left"], top=word["top"],
            width=word["width"], height=word["height"],
            confidence=word.get("conf", 1.0), size=word.get("size"),
            bold=bool(word.get("bold", False)),
        )


class StoredLine(BaseModel):
    """One visual line, with the words it was built from.

    Lines are grouped at ingest and never regrouped, which is what makes
    `line_id` a durable citation: "p3:L5" has to mean the same line next month
    as it does today, or evidence recorded against it stops being checkable.
    """

    model_config = ConfigDict(extra="ignore")

    line_id: str
    text: str
    # Which page column this line came from (0 for single-column pages).
    # Recorded so a consumer can tell that two adjacent lines belong to
    # different columns even though nothing in the text says so.
    column: int = 0
    words: List[StoredWord] = Field(default_factory=list)

    def word_dicts(self) -> List[dict]:
        return [word.as_positioned() for word in self.words]


class StoredCell(BaseModel):
    """One table cell: its column, its text, and where it sits on the page.

    The bbox is what makes table-aware evidence possible. A citation that
    combines two columns ("<trigger> CC 5") exists on no single text line, so
    it can only be boxed by unioning the boxes of the cells it came from —
    which requires every cell to carry its own geometry.
    """

    model_config = ConfigDict(extra="ignore")

    column: str = ""                  # header name, or colN when the table has no header row
    row_index: int = 0
    col_index: int = 0
    text: str = ""
    bbox: List[float] = Field(default_factory=list)     # [x0, y0, x1, y1], page coords


class StoredRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = 0
    cells: List[StoredCell] = Field(default_factory=list)

    def cell(self, column: str) -> Optional[StoredCell]:
        return next((c for c in self.cells if c.column == column), None)

    def texts(self) -> List[str]:
        return [cell.text for cell in self.cells]


class StoredTable(BaseModel):
    """A table as rows of cells, each cell carrying its own bounding box.

    Reconstructed at ingest by DocumentInspector.page_tables(), which merges
    the padding columns and hairline rows a styled PDF reports into the rows
    and columns a human actually reads — including rejoining a cell whose text
    wraps across several lines.
    """

    model_config = ConfigDict(extra="ignore")

    index: int = 1
    page: int = 1
    bbox: List[float] = Field(default_factory=list)
    headers: List[str] = Field(default_factory=list)
    rows: List[StoredRow] = Field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.headers) or (len(self.rows[0].cells) if self.rows else 0)

    def text_rows(self) -> List[List[str]]:
        """The plain text grid, for callers that only want the values."""
        return [row.texts() for row in self.rows]

    def records(self) -> List[dict]:
        """{column: value} per row, excluding the header row. Empty when the
        table has no usable header — keying on positional names would invent
        column identities the document never stated."""
        if not self.headers or len(set(self.headers)) != len(self.headers):
            return []
        return [dict(zip(self.headers, row.texts())) for row in self.rows[1:]]


class StoredPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_number: int = Field(ge=1)
    width: float
    height: float
    lines: List[StoredLine] = Field(default_factory=list)
    tables: List[StoredTable] = Field(default_factory=list)
    image_path: str = ""              # cached raster, when --render-pages was used

    # ── the transcript view the Q&A pipeline consumes ──

    @property
    def page_size(self) -> Tuple[int, int]:
        return int(self.width), int(self.height)

    def render(self) -> str:
        return "\n".join(f"[{line.line_id}] {line.text}" for line in self.lines)

    def find(self, line_id: str) -> Optional[dict]:
        """Look up a cited line. Returns the same {line_id, text, words} shape
        the in-memory transcript used, so evidence resolution is unchanged by
        the introduction of the store."""
        for line in self.lines:
            if line.line_id == line_id:
                return {"line_id": line.line_id, "text": line.text,
                        "words": line.word_dicts()}
        return None

    def all_words(self) -> List[dict]:
        return [word for line in self.lines for word in line.word_dicts()]


class StoredDocument(BaseModel):
    """A document parsed once, in full."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = SCHEMA_VERSION
    document_id: str
    document_name: str
    source_path: str
    source_sha256: str
    page_count: int
    backend: str                      # 'text' or 'ocr' — how the words were read
    ingested_at: str
    # The source's own metadata (PDF /Info: Title, Author, Producer, ...).
    # Stored because it is read during the one parse and is otherwise only
    # recoverable by reopening the file, which defeats the point of the store.
    metadata: Dict[str, str] = Field(default_factory=dict)
    # Folder holding this document's tables as CSV, written at ingest. Empty
    # when the document has no tables, or when export was switched off.
    tables_dir: str = ""
    pages: List[StoredPage] = Field(default_factory=list)

    def page(self, page_number: int) -> Optional[StoredPage]:
        return next((p for p in self.pages if p.page_number == page_number), None)

    def n_lines(self) -> int:
        return sum(len(page.lines) for page in self.pages)

    def n_words(self) -> int:
        return sum(len(line.words) for page in self.pages for line in page.lines)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# IDENTITY
# ══════════════════════════════════════════════════════════════════════════════


def file_sha256(file_path: Union[str, Path]) -> str:
    """Content hash, streamed so a large PDF is not read into memory at once."""
    digest = hashlib.sha256()
    with open(Path(file_path).expanduser().resolve(), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id_for(file_path: Union[str, Path], sha256: Optional[str] = None) -> str:
    """`<sanitized-stem>-<sha12>`: readable in a directory listing, and unique
    per CONTENT so an edited file cannot reuse a stale entry."""
    path = Path(file_path)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_") or "document"
    return f"{stem}-{(sha256 or file_sha256(path))[:12]}"


def store_path(document_id: str, artifacts_dir: Optional[Union[str, Path]] = None) -> Path:
    return documents_dir(artifacts_dir) / f"{document_id}.json"


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION
# ══════════════════════════════════════════════════════════════════════════════


def _table_index_of(segment: Sequence[dict], tables: Sequence[StoredTable]):
    """Which detected table this segment sits in, or None.

    Returns the identity of the table, not just a yes/no, because "both of
    these are in a table" is not the same claim as "both of these are in the
    SAME table" — and only the second one makes two segments a row. A page
    holding two tables side by side satisfies the first and must not be joined.
    """
    if not segment:
        return None
    x = sum(w["left"] + w["width"] / 2 for w in segment) / len(segment)
    y = sum(w["top"] + w["height"] / 2 for w in segment) / len(segment)
    for index, table in enumerate(tables):
        if (len(table.bbox) == 4 and table.bbox[0] - 1 <= x <= table.bbox[2] + 1
                and table.bbox[1] - 1 <= y <= table.bbox[3] + 1):
            return index
    return None


def _is_prose(segment: dict) -> bool:
    """Does this segment read as a sentence rather than a cell value?"""
    return len(segment["text"].split()) > PROSE_SEGMENT_WORDS


def _same_row(previous: dict, current: dict, tables: Sequence[StoredTable],
              page_width: float) -> bool:
    """
    Are these two side-by-side segments cells of ONE row, or two page columns?

    Neither signal is sufficient alone, and each covers the other's blind spot:

      * **Detected table** — authoritative when present, and it has to be,
        because a table's own columns can sit further apart than some page
        gutters (a contacts table measured 43% of page width). Distance alone
        would shred those rows.
      * **Distance** — the fallback when no table is detected, which is the
        common case for a table drawn without ruling. `find_tables()` sees only
        ruled tables, so relying on detection alone shredded three borderless
        tables in a document that reads perfectly well to a human
        (`Date | Activity` became two lines). Their cells sit 2-6% apart, far
        under the threshold, so proximity rescues them.

    A page gutter is the case both signals agree on: no table, and a chasm
    (measured 54% of page width on the two-column slide that motivated this).
    """
    left_table = _table_index_of(previous["words"], tables)
    right_table = _table_index_of(current["words"], tables)
    if left_table is not None and left_table == right_table:
        return True
    # In two DIFFERENT tables: side-by-side tables, whose rows are unrelated
    # however close they sit. Never a row.
    if left_table is not None and right_table is not None:
        return False
    gap = current["left"] - max(w["left"] + w["width"] for w in previous["words"])
    if gap >= COLUMN_JOIN_GAP * max(page_width, 1.0):
        return False

    # The distance rule has now said "close enough to be one row". It assumes
    # page gutters are wide; on a two-column notice block they need not be. The
    # UPPCL bill's is 4-6% of page width, *narrower* than the borderless-table
    # cells (2-5%) the rule exists to rescue, so distance cannot separate them.
    # Neither can the other geometric signals: that bill's two columns share
    # left-edge alignment and baselines exactly, like a table's, and its
    # left-edge distribution is a continuum, so column_bands sees one band.
    #
    # Content length does separate them. A table cell is a value ("24 Jul",
    # "Idea submission closes"); a page column is a sentence. Both sides being
    # prose is the signal.
    #
    # This trades one error for a rarer one, in the safe direction. Merging two
    # prose columns fabricates an adjacency the document never made — the
    # failure that made a model name the mentor as a team member. Splitting a
    # borderless table whose cells are long sentences loses a row grouping, but
    # every line it produces is still text that is genuinely on the page. An
    # invented relationship is worse than a missing one.
    if _is_prose(previous) and _is_prose(current):
        return False
    return True


def _lines_for_page(
    page_number: int,
    words: Sequence[dict],
    tables: Sequence[StoredTable] = (),
    page_width: float = 0.0,
) -> List[StoredLine]:
    """
    Group positioned words into the numbered lines the transcript uses.

    Run ONCE here rather than on every question. This is the single place lines
    are defined; PageTranscript reads them rather than recomputing them, so a
    stored `line_id` and a freshly computed one cannot drift apart.

    Two rules decide what a "line" is, and both exist because a horizontal band
    of a page is not automatically one statement:

    1. Segments are joined with " | " ONLY inside a detected table.
       In a table row the cells genuinely belong together, and the pipes are
       what let a model read `Revenue | 120100 | 148200` as one record. Outside
       a table the same join asserts a relationship the document never made:
       on a two-column slide it produced

           TEAM MEMBERS | Jaimin Nalin Desai

       where "TEAM MEMBERS" is a caption in the LEFT column and the name is a
       value in the RIGHT one, under its own "MENTOR NAME" caption. A model
       reading that will say Jaimin is a team member, and it will be right to,
       because that is what the transcript said. Outside a table each segment
       therefore becomes its own line.

    2. A multi-column page is emitted in COLUMN-MAJOR reading order — the left
       column top to bottom, then the next. Un-joining alone is not enough:
       ordered by vertical position, the name still lands immediately under the
       other column's caption, and the ambiguity survives. Reading order is how
       a human resolves it, so it is how the transcript should present it.

    Reordering is applied only when the page has no detected tables and more
    than one column band — the conservative case. A page mixing a table with
    page columns keeps positional order, which is never wrong, only less
    helpful.
    """
    visual_lines = group_lines(words)

    # Flatten to segments, remembering which visual line each came from so a
    # table row can be reassembled from its parts.
    entries: List[dict] = []
    for line_index, visual_line in enumerate(visual_lines):
        for segment in _split_columns(visual_line):
            text = _line_text_and_spans(segment)[0]
            if not text.strip():
                continue
            entries.append({
                "line_index": line_index,
                "text": text,
                "words": segment,
                "left": min(w["left"] for w in segment),
                "top": min(w["top"] for w in segment),
            })
    if not entries:
        return []

    # Table rows keep their cells together on one line; everything else stands
    # alone. Grouping by the visual line they came from is what rejoins a row.
    units: List[dict] = []
    for entry in entries:
        previous = units[-1] if units else None
        if (previous and previous["line_index"] == entry["line_index"]
                and _same_row(previous, entry, tables, page_width)):
            previous["text"] += " | " + entry["text"]
            previous["words"] = list(previous["words"]) + list(entry["words"])
        else:
            units.append(dict(entry))

    # Column-major ordering whenever the page really has columns. A joined row
    # is one unit at its table's left edge, so every row of a table lands in
    # the same band and stays contiguous and in order — which is why this no
    # longer needs to bail out on pages that contain tables. It used to, and
    # that left a page holding two SIDE-BY-SIDE tables interleaving their rows.
    bands = column_bands([u["left"] for u in units], page_width) if page_width else []
    if len(bands) > 1:
        units.sort(key=lambda u: (band_of(u["left"], bands), u["top"], u["left"]))
        for unit in units:
            unit["column"] = band_of(unit["left"], bands)
    else:
        units.sort(key=lambda u: (u["top"], u["left"]))
        for unit in units:
            unit["column"] = 0

    return [
        StoredLine(
            line_id=f"p{page_number}:L{index}",
            text=unit["text"],
            column=unit["column"],
            words=[StoredWord.from_positioned(word) for word in unit["words"]],
        )
        for index, unit in enumerate(units, start=1)
    ]


def _tables_for_page(
    inspector: DocumentInspector, file_path: str, page_number: int,
) -> List[StoredTable]:
    """
    Structured tables for one page, with per-cell bounding boxes.

    Delegates to DocumentInspector.page_tables() — table reconstruction is
    extraction, and belongs in the extraction layer next to page_words(), not
    here. This function only maps its output into the stored schema.
    """
    tables = []
    for index, raw in enumerate(inspector.page_tables(file_path, page_number), start=1):
        rows = [
            StoredRow(index=row_index, cells=[StoredCell(**cell) for cell in cells])
            for row_index, cells in enumerate(raw.get("rows", []))
        ]
        if not rows:
            continue
        tables.append(StoredTable(
            index=index, page=page_number,
            bbox=raw.get("bbox", []), headers=raw.get("headers", []), rows=rows,
        ))
    return tables


def _source_metadata(source: Path) -> Dict[str, str]:
    """The PDF's own /Info dictionary, best-effort.

    Never raises: metadata is a nicety next to the words, and a producer that
    writes a malformed /Info must not cost the caller the whole parse.
    """
    if source.suffix.lower() != ".pdf":
        return {}
    try:
        import pdfplumber
        with pdfplumber.open(str(source)) as pdf:
            return {str(k): str(v) for k, v in (pdf.metadata or {}).items()}
    except Exception:  # noqa: BLE001
        return {}


def ingest_document(
    file_path: str,
    inspector: Optional[DocumentInspector] = None,
    backend: Optional[str] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
    render_pages: bool = False,
    save: bool = True,
    export_csv: bool = True,
) -> StoredDocument:
    """
    Parse a document in full and return (optionally persisting) its structured form.

    This is the ONLY place the source file is read for text. Extraction itself
    is delegated to DocumentInspector.page_words(), so ingestion inherits the
    existing backend rules unchanged — `auto` resolves to the PDF text layer,
    else local OCR, and never to the synthetic mock.

    `render_pages` additionally caches a raster of each page beside the JSON.
    Off by default because it costs real disk (~200 KB/page) and the source
    file is normally still there; on, the store is entirely self-contained and
    evidence images can be drawn with the original PDF absent.
    """
    source = Path(file_path).expanduser().resolve()
    if not source.exists():
        raise DocumentIntelligenceError(f"file not found: {file_path}")

    inspector = inspector or DocumentInspector(artifacts_dir=artifacts_dir)
    sha256 = file_sha256(source)
    doc_id = document_id_for(source, sha256)

    _, page_count = inspector.load_page_image(str(source), 1)

    pages: List[StoredPage] = []
    chosen = ""
    for page_number in range(1, page_count + 1):
        words, page_size, chosen = inspector.page_words(str(source), page_number, backend)
        image_path = ""
        if render_pages:
            image, _ = inspector.load_page_image(str(source), page_number)
            target = (documents_dir(artifacts_dir) / PAGES_SUBDIR /
                      f"{doc_id}_p{page_number}.png")
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            image_path = str(target)

        # Tables first: line grouping needs to know which segments are table
        # cells (join them) and which are separate page columns (do not).
        page_tables = _tables_for_page(inspector, str(source), page_number)
        pages.append(StoredPage(
            page_number=page_number,
            width=page_size[0], height=page_size[1],
            lines=_lines_for_page(page_number, words, page_tables, page_size[0]),
            tables=page_tables,
            image_path=image_path,
        ))

    document = StoredDocument(
        document_id=doc_id,
        document_name=source.name,
        source_path=str(source),
        source_sha256=sha256,
        page_count=page_count,
        backend=chosen,
        ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        metadata=_source_metadata(source),
        pages=pages,
    )
    if export_csv:
        document.tables_dir = _export_tables(document, artifacts_dir)
    if save:
        save_document(document, artifacts_dir)
    return document


def _export_tables(document: StoredDocument,
                   artifacts_dir: Optional[Union[str, Path]]) -> str:
    """
    Write this document's tables to CSV beside its stored JSON, at ingest.

    A document is parsed once; the tables are already in hand at that moment,
    and a caller who uploads a PDF holding tables wants those tables as files,
    not as a nested array inside a JSON blob they have to write code against.
    Doing it here means it happens however the document arrived — `swarn
    ingest`, the first `swarn ask`, or the agent's swarn_doc_ingest.

    Never fatal: a document that cannot be tabulated is still a perfectly good
    stored document, and failing the parse over a CSV would be absurd.
    """
    try:
        from swarn.capabilities.doc_csv import pdf_to_csv

        result = pdf_to_csv(
            document.source_path,
            out_dir=str(documents_dir(artifacts_dir) / TABLES_SUBDIR),
            document=document,
        )
        return result.folder or ""
    except Exception:  # noqa: BLE001
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════


def save_document(
    document: StoredDocument, artifacts_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Write atomically: a half-written store read by a concurrent `ask` would
    fail in a far more confusing way than a missing one."""
    destination = store_path(document.document_id, artifacts_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(document.to_json(), encoding="utf-8")
    os.replace(temporary, destination)
    return str(destination)


def load_document(
    document_id: str, artifacts_dir: Optional[Union[str, Path]] = None,
) -> Optional[StoredDocument]:
    """Load by id, or None if absent/unreadable/outdated.

    A corrupt or old-schema file returns None rather than raising, so the
    caller re-ingests. A cache that can break the tool it is meant to
    accelerate is not worth having.
    """
    path = store_path(document_id, artifacts_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            return None
        return StoredDocument.model_validate(payload)
    except Exception:                                                  # noqa: BLE001
        return None


def load_for_file(
    file_path: str, artifacts_dir: Optional[Union[str, Path]] = None,
) -> Optional[StoredDocument]:
    """The stored form of this exact file content, or None if not ingested.

    Hashing the file is a read of its bytes, not a parse — no pdfplumber, no
    OCR, no page rendering. That is what makes the check cheap enough to run
    on every question while still catching an edited document.
    """
    source = Path(file_path).expanduser().resolve()
    if not source.exists():
        raise DocumentIntelligenceError(f"file not found: {file_path}")
    return load_document(document_id_for(source, file_sha256(source)), artifacts_dir)


def get_or_ingest(
    file_path: str,
    inspector: Optional[DocumentInspector] = None,
    backend: Optional[str] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
    on_ingest=None,
) -> Tuple[StoredDocument, bool]:
    """
    Return (document, was_ingested_now).

    The load-or-parse seam every caller goes through. `on_ingest` is called
    with the file path when a parse is about to happen, so the CLI can say so
    — a first-run pause on a scanned document is a long silence otherwise, and
    an unexplained one.
    """
    stored = load_for_file(file_path, artifacts_dir)
    if stored is not None:
        # A store built by a different backend than the caller now demands is
        # not the document they asked for — re-ingest rather than silently
        # answering from OCR text when 'text' was requested.
        if backend and backend not in ("auto", stored.backend):
            stored = None
        else:
            return stored, False

    if on_ingest:
        on_ingest(file_path)
    return ingest_document(file_path, inspector=inspector, backend=backend,
                           artifacts_dir=artifacts_dir), True


def list_documents(artifacts_dir: Optional[Union[str, Path]] = None) -> List[dict]:
    """Summaries of every ingested document, newest first."""
    directory = documents_dir(artifacts_dir)
    if not directory.exists():
        return []
    summaries = []
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                              # noqa: BLE001
            continue
        summaries.append({
            "document_id": payload.get("document_id", path.stem),
            "document_name": payload.get("document_name", ""),
            "page_count": payload.get("page_count", 0),
            "backend": payload.get("backend", ""),
            "ingested_at": payload.get("ingested_at", ""),
            "source_path": payload.get("source_path", ""),
            "stored_at": str(path),
            "size_kb": round(path.stat().st_size / 1024, 1),
        })
    return sorted(summaries, key=lambda item: item["ingested_at"], reverse=True)


__all__ = [
    "DOCUMENTS_SUBDIR",
    "SCHEMA_VERSION",
    "StoredDocument",
    "StoredLine",
    "StoredPage",
    "StoredTable",
    "StoredWord",
    "document_id_for",
    "documents_dir",
    "file_sha256",
    "get_or_ingest",
    "ingest_document",
    "list_documents",
    "load_document",
    "load_for_file",
    "save_document",
    "store_path",
]
