"""
Phase 13: Multi-Modal RAG

Extends Phase 3's repo-RAG (context_engine.py) so the same searchable
index can hold PDFs, images, and audio — not just source code and
markdown. The goal, per the original blueprint, is "multi-modal RAG ...
with citation-grounded generation": search results should always say
exactly where a fact came from (file + page, or file + timestamp), the
same way Phase 3's search() already reports file + line range for code.

Why this reuses ContextEngine instead of building a parallel system
─────────────────────────────────────────────────────────────────────────
Phase 3's ContextEngine already solved "chunk → embed → upsert to one
ChromaDB collection → cosine-similarity search," and a real research
question is rarely "search only the code" or "search only the PDFs" —
it's "find whatever's relevant," which could be a function, a table in
a spec PDF, or a caption on a diagram. Building a second, separate
vector store for non-text content would mean every search has to query
two systems and merge results, and a query like "where do we define the
rate limit" might need to find both a PDF row AND a Python constant.

So MultiModalIndexer doesn't replace or wrap ContextEngine — it reuses
the EXACT SAME singleton, collection, and embedder
(get_context_engine()._collection / ._embedder / ._ensure_ready()), and
just adds new ingestion paths that produce chunks in the same
{"content", "metadata"} shape ContextEngine._make_chunk() already uses,
tagged with new `type` values (pdf_text, pdf_table, image_ocr,
image_caption, audio_transcript) alongside Phase 3's existing
module_header/FunctionDef/ClassDef/text_chunk types. One search() call
— Phase 3's, unmodified — now returns a blend of all of them, ranked
purely by relevance, with the `type` field telling you what kind of
source a given result came from.

Per-modality extraction strategy
─────────────────────────────────
  PDF      pdfplumber extracts both prose text (chunked like Phase 3's
           sliding-window text chunker) and tables (kept as a single
           pipe-delimited chunk per table — splitting a table mid-row
           would destroy the only thing that makes it useful, the
           row/column alignment).
  Image    pytesseract OCR extracts any text the image contains
           (diagrams with labels, screenshots, scanned pages saved as
           images). Separately, if a CLIP-family sentence-transformers
           model is available, the image itself is embedded directly
           (not just its OCR text) so semantic image search works even
           for images with no text at all — "a photo of a server rack"
           should be findable by meaning, not by hoping there's a
           caption.
  Audio    openai-whisper transcribes to text, chunked the same way as
           PDF prose, with timestamps as the citation anchor instead of
           page numbers.

All three are lazy-imported, exactly like Phase 3's sentence-transformers/
chromadb — a missing optional dependency is reported as a clear error
string from the specific tool that needed it, not a startup crash for
the whole agent.
"""

import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from agent.memory.context_engine import get_context_engine, ContextEngine

MAX_PDF_BYTES   = 50_000_000   # 50 MB — large scanned PDFs can run OCR-equivalent extraction for a long time
MAX_IMAGE_BYTES = 20_000_000
MAX_AUDIO_BYTES = 200_000_000  # whisper transcription is slow; cap to keep a single call bounded

PDF_TEXT_CHUNK_LINES = 40   # smaller than Phase 3's CHUNK_LINES=60 — PDF text reflows oddly per-page

# ── Document-structure tuning (extract_pdf_document) ────────────────────
# A line counts as a heading when it's meaningfully LARGER than body text.
# 1.12 is deliberately loose: many documents step h2 only ~1.1× above body,
# and a missed heading costs a whole section boundary, while a false one
# only adds a spurious (still readable) section.
HEADING_SIZE_RATIO   = 1.12
# A bold line can also be a heading at body size — but only if it's short
# and doesn't read like a sentence, or every bold run-in emphasis would
# split the document.
HEADING_MAX_CHARS    = 90
# Vertical gap (as a multiple of the line's own height) that ends a
# paragraph. Normal leading is ~1.2×; anything past this is a real break.
PARA_BREAK_GAP_RATIO = 1.75
# Longest string still plausible as a field LABEL in "Key: value".
KV_MAX_KEY_CHARS     = 45

# Bullet/enumerator forms, plus `(cid:N)`: when a PDF embeds a bullet in a
# font whose glyph has no Unicode mapping, every extractor emits that
# literal placeholder. It shows up constantly in real documents, and
# without this branch each bullet line reads as a field named "(cid".
_LIST_MARKER_RE = re.compile(
    r"^\s*(?:\(cid:\d+\)|[•·◦▪‣∙*\-–—]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)])\s*")
_KV_RE          = re.compile(r"^\s*([^:\n]{1,%d}?)\s*[::]\s*(\S.*)$" % KV_MAX_KEY_CHARS)
# A field LABEL is a short noun phrase. Requiring it to start with a letter
# (or #) and hold only label-ish characters rejects the common mis-parses:
# "(cid" from an unmapped glyph, a bare number from a citation, a URL's
# "https" before its "//".
_KEY_OK_RE      = re.compile(r"^[A-Za-z#][A-Za-z0-9 \-/&.'’#]*$")
KV_MAX_KEY_WORDS   = 5
KV_MAX_VALUE_CHARS = 120
# A field label is a noun phrase — it has no verb and no pronoun subject.
# Prose that merely contains a colon almost always does: "There was one
# conclusion: the rollout should continue" would otherwise become a field
# named "There was one conclusion". Checking for these words is a cheap
# stand-in for the parse that would prove it.
_NON_LABEL_WORDS = frozenset("""
    is are was were be been being am has have had do does did will would
    shall should can could may might must that which there this these those
    it they we you he she who whose when while because if
""".split())
# A ". " followed by a capital means the "value" is really flowing prose
# that happens to contain a colon — not a field.
_SENTENCE_RE    = re.compile(r"[.!?]\s+[A-Z(]")


class MultiModalIndexer:
    """
    Stateless ingestion logic that feeds Phase 3's ContextEngine. Holds
    no index state of its own — every method reads/writes through
    get_context_engine(), so a PDF indexed here and a Python file
    indexed via Phase 3's index_project both land in the same
    searchable collection.
    """

    # ───────────────────────────────────────────────── PDF

    def index_pdf(self, path: str) -> str:
        """
        Extract prose text (chunked, sliding-window) and tables (one
        chunk per table, pipe-delimited) from a PDF, embed them, and
        upsert into the same ChromaDB collection Phase 3's
        index_project() uses. Citation metadata is (file, page) instead
        of (file, line range) — set via the same _make_chunk shape, with
        start_line/end_line repurposed to mean the page number (both set
        to the same value, since a PDF chunk doesn't span pages here).

        Table detection note: pdfplumber's extract_tables() finds tables
        by visual structure — ruled lines or consistent column gaps —
        not just text that happens to be column-aligned. A PDF where
        tabular data was laid out as plain positioned text without any
        ruling will have that data picked up by the prose-text path
        instead (still indexed and searchable, just not chunked as a
        single coherent pipe-delimited table). This was confirmed during
        testing: a reportlab-generated page using raw positioned text
        for a table found 0 tables via extract_tables(), while the same
        data laid out with an actual GRID-style table style was detected
        correctly.
        """
        try:
            import pdfplumber
        except ImportError:
            return "Error: PDF indexing requires 'pip install pdfplumber'."

        engine = get_context_engine()
        err = engine._ensure_ready()
        if err:
            return f"Error: {err}"

        full = Path(path).resolve()
        if not full.exists():
            return f"Error: file not found: {path}"
        if full.stat().st_size > MAX_PDF_BYTES:
            return f"Error: {path} exceeds the {MAX_PDF_BYTES // 1_000_000} MB indexing limit."

        chunks: list[dict] = []
        try:
            with pdfplumber.open(str(full)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        chunks.extend(self._chunk_pdf_text(text, str(full), page_num))

                    for table_idx, table in enumerate(page.extract_tables(), start=1):
                        rendered = self._render_table(table)
                        if rendered:
                            chunks.append(self._make_pdf_chunk(
                                rendered, str(full), "pdf_table",
                                f"{full.name}:p{page_num}:table{table_idx}", page_num,
                            ))
        except Exception as e:
            return f"Error reading PDF '{path}': {type(e).__name__}: {e}"

        if not chunks:
            return f"No extractable text or tables found in {path} (it may be a scanned/image-only PDF — try index_image on a rasterized page instead)."

        n_pages = len({c["metadata"]["start_line"] for c in chunks})
        self._embed_and_upsert(engine, chunks)
        return (
            f"Indexed {path}: {n_pages} pages with content → {len(chunks)} chunks "
            f"(text + tables). Searchable via search_codebase alongside any indexed code."
        )

    def _chunk_pdf_text(self, text: str, file_path: str, page_num: int) -> list[dict]:
        lines = text.splitlines()
        chunks = []
        i = 0
        while i < len(lines):
            end = min(i + PDF_TEXT_CHUNK_LINES, len(lines))
            piece = "\n".join(lines[i:end]).strip()
            if piece:
                chunks.append(self._make_pdf_chunk(
                    piece, file_path, "pdf_text",
                    f"{Path(file_path).name}:p{page_num}:chunk{i}", page_num,
                ))
            i = end
        return chunks

    @staticmethod
    def _render_table(table: list[list]) -> str:
        """Pipe-delimited rendering — keeps row/column alignment readable in a text chunk."""
        rows = []
        for row in table:
            cells = [str(c) if c is not None else "" for c in row]
            rows.append(" | ".join(cells))
        return "\n".join(r for r in rows if r.strip(" |"))

    @staticmethod
    def _make_pdf_chunk(content: str, file: str, kind: str, name: str, page: int) -> dict:
        # Reuses ContextEngine._make_chunk's exact shape so Phase 3's
        # search() needs zero changes to display these results —
        # start_line/end_line both hold the page number here, since a
        # PDF chunk's "location" is a page, not a line range.
        return ContextEngine._make_chunk(content, file, kind, name, page, page)

    # ──────────────────────────────────── PDF → structured data

    def extract_pdf_structured(self, path: str) -> dict:
        """
        Return a PDF's contents as STRUCTURED DATA — rows and columns kept
        as nested lists/dicts — instead of the flattened text chunks
        index_pdf() produces.

        Why this exists as a separate method rather than a flag on
        index_pdf: the two have genuinely opposite goals. index_pdf feeds
        the vector store, where everything must end up as one embeddable
        string, so _render_table() deliberately flattens a table's
        list-of-lists into pipe-delimited text — the structure pdfplumber
        recovered is discarded on purpose, because an embedding of "rows
        and columns" isn't a thing. That's the right call for semantic
        search and the wrong call for anything programmatic: you can't sum
        a column, feed a DataFrame, or validate a field out of a prose
        blob. This method exits one step earlier, at the point where
        extract_tables() has already handed back real structure, and
        returns it untouched.

        Two consequences worth knowing, both deliberate:

          * No embedding, no ChromaDB, no network. index_pdf calls
            _ensure_ready(), which downloads a ~90 MB sentence-transformers
            model on first use. Extraction needs none of that, so this
            path works offline and in a sandbox — and stays fast.
          * Nothing is indexed. If you want the PDF searchable too, call
            index_pdf as well; the two are independent.

        Shape of the returned dict:

            {
              "file": "/abs/path.pdf",
              "n_pages": 3,              # pages in the document
              "n_tables": 2,             # tables detected across all pages
              "pages": [
                {
                  "page": 1,                       # 1-based, matches the citation anchor index_pdf uses
                  "text": "prose text of the page",
                  "tables": [
                    {
                      "index": 1,                  # 1-based within this page
                      "n_rows": 8, "n_cols": 6,
                      "header": ["Line", "SKU", ...],
                      "rows": [["1", "CMP-M4", ...], ...],   # data rows, header excluded
                      "records": [{"Line": "1", "SKU": "CMP-M4", ...}, ...]
                    }
                  ]
                }
              ]
            }

        `rows` is the raw grid (what you want for a DataFrame or a
        spreadsheet write-out); `records` is the same data keyed by header
        (what you want to index by field name, or hand to an LLM with a
        target schema). `records` is omitted when the header row can't be
        trusted — see _table_to_structured.

        Errors follow this codebase's contract of never raising: a problem
        comes back as {"error": "..."} , which the tools.py wrapper
        serializes like any other result.

        Table-detection caveat is the same one index_pdf documents above:
        pdfplumber finds tables by VISUAL structure (ruled lines or
        consistent column gaps). A "table" that's really just positioned
        text with no ruling won't appear in `tables` at all — its content
        lands in that page's `text` instead. If a table you can see is
        missing from the output, that's what happened, and the fix is at
        the PDF's end, not here.
        """
        try:
            import pdfplumber
        except ImportError:
            return {"error": "PDF extraction requires 'pip install pdfplumber'."}

        full = Path(path).resolve()
        if not full.exists():
            return {"error": f"file not found: {path}"}
        if full.stat().st_size > MAX_PDF_BYTES:
            return {"error": f"{path} exceeds the {MAX_PDF_BYTES // 1_000_000} MB limit."}

        pages: list[dict] = []
        n_tables = 0
        try:
            with pdfplumber.open(str(full)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = []
                    for table_idx, raw in enumerate(page.extract_tables(), start=1):
                        parsed = self._table_to_structured(raw, table_idx)
                        if parsed:
                            tables.append(parsed)
                            n_tables += 1
                    pages.append({
                        "page":   page_num,
                        "text":   (page.extract_text() or "").strip(),
                        "tables": tables,
                    })
        except Exception as e:
            return {"error": f"reading PDF '{path}': {type(e).__name__}: {e}"}

        return {
            "file":     str(full),
            "n_pages":  len(pages),
            "n_tables": n_tables,
            "pages":    pages,
        }

    @staticmethod
    def _clean_cell(cell) -> str:
        """
        Normalize one pdfplumber cell. Empty cells come back as None, and a
        wrapped cell keeps the literal newline the PDF laid out — neither
        is useful downstream, where a cell should be a plain single-line
        string.
        """
        if cell is None:
            return ""
        return " ".join(str(cell).split())

    @classmethod
    def _table_to_structured(cls, table: list[list], index: int) -> Optional[dict]:
        """
        Turn one raw pdfplumber table into the structured dict described in
        extract_pdf_structured's docstring. Returns None for a table with
        no content at all (extract_tables() occasionally returns an empty
        grid for a stray ruled box).

        The header decision: the first row is treated as a header only if
        every one of its cells is non-empty and no name repeats. That's
        strict on purpose — a header with blanks or duplicates would
        silently collapse columns when zipped into a dict, losing data
        with no error. When it fails, `header` is null and `rows` holds
        EVERY row including the first, so nothing is dropped; `records` is
        simply absent, and the caller works off the grid.
        """
        grid = [[cls._clean_cell(c) for c in row] for row in table]
        grid = [row for row in grid if any(cell for cell in row)]
        if not grid:
            return None

        n_cols = max(len(row) for row in grid)
        # Pad short rows so every row is the same width — a merged cell can
        # make pdfplumber return a row with fewer entries than its
        # neighbours, and ragged rows break both zip() and DataFrame().
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

    # ────────────────────── whole PDF → structured document

    def extract_pdf_document(self, path: str) -> dict:
        """
        Structure an ENTIRE PDF — not just its tables — into a typed
        document tree: sections with headings, and inside them typed
        blocks (paragraph / list / key_values / table), all in reading
        order, plus a flat `fields` map of every label→value pair found
        anywhere in the document.

        How this differs from the two methods above
        ────────────────────────────────────────────
        index_pdf gives you an embeddable blob; extract_pdf_structured
        gives you tables as real rows but leaves everything else as one
        undifferentiated `text` string per page. That string is where
        most of a document's actual data lives — headings that define
        what a section is about, "Invoice date: 2024-09-17" field pairs,
        bulleted findings — and none of it is addressable while it's a
        wall of text. This method types all of it.

        How the structure is recovered (deterministic, no LLM, no network)
        ──────────────────────────────────────────────────────────────────
        PDFs carry no semantic markup — there is no <h1>, no <li>, not
        even a reliable paragraph mark. What a PDF does carry is layout:
        glyph sizes, font names, and coordinates. So every judgement here
        is made from layout evidence:

          Headings   a line whose font size exceeds the document's body
                     size by HEADING_SIZE_RATIO, or a short bold line
                     that doesn't read like a sentence. Distinct heading
                     sizes are ranked largest-first to assign `level`
                     (1, 2, 3...), so an h2 nests under the h1 above it.
          Body size  the char-count-weighted most common size in the
                     document — weighting by characters, not by line,
                     stops a title page of large text from being mistaken
                     for the body.
          Paragraphs consecutive body lines joined into one string, broken
                     when the vertical gap exceeds PARA_BREAK_GAP_RATIO or
                     the page changes. Hyphenated line-ends are rejoined.
          Lists      consecutive lines starting with a bullet glyph or an
                     enumerator (1. / a) / •). Markers are stripped from
                     the stored items — the marker is presentation, the
                     text is the data.
          Key/values consecutive "Label: value" lines, collected into one
                     `key_values` block. Also mined from inside prose
                     sentence-by-sentence, which is how a paragraph like
                     "Invoice date: 2024-09-17. Payment terms: Net 30."
                     still yields both fields.
          Tables     detected exactly as extract_pdf_structured does, and
                     — importantly — their words are REMOVED from the text
                     layer before lines are built, so table contents no
                     longer appear twice (once as a table, once as
                     garbled prose). extract_pdf_structured has that
                     duplication; this method does not.

        Because it is heuristic, layout evidence can mislead: a document
        that styles body text bold, or one with no size variation at all,
        will produce fewer sections (everything lands in a single
        heading-less section) rather than wrong ones — the failure mode
        is under-structuring, not invented structure.

        Returns
        ───────
            {
              "file": "/abs/path.pdf",
              "n_pages": 2,
              "metadata": {"Title": ..., "Author": ..., ...},   # PDF's own metadata
              "title": "Quarterly Infrastructure Review",        # largest text on page 1, or metadata Title
              "fields": {"Report ID": "QIR-2024-Q3", ...},       # every label→value found, document-wide
              "sections": [
                {
                  "heading": "Executive Summary",   # null for content before the first heading
                  "level": 1,                        # 0 when heading is null
                  "page": 1,                         # page the heading appears on
                  "blocks": [
                    {"type": "paragraph",  "page": 1, "text": "..."},
                    {"type": "list",       "page": 1, "ordered": false, "items": ["...", "..."]},
                    {"type": "key_values", "page": 1, "fields": {"Status": "Approved"}},
                    {"type": "table",      "page": 2, "header": [...], "rows": [...], "records": [...]}
                  ]
                }
              ],
              "counts": {"sections": 5, "paragraphs": 4, "lists": 2, "tables": 1, "fields": 4}
            }

        Errors follow the same never-raise contract: {"error": "..."}.
        """
        full = Path(path).resolve()
        if not full.exists():
            return {"error": f"file not found: {path}"}
        if full.stat().st_size > MAX_PDF_BYTES:
            return {"error": f"{path} exceeds the {MAX_PDF_BYTES // 1_000_000} MB limit."}

        # Elements come from the document STORE, not from a second pdfplumber
        # pass of our own. The store already groups words into lines, orders
        # page columns, and detects tables; doing it again here produced a
        # different answer from the one `swarn ask` works off, and a worse one
        # (fused words, merged columns). See swarn/capabilities/doc_structure.
        try:
            from swarn.capabilities.doc_store import get_or_ingest
            from swarn.capabilities.doc_structure import elements_from_stored
        except ImportError as exc:
            return {"error": ("PDF extraction requires 'pip install pdfplumber "
                              f"pydantic pillow' ({exc}).")}

        try:
            stored, _ = get_or_ingest(str(full))
        except Exception as e:
            return {"error": f"reading PDF '{path}': {type(e).__name__}: {e}"}

        elements = elements_from_stored(stored)
        metadata = dict(stored.metadata)
        n_pages = stored.page_count

        if not elements:
            return {"error": (f"no extractable text or tables in {path} — it may be a "
                              "scanned/image-only PDF, in which case run index_image on a "
                              "rasterized page to OCR it instead.")}

        body_size = self._body_font_size(elements)
        heading_levels = self._heading_levels(elements, body_size)
        sections = self._build_sections(elements, body_size, heading_levels)

        fields: dict = {}
        for section in sections:
            for block in section["blocks"]:
                if block["type"] == "key_values":
                    for k, v in block["fields"].items():
                        fields.setdefault(k, v)
                elif block["type"] == "paragraph":
                    for k, v in self._mine_prose_fields(block["text"]).items():
                        fields.setdefault(k, v)

        def _count(kind: str) -> int:
            return sum(1 for s in sections for b in s["blocks"] if b["type"] == kind)

        return {
            "file":     str(full),
            "n_pages":  n_pages,
            "metadata": metadata,
            "title":    self._document_title(elements, metadata),
            "fields":   fields,
            "sections": sections,
            "counts": {
                "sections":   len(sections),
                "paragraphs": _count("paragraph"),
                "lists":      _count("list"),
                "tables":     _count("table"),
                "fields":     len(fields),
            },
        }

    # ---- element extraction -------------------------------------------------
    # REMOVED: _page_elements() used to open the PDF a second time, group its
    # words into lines and find its tables — duplicating, with different
    # results, what the document store already does. Elements now come from
    # swarn/capabilities/doc_structure.elements_from_stored(), so this path and
    # `swarn ask` read one parse and cannot disagree.

    # ---- heading detection --------------------------------------------------

    @staticmethod
    def _body_font_size(elements: list[dict]) -> float:
        """
        Most common font size, weighted by CHARACTER COUNT rather than by
        line. A title page or a run of headings has few characters at a
        large size; body text has many characters at one size. Weighting
        by line would let a heading-heavy first page redefine "body".
        """
        weights: Counter = Counter()
        for e in elements:
            if e["kind"] == "line" and e["size"]:
                weights[e["size"]] += e["n_chars"]
        return weights.most_common(1)[0][0] if weights else 0.0

    @classmethod
    def _is_heading(cls, e: dict, body_size: float) -> bool:
        if e["kind"] != "line":
            return False
        text = e["text"]
        if len(text) > HEADING_MAX_CHARS or _LIST_MARKER_RE.match(text):
            return False
        if body_size and e["size"] >= body_size * HEADING_SIZE_RATIO:
            return True
        # Bold-at-body-size: only when it doesn't read like a sentence and
        # isn't a "Label: value" line, which is a field, not a heading.
        if e["bold"] and not text.endswith((".", ",", ";")) and not cls._kv_match(text):
            return True
        return False

    @classmethod
    def _heading_levels(cls, elements: list[dict], body_size: float) -> dict:
        """
        Map each distinct heading font size to a level: largest size → 1,
        next → 2, and so on. Bold-only headings (no size jump) sort in at
        their own size, which puts them below any larger heading — the
        same nesting a reader infers visually.
        """
        sizes = sorted({e["size"] for e in elements
                        if cls._is_heading(e, body_size)}, reverse=True)
        return {size: i for i, size in enumerate(sizes, start=1)}

    @staticmethod
    def _document_title(elements: list[dict], metadata: dict) -> Optional[str]:
        """
        The PDF's own Title metadata when it's meaningful — many producers
        write a placeholder like "(anonymous)" or the source filename —
        otherwise the largest text on page 1, which is what a reader would
        call the title.
        """
        declared = (metadata.get("Title") or "").strip()
        if declared and not declared.startswith("(") and not declared.lower().endswith(".pdf"):
            return declared
        first_page = [e for e in elements if e["kind"] == "line" and e["page"] == 1]
        if not first_page:
            return None
        biggest = max(first_page, key=lambda e: e["size"])
        return biggest["text"] if biggest["size"] else None

    # ---- section / block assembly -------------------------------------------

    @classmethod
    def _build_sections(cls, elements: list[dict], body_size: float,
                        heading_levels: dict) -> list[dict]:
        """
        Walk the reading-order elements, starting a new section at every
        heading and grouping everything between headings into typed
        blocks. Content that appears before the first heading still gets a
        section, with heading=None — dropping it would lose the front
        matter of any document that opens with prose.
        """
        sections: list[dict] = []
        current = {"heading": None, "level": 0,
                    "page": elements[0]["page"], "blocks": []}
        pending: list[dict] = []   # body lines not yet flushed into blocks

        def flush():
            if pending:
                current["blocks"].extend(cls._lines_to_blocks(pending))
                pending.clear()

        for e in elements:
            if e["kind"] == "table":
                flush()
                block = {"type": "table", "page": e["page"]}
                block.update({k: v for k, v in e["table"].items() if k != "index"})
                current["blocks"].append(block)
                continue

            if cls._is_heading(e, body_size):
                flush()
                if current["blocks"] or current["heading"]:
                    sections.append(current)
                current = {"heading": e["text"],
                            "level": heading_levels.get(e["size"], 1),
                            "page": e["page"], "blocks": []}
            else:
                pending.append(e)

        flush()
        if current["blocks"] or current["heading"]:
            sections.append(current)
        return sections

    @classmethod
    def _lines_to_blocks(cls, lines: list[dict]) -> list[dict]:
        """
        Group a run of body lines into paragraph / list / key_values
        blocks. Each line is classified first, then adjacent lines of the
        same class merge — which is what makes a four-bullet list one
        block with four items rather than four one-line paragraphs.
        """
        blocks: list[dict] = []
        buf: list[dict] = []
        buf_kind: Optional[str] = None

        def flush():
            nonlocal buf, buf_kind
            if buf:
                blocks.append(cls._make_block(buf_kind, buf))
                buf, buf_kind = [], None

        prev: Optional[dict] = None
        for line in lines:
            kind = cls._line_kind(line["text"])
            # A page change always ends a block. A wide vertical gap ends
            # only a PARAGRAPH: fusing two paragraphs into one would lose a
            # real boundary, whereas list items and field rows are routinely
            # laid out with generous spacing between them, and splitting a
            # four-item list into four one-item lists on that basis would
            # destroy the very grouping this is meant to recover.
            broke = prev is not None and (
                line["page"] != prev["page"] or
                (kind == "paragraph" and
                 line["top"] - prev["top"] > prev["height"] * PARA_BREAK_GAP_RATIO))
            if kind != buf_kind or broke:
                flush()
                buf_kind = kind
            buf.append(line)
            prev = line
        flush()
        return [b for b in blocks if b]

    @staticmethod
    def _kv_segments(text: str) -> list[str]:
        """
        Split a stored line into the cells it was assembled from.

        Lines arrive from the document store with " | " between the segments
        of one row, which is real structure and not punctuation: a bill header
        reading

            Bill Month | : JUL-2026 | Power Factor | : 0.99

        holds two independent fields, and reading it as one string finds
        neither. A segment that STARTS with a colon is the value belonging to
        the segment before it — forms routinely set the label at one tab stop
        and the colon-prefixed value at the next — so those are rejoined
        before matching rather than left as a valueless label and a keyless
        value.
        """
        parts = [p.strip() for p in text.split("|")]
        merged: list[str] = []
        for part in parts:
            if not part:
                continue
            if part.startswith(":") and merged:
                merged[-1] = f"{merged[-1]} {part}"
            else:
                merged.append(part)
        return merged or [text.strip()]

    @classmethod
    def _kv_pairs(cls, text: str) -> dict:
        """Every field on one line, in order. See _kv_match for the rules."""
        pairs = {}
        for segment in cls._kv_segments(text):
            match = cls._kv_match_one(segment)
            if match:
                pairs.setdefault(match[0], match[1])
        return pairs

    @classmethod
    def _kv_match(cls, text: str) -> Optional[tuple]:
        """
        The single gate every "is this a field?" decision goes through, so
        line blocks, prose mining, and heading detection can't disagree
        about what counts as one. Returns the first (key, value) or None.
        """
        for segment in cls._kv_segments(text):
            match = cls._kv_match_one(segment)
            if match:
                return match
        return None

    @staticmethod
    def _latin_label(key: str) -> str:
        """
        The Latin-script half of a bilingual label.

        Indian utility, tax and banking documents label every field twice,
        separated by a slash — "देय ितिथ / Due Date", "िबल संखया / Bill No".
        _KEY_OK_RE requires a key to begin with a Latin letter, so the whole
        label fails and the field is lost, even though half of it is exactly
        the key a caller wants.

        Applied only after the full key has already failed validation, so a
        label that was acceptable as written is never rewritten — a document
        with a genuine slash in its label ("Net / Gross") keeps it.
        """
        if "/" not in key:
            return key
        for part in (p.strip() for p in key.split("/")):
            if part and _KEY_OK_RE.match(part):
                return part
        return key

    @classmethod
    def _kv_match_one(cls, text: str) -> Optional[tuple]:
        """The rules themselves, applied to ONE segment."""
        m = _KV_RE.match(text.strip())
        if not m:
            return None
        key, value = m.group(1).strip(), m.group(2).strip()
        # A rejoined "Label | : value" can leave a second colon at the front of
        # the value when the label already carried one of its own.
        value = value.lstrip(":：").strip()
        if not key or not value:
            return None
        if not _KEY_OK_RE.match(key):
            key = cls._latin_label(key)
        words = key.split()
        if not _KEY_OK_RE.match(key) or len(words) > KV_MAX_KEY_WORDS:
            return None
        if any(w.lower() in _NON_LABEL_WORDS for w in words):
            return None      # the "key" is a clause, not a label
        if _SENTENCE_RE.search(value) or len(value) > KV_MAX_VALUE_CHARS:
            return None      # the "value" is prose, not a field
        return key, value

    @classmethod
    def _line_kind(cls, text: str) -> str:
        # List marker is checked first: "1. Promote the standby..." would
        # otherwise be readable as a field keyed "1".
        if _LIST_MARKER_RE.match(text):
            return "list"
        return "key_values" if cls._kv_match(text) else "paragraph"

    @classmethod
    def _make_block(cls, kind: str, lines: list[dict]) -> Optional[dict]:
        page = lines[0]["page"]

        if kind == "list":
            items = [_LIST_MARKER_RE.sub("", ln["text"]).strip() for ln in lines]
            first = lines[0]["text"].lstrip()
            return {
                "type":    "list",
                "page":    page,
                # Ordered vs unordered changes meaning — "step 3" is not
                # interchangeable with "one of these three".
                "ordered": bool(re.match(r"^\(?[\dA-Za-z][.)]", first)),
                "items":   [i for i in items if i],
            }

        if kind == "key_values":
            fields = {}
            for ln in lines:
                # Every pair on the line, not just the first — one row of a
                # bill header routinely carries two unrelated fields.
                for key, value in cls._kv_pairs(ln["text"]).items():
                    fields.setdefault(key, value)
            return {"type": "key_values", "page": page, "fields": fields} if fields else None

        return {"type": "paragraph", "page": page,
                "text": cls._join_wrapped([ln["text"] for ln in lines])}

    @staticmethod
    def _join_wrapped(parts: list[str]) -> str:
        """
        Rejoin lines that a PDF broke for layout. A trailing hyphen means
        a word was split across the line break, so the hyphen goes away
        and no space replaces it; otherwise lines join with a space.
        """
        out = ""
        for part in parts:
            if not out:
                out = part
            elif out.endswith("-") and not out.endswith((" -", "--")):
                out = out[:-1] + part
            else:
                out += " " + part
        return out

    @classmethod
    def _mine_prose_fields(cls, text: str) -> dict:
        """
        Pull "Label: value" pairs out of flowing prose, which is where a
        lot of real document data hides — an invoice header often reads
        "Billed to: ESDS. Invoice date: 2024-09-17. Terms: Net 30." as a
        single sentence-run, not as separate lines.

        Splitting on sentence boundaries first is what makes this safe:
        each fragment is then tested with the SAME single-pair regex used
        for lines, so a fragment with no colon (ordinary prose) yields
        nothing, and a colon deep inside a long clause fails the
        key-length limit rather than producing a junk field.
        """
        fields = {}
        for fragment in re.split(r"(?<=[.!?;])\s+", text):
            pair = cls._kv_match(fragment)
            if pair:
                fields.setdefault(pair[0], pair[1].rstrip(".;"))
        return fields

    # ───────────────────────────────────────────────── images

    def index_image(self, path: str, caption: Optional[str] = None) -> str:
        """
        Index an image two ways, both optional depending on what's
        available:
          1. OCR (pytesseract) — extracts any text visible in the image
             (diagram labels, screenshots, scanned pages). Always
             attempted; pytesseract/tesseract missing is reported, not
             fatal to the whole call if a caption was also given.
          2. CLIP embedding (sentence-transformers' clip-ViT-B-32, if
             installed) — embeds the image itself, so images with no
             text at all are still findable by semantic meaning. Falls
             back to embedding just the caption/OCR text if CLIP isn't
             available, so the image is still indexed, just not by
             visual content.
          3. caption — an optional human-written description, indexed
             as its own chunk (type=image_caption) regardless of
             whether OCR/CLIP succeed, since a caption often captures
             intent OCR/CLIP can't ("architecture diagram showing the
             retry path").
        """
        full = Path(path).resolve()
        if not full.exists():
            return f"Error: file not found: {path}"
        if full.stat().st_size > MAX_IMAGE_BYTES:
            return f"Error: {path} exceeds the {MAX_IMAGE_BYTES // 1_000_000} MB indexing limit."

        engine = get_context_engine()
        err = engine._ensure_ready()
        if err:
            return f"Error: {err}"

        chunks: list[dict] = []
        notes: list[str] = []

        # ── 1. OCR ──────────────────────────────────────────────────
        ocr_text = ""
        try:
            import pytesseract
            from PIL import Image
            ocr_text = pytesseract.image_to_string(Image.open(full)).strip()
            if ocr_text:
                chunks.append(self._make_pdf_chunk(
                    ocr_text, str(full), "image_ocr", f"{full.name}:ocr", 1,
                ))
            else:
                notes.append("OCR found no text in the image (this is normal for photos/diagrams without labels).")
        except ImportError:
            notes.append("OCR skipped: 'pip install pytesseract pillow' and install the tesseract binary to enable it.")
        except Exception as e:
            notes.append(f"OCR failed: {type(e).__name__}: {e}")

        # ── 2. caption ────────────────────────────────────────────────
        if caption:
            chunks.append(self._make_pdf_chunk(
                caption, str(full), "image_caption", f"{full.name}:caption", 1,
            ))

        # ── 3. CLIP embedding of the image itself, if available ──────
        # Tried independently of OCR/caption — even if both of those
        # found nothing, a CLIP embedding can still make a purely visual
        # image (a photo, a chart with no readable labels) searchable
        # by what it shows, not what it says.
        clip_indexed = self._try_index_image_embedding(engine, full)
        if clip_indexed:
            notes.append("Indexed by direct image embedding (CLIP) — searchable by visual content even without text.")
        else:
            notes.append(
                "Direct image-content search not available this session "
                "(requires a CLIP-family sentence-transformers model) — "
                "image is searchable via OCR text/caption only, if any was found."
            )

        if not chunks and not clip_indexed:
            return (
                f"Nothing indexable found for {path}: {' '.join(notes)} "
                f"Consider passing a `caption` describing the image."
            )

        if chunks:
            self._embed_and_upsert(engine, chunks)

        return f"Indexed {path}: {len(chunks)} text-based chunk(s) (OCR/caption).\n" + "\n".join(notes)

    def _try_index_image_embedding(self, engine: ContextEngine, image_path: Path) -> bool:
        """
        Best-effort: embed the image itself with a CLIP-family model and
        upsert directly into the same collection, alongside the
        text-embedding chunks everything else in this codebase uses.
        Returns False (never raises) if no CLIP model is available —
        this is treated as a normal, expected fallback, not an error,
        since the OCR/caption path above already covers the common case.
        """
        try:
            from sentence_transformers import SentenceTransformer
            from PIL import Image
        except ImportError:
            return False

        try:
            # A separate small model instance, NOT engine._embedder —
            # Phase 3's embedder is a text model (all-MiniLM-L6-v2) and
            # cannot embed images. CLIP is loaded lazily and only on the
            # first image-embedding call, same "pay for it only when
            # used" rule Phase 3 already follows for its own model.
            if not hasattr(engine, "_clip_embedder") or engine._clip_embedder is None:
                engine._clip_embedder = SentenceTransformer("clip-ViT-B-32")
            clip = engine._clip_embedder

            image = Image.open(image_path)
            embedding = clip.encode(image, show_progress_bar=False).tolist()

            chunk_id = ContextEngine._make_id({
                "metadata": {"file": str(image_path), "start_line": 1, "end_line": 1, "name": f"{image_path.name}:clip"}
            })
            engine._collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[f"[image content embedding: {image_path.name}]"],
                metadatas=[{
                    "file": str(image_path), "type": "image_embedding",
                    "name": f"{image_path.name}:clip", "start_line": 1, "end_line": 1,
                }],
            )
            return True
        except Exception:
            return False

    # ───────────────────────────────────────────────── audio

    def index_audio(self, path: str, model_size: str = "base") -> str:
        """
        Transcribe an audio file with openai-whisper and index the
        transcript, chunked by a rolling window of whisper's own
        segments (which already carry timestamps) rather than
        re-splitting by line count — segment boundaries are natural
        pause points, so chunking along them keeps each chunk coherent.

        model_size: whisper model size ("tiny", "base", "small",
        "medium", "large"). "base" is a reasonable default — larger
        models are slower but more accurate; this isn't auto-selected
        since the right tradeoff depends on the audio and isn't
        something this function can know.
        """
        try:
            import whisper
        except ImportError:
            return "Error: audio indexing requires 'pip install openai-whisper' (and ffmpeg installed on the system)."

        full = Path(path).resolve()
        if not full.exists():
            return f"Error: file not found: {path}"
        if full.stat().st_size > MAX_AUDIO_BYTES:
            return f"Error: {path} exceeds the {MAX_AUDIO_BYTES // 1_000_000} MB indexing limit."

        engine = get_context_engine()
        err = engine._ensure_ready()
        if err:
            return f"Error: {err}"

        try:
            model = whisper.load_model(model_size)
            result = model.transcribe(str(full))
        except Exception as e:
            return f"Error transcribing '{path}': {type(e).__name__}: {e}"

        segments = result.get("segments", [])
        if not segments:
            return f"No speech detected in {path}."

        chunks = self._chunk_audio_segments(segments, str(full))
        self._embed_and_upsert(engine, chunks)

        duration_s = segments[-1]["end"] if segments else 0
        return (
            f"Indexed {path}: {len(segments)} speech segments "
            f"(~{duration_s / 60:.1f} min) → {len(chunks)} chunks. "
            f"Citations use timestamps (e.g. '12:34') instead of page/line numbers."
        )

    def _chunk_audio_segments(self, segments: list[dict], file_path: str, window_s: float = 60.0) -> list[dict]:
        """
        Group whisper's per-segment transcript into ~window_s-second
        chunks (default 1 minute) — long enough to be a coherent excerpt,
        short enough that a citation timestamp still points somewhere
        useful to skip to.
        """
        chunks = []
        current_text: list[str] = []
        window_start = segments[0]["start"]
        window_end = window_start

        def flush():
            if current_text:
                text = " ".join(current_text).strip()
                if text:
                    chunks.append(self._make_pdf_chunk(
                        text, file_path, "audio_transcript",
                        f"{Path(file_path).name}:{self._format_timestamp(window_start)}",
                        int(window_start),   # reused as a sortable "page" proxy
                    ))

        for seg in segments:
            if seg["start"] - window_start > window_s and current_text:
                flush()
                current_text = []
                window_start = seg["start"]
            current_text.append(seg["text"].strip())
            window_end = seg["end"]
        flush()
        return chunks

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    # ───────────────────────────────────────────────── shared upsert path

    def _embed_and_upsert(self, engine: ContextEngine, chunks: list[dict]) -> None:
        """
        Identical batching/embedding logic to ContextEngine.index_directory
        — duplicated rather than imported as a private method call across
        files, since it's a five-line loop and not worth coupling this
        module to Phase 3's exact internal batch size.
        """
        BATCH = 64
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i + BATCH]
            texts = [c["content"] for c in batch]
            ids = [ContextEngine._make_id(c) for c in batch]
            metas = [c["metadata"] for c in batch]
            embeds = engine._embedder.encode(texts, show_progress_bar=False, batch_size=32).tolist()
            engine._collection.upsert(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)


# ─── singleton, matching the rest of the codebase ──────────────────────────────

_indexer: Optional[MultiModalIndexer] = None


def get_multimodal_indexer() -> MultiModalIndexer:
    global _indexer
    if _indexer is None:
        _indexer = MultiModalIndexer()
    return _indexer
