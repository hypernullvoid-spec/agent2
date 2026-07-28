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
from pathlib import Path
from typing import Optional

from agent.context_engine import get_context_engine, ContextEngine

MAX_PDF_BYTES   = 50_000_000   # 50 MB — large scanned PDFs can run OCR-equivalent extraction for a long time
MAX_IMAGE_BYTES = 20_000_000
MAX_AUDIO_BYTES = 200_000_000  # whisper transcription is slow; cap to keep a single call bounded

PDF_TEXT_CHUNK_LINES = 40   # smaller than Phase 3's CHUNK_LINES=60 — PDF text reflows oddly per-page


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
