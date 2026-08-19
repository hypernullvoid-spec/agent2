"""
Phase 3: Project Context Engine (Repo-RAG)

Gives the agent "semantic eyes" over a codebase. Instead of reading every
file blindly, the agent can:
  1. index_directory(path) — walk the tree, chunk every file into meaningful
     pieces, embed each chunk, and persist everything to ChromaDB.
  2. search(query) — find the most relevant code/doc chunks using cosine
     similarity on the embeddings.

This is what separates "file reader" from "codebase-aware agent": the agent
can find the right function in a 500-file repo with one query, then read
just that file with read_file for the full picture.

Stack
─────
  sentence-transformers / all-MiniLM-L6-v2
    ↳ ~90 MB download, CPU-only, no API key, first call triggers the download.
  ChromaDB (PersistentClient)
    ↳ written to <project-root>/.chroma/, no server, no config.

Both are lazy-imported so startup is instant even before the deps are
installed — the error surfaces only when a tool actually tries to use them.

Dependencies (install once):
  pip install sentence-transformers chromadb
"""

import ast
import hashlib
import os
from pathlib import Path
from typing import Optional

CHROMA_DIR  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".chroma")
)
EMBED_MODEL = "all-MiniLM-L6-v2"

# File types we can meaningfully index
INDEXABLE_EXT = {
    ".py", ".md", ".txt", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".html", ".css", ".scss", ".sql", ".sh", ".bash",
    ".rst", ".env.example", ".gitignore", ".dockerfile",
}

# Directories that should never be indexed
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", ".env",
    "node_modules", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".chroma", "runs",
}

MAX_FILE_BYTES = 300_000   # skip files > 300 KB (generated or binary-ish)
CHUNK_LINES    = 60        # target lines per text chunk
OVERLAP_LINES  = 10        # overlap between consecutive text chunks


class ContextEngine:
    """
    Manages a ChromaDB collection that holds embeddings for an indexed codebase.

    index_directory() and search() are the two public methods; tools.py wraps
    them as agent-callable tools. The engine keeps a single ChromaDB collection
    called "codebase" shared across all indexed directories in a session
    (you can call index_directory multiple times to add more content).
    """

    def __init__(self):
        self._ready      = False
        self._collection = None
        self._embedder   = None

    # ─────────────────────────────────────────────────── lazy init

    def _ensure_ready(self) -> Optional[str]:
        """
        Load sentence-transformers + ChromaDB on first use.
        Returns None on success, or an error string on failure.

        Catches everything, not just ImportError: the original version
        of this method only handled a missing package, but
        SentenceTransformer(EMBED_MODEL) also makes a network call on
        first use (to download the model from huggingface.co) — and a
        network failure there (no internet, a firewalled sandbox, a
        registry outage) raised an unhandled exception straight through
        every caller, breaking this method's own documented "returns an
        error string on failure" contract. This surfaced concretely
        while testing Phase 13's PDF/image indexing tools in a sandboxed
        environment where huggingface.co wasn't reachable — the bare
        exception propagated all the way up through index_pdf/
        index_image with a multi-page traceback instead of the clear,
        actionable error string every other tool in this codebase
        returns for a failure. Catching broadly here (not just
        ImportError) is what makes that the same "errors are strings"
        experience as everywhere else.
        """
        if self._ready:
            return None

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            return (
                f"Missing dependency: {e}\n"
                "Run: pip install sentence-transformers chromadb"
            )

        try:
            print("[context] Loading embedding model "
                  "(~90 MB download on first run)…")
            self._embedder = SentenceTransformer(EMBED_MODEL)
            print("[context] ✓ Embedding model ready.")

            os.makedirs(CHROMA_DIR, exist_ok=True)
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            self._collection = client.get_or_create_collection(
                name     = "codebase",
                metadata = {"hnsw:space": "cosine"},
            )
        except Exception as e:
            return (
                f"Failed to initialize the embedding model or vector store: "
                f"{type(e).__name__}: {e}\n"
                "If this is the first run, it needs to download "
                f"'{EMBED_MODEL}' from huggingface.co — check network "
                "access if you're in a restricted/offline environment."
            )

        self._ready = True
        return None

    # ─────────────────────────────────────────────────── indexing

    def index_directory(self, directory: str) -> str:
        """
        Walk `directory`, chunk every indexable file, embed, and upsert
        to ChromaDB. Calling again on the same directory safely updates
        (upsert = insert-or-replace on the chunk ID).
        """
        err = self._ensure_ready()
        if err:
            return f"Error: {err}"

        root = Path(directory).resolve()
        if not root.exists():
            return f"Directory not found: {directory}"

        chunks: list[dict] = []
        skipped = 0

        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            if any(d in path.parts for d in SKIP_DIRS):
                continue
            if path.suffix.lower() not in INDEXABLE_EXT:
                # also accept extensionless files like Dockerfile
                if path.suffix != "" or path.name.startswith("."):
                    skipped += 1
                    continue
            if path.stat().st_size > MAX_FILE_BYTES:
                skipped += 1
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                skipped += 1
                continue

            rel = str(path.relative_to(root))
            new_chunks = (
                self._chunk_python(content, rel)
                if path.suffix == ".py"
                else self._chunk_text(content, rel)
            )
            chunks.extend(new_chunks)

        if not chunks:
            return f"No indexable content found in {directory}"

        # Embed + upsert in batches of 64 to avoid OOM on large repos
        BATCH = 64
        total = 0
        for i in range(0, len(chunks), BATCH):
            batch  = chunks[i : i + BATCH]
            texts  = [c["content"]  for c in batch]
            ids    = [self._make_id(c) for c in batch]
            metas  = [c["metadata"] for c in batch]
            embeds = self._embedder.encode(
                texts, show_progress_bar=False, batch_size=32
            ).tolist()
            self._collection.upsert(
                ids=ids, embeddings=embeds,
                documents=texts, metadatas=metas,
            )
            total += len(batch)

        n_files = len({c["metadata"]["file"] for c in chunks})
        return (
            f"✓ Indexed {n_files} files → {total} chunks "
            f"({skipped} files skipped — non-text / oversized)\n"
            f"Index persisted at: {CHROMA_DIR}"
        )

    # ─────────────────────────────────────────────────── search

    def search(self, query: str, n_results: int = 6) -> str:
        """
        Return the top-k most semantically similar chunks to `query`.
        Results include file path, line range, chunk type, and a 500-char
        preview of the chunk content.
        """
        err = self._ensure_ready()
        if err:
            return f"Error: {err}"

        count = self._collection.count()
        if count == 0:
            return (
                "The index is empty — call index_project first to "
                "index a directory before searching."
            )

        q_embed = self._embedder.encode(
            [query], show_progress_bar=False
        ).tolist()

        res = self._collection.query(
            query_embeddings = q_embed,
            n_results        = min(n_results, count),
            include          = ["documents", "metadatas", "distances"],
        )

        docs   = res["documents"][0]
        metas  = res["metadatas"][0]
        dists  = res["distances"][0]

        if not docs:
            return "No results found."

        lines = [f"Top {len(docs)} results for: \"{query}\"\n"]
        for doc, meta, dist in zip(docs, metas, dists):
            score   = round(1.0 - dist, 3)   # cosine similarity (higher = better)
            f_path  = meta.get("file",       "?")
            s_line  = meta.get("start_line", "?")
            e_line  = meta.get("end_line",   "?")
            kind    = meta.get("type",       "")
            name    = meta.get("name",       "")

            header  = f"### {f_path}  lines {s_line}–{e_line}"
            if name and kind:
                header += f"  [{kind}: {name}]"
            header += f"  similarity={score}"

            preview = doc[:500] + ("…" if len(doc) > 500 else "")
            lines  += [header, "```", preview, "```", ""]

        return "\n".join(lines)

    # ─────────────────────────────────────────────────── chunking

    def _chunk_python(self, content: str, file_path: str) -> list[dict]:
        """
        Chunk a Python file into semantically meaningful units using the AST.

        Strategy:
          1. A "module header" chunk containing all import statements and
             top-level assignments (up to 40 lines) — gives context about
             what the file uses and exports.
          2. One chunk per top-level / class-level function or class
             definition (FunctionDef, AsyncFunctionDef, ClassDef).

        Falls back to sliding-window text chunking on SyntaxError.
        """
        lines  = content.splitlines()
        chunks = []

        # Module header: imports + top-level assignments
        header_text = "\n".join(lines[:40]).strip()
        if header_text:
            chunks.append(self._make_chunk(
                f"# {file_path}\n{header_text}",
                file_path, "module_header",
                f"{file_path}:header", 1, min(40, len(lines)),
            ))

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._chunk_text(content, file_path)

        seen_spans = set()
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            start = node.lineno - 1                          # 0-indexed
            end   = getattr(node, "end_lineno", start + 1)  # 1-indexed inclusive
            span  = (start, end)
            if span in seen_spans:
                continue
            seen_spans.add(span)

            text = "\n".join(lines[start:end]).strip()
            if len(text) < 20:
                continue

            chunks.append(self._make_chunk(
                text, file_path,
                type(node).__name__, node.name,
                start + 1, end,
            ))

        return chunks if chunks else self._chunk_text(content, file_path)

    def _chunk_text(self, content: str, file_path: str) -> list[dict]:
        """
        Chunk any text file with a fixed-size sliding window.
        Window = CHUNK_LINES lines; overlap = OVERLAP_LINES lines.
        """
        lines  = content.splitlines()
        chunks = []
        i, idx = 0, 0

        while i < len(lines):
            end  = min(i + CHUNK_LINES, len(lines))
            text = "\n".join(lines[i:end]).strip()
            if text:
                chunks.append(self._make_chunk(
                    text, file_path,
                    "text_chunk", f"{file_path}:{idx}",
                    i + 1, end,
                ))
            i  += CHUNK_LINES - OVERLAP_LINES
            idx += 1

        return chunks

    # ─────────────────────────────────────────────────── helpers

    @staticmethod
    def _make_chunk(
        content: str, file: str, kind: str,
        name: str, start: int, end: int,
    ) -> dict:
        return {
            "content":  content,
            "metadata": {
                "file":       file,
                "type":       kind,
                "name":       name,
                "start_line": start,
                "end_line":   end,
            },
        }

    @staticmethod
    def _make_id(chunk: dict) -> str:
        """
        Stable, collision-resistant chunk ID used for ChromaDB upsert.
        Built from (file, start, end, name) so re-indexing a changed file
        updates the right record rather than creating a duplicate.
        """
        m   = chunk["metadata"]
        raw = f"{m['file']}:{m['start_line']}:{m['end_line']}:{m['name']}"
        return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Module-level singleton — one context engine per process / session.
# ---------------------------------------------------------------------------

_engine: Optional[ContextEngine] = None


def get_context_engine() -> ContextEngine:
    global _engine
    if _engine is None:
        _engine = ContextEngine()
    return _engine
