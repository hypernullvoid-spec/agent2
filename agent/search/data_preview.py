"""
Data preview — a compact textual snapshot of the task's data directory,
injected into every draft prompt so the model writes code against the
*actual* files, columns, and dtypes instead of guessing.
"""

from __future__ import annotations

import json
import os

MAX_FILES_LISTED = 50
MAX_PREVIEW_CHARS = 4000


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _preview_csv(path: str) -> str:
    try:
        import pandas as pd
        df = pd.read_csv(path, nrows=100)
        buf = [f"shape (first 100 rows read): {df.shape[0]}x{df.shape[1]}",
               "columns/dtypes:"]
        for col, dt in list(df.dtypes.items())[:40]:
            buf.append(f"  {col}: {dt}")
        buf.append("head(3):")
        buf.append(df.head(3).to_string(max_cols=12, max_colwidth=24))
        return "\n".join(buf)
    except Exception as e:  # noqa: BLE001
        return f"(could not preview: {e})"


def _preview_json(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            snippet = f.read(2000)
        data = json.loads(snippet) if len(snippet) < 2000 else None
        if isinstance(data, dict):
            return "top-level keys: " + ", ".join(list(data)[:20])
        return "starts with: " + snippet[:300]
    except Exception:  # noqa: BLE001
        return "(binary or unparseable)"


def generate(data_dir: str) -> str:
    """File tree + per-file previews, truncated to a sane prompt size."""
    if not data_dir or not os.path.isdir(data_dir):
        return "(no data directory provided)"

    entries: list[tuple[str, int]] = []
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            p = os.path.join(root, fname)
            try:
                entries.append((os.path.relpath(p, data_dir), os.path.getsize(p)))
            except OSError:
                continue

    entries.sort()
    lines = [f"Data directory contains {len(entries)} file(s):"]
    for rel, size in entries[:MAX_FILES_LISTED]:
        lines.append(f"  {rel} ({_human_size(size)})")
    if len(entries) > MAX_FILES_LISTED:
        lines.append(f"  … and {len(entries) - MAX_FILES_LISTED} more")

    # detailed previews for the first few tabular/structured files
    shown = 0
    for rel, _size in entries:
        if shown >= 4:
            break
        ext = os.path.splitext(rel)[1].lower()
        p = os.path.join(data_dir, rel)
        if ext == ".csv":
            lines += [f"\n--- {rel} ---", _preview_csv(p)]
            shown += 1
        elif ext == ".json":
            lines += [f"\n--- {rel} ---", _preview_json(p)]
            shown += 1
        elif ext in (".txt", ".md"):
            with open(p, encoding="utf-8", errors="replace") as f:
                lines += [f"\n--- {rel} (first 500 chars) ---", f.read(500)]
            shown += 1

    text = "\n".join(lines)
    return text[:MAX_PREVIEW_CHARS] + ("\n… (truncated)" if len(text) > MAX_PREVIEW_CHARS else "")
