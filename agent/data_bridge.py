"""
Bridge between the in-memory dataset registry and the sandbox process.

The problem this solves
───────────────────────
Datasets loaded by load_csv live in THIS process (data_pipeline.datasets).
run_python executes in a SEPARATE process — a fresh subprocess per call, or a
Docker container. Those two never shared memory, so code in the sandbox had no
way to reach a loaded dataset, and the only way a model could get a DataFrame
was pd.read_csv() — the exact stale read that run_python then refused. That
left the model with a correct prohibition and no correct alternative, so it
retried the same forbidden call until its attempts ran out.

How it works
────────────
Before the code runs, the datasets it actually MENTIONS are written to the
workspace and bound to plain variables by a one-line bootstrap import.
Referencing `orders` in the code is enough; nothing else is materialised, so a
session holding twenty datasets pays only for the two a given query touches.

Nothing is copied back into the model's context — only what the code prints.
That is the point of running analysis as code: a 2-million-row frame stays in
the sandbox and the model sees the six-line groupby result.

Transport format
────────────────
CSV plus a JSON sidecar carrying the dtypes, rather than parquet or pickle.
The sandbox image ships neither pyarrow nor fastparquet, so parquet cannot be
read there at all; pickle would work only while the image's pandas happens to
match the host's, and the image is rebuilt independently of the host venv — a
silent version skew is exactly the kind of failure that would surface as
corrupted numbers rather than an error. CSV survives any version pair, and the
sidecar restores the dtypes that CSV alone would flatten.

Paths are RELATIVE. The subprocess backend runs with cwd=workspace and the
Docker backend with workdir=/workspace, so the same relative path resolves in
both — an absolute host path would not exist inside the container.

Datasets are cached by object identity, so repeated calls against an unchanged
frame re-use the files already on disk. The cache holds a reference to the
frame, which is what makes identity safe to compare — a live reference cannot
be garbage collected, so its id() cannot be recycled onto a different object.
"""

import ast
import json
import os
import re
import shutil
from typing import Optional

SCRATCH_DIRNAME = ".swarn_datasets"
BOOTSTRAP_NAME = "_swarn_bootstrap.py"
OUT_DIRNAME = "_published"

# name → (frame reference, relative base path). The reference is deliberate.
_materialised: dict = {}


def _pipeline():
    from agent.ml.data_pipeline import get_data_pipeline
    return get_data_pipeline()


def _workspace() -> str:
    from agent.runtime.execution import WORKSPACE_DIR
    return WORKSPACE_DIR


def _abs(rel: str) -> str:
    return os.path.join(_workspace(), rel)


def _scratch_dir() -> str:
    os.makedirs(_abs(SCRATCH_DIRNAME), exist_ok=True)
    return SCRATCH_DIRNAME


def _out_dir() -> str:
    rel = os.path.join(SCRATCH_DIRNAME, OUT_DIRNAME)
    os.makedirs(_abs(rel), exist_ok=True)
    return rel


# ── which datasets does this code actually need? ──────────────────────────────

def referenced_datasets(code: str, available) -> list:
    """Registry names the code mentions — as a bare variable or in load_dataset().

    Parsed with ast so a name inside a comment does not count. Code that does not
    parse (a syntax error we are about to report anyway) falls back to a word
    scan: over-including a dataset costs one file write, missing one costs a
    failed call.
    """
    available = list(available)
    if not code or not available:
        return []
    found: set = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        words = set(re.findall(r"\b\w+\b", code))
        return [n for n in available if n in words]
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)          # load_dataset('sales-2024')
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
    return [n for n in available if n in found]


def _is_identifier(name: str) -> bool:
    return name.isidentifier() and not name.startswith("_")


# ── materialise ───────────────────────────────────────────────────────────────

def _dump(df, rel_base: str) -> None:
    """Write one frame as CSV + a dtype sidecar, both under the workspace."""
    import pandas as pd
    index = not (isinstance(df.index, pd.RangeIndex)
                 and df.index.start == 0 and df.index.step == 1)
    meta = {
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "datetime_cols": [str(c) for c, t in df.dtypes.items()
                          if pd.api.types.is_datetime64_any_dtype(t)],
        "index": index,
    }
    df.to_csv(_abs(rel_base + ".csv"), index=index)
    with open(_abs(rel_base + ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _materialise(name: str, df) -> Optional[str]:
    """Serialise one frame, re-using the last files when the frame is unchanged."""
    cached = _materialised.get(name)
    if cached is not None and cached[0] is df and os.path.exists(_abs(cached[1] + ".csv")):
        return cached[1]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    rel_base = os.path.join(_scratch_dir(), safe)
    try:
        _dump(df, rel_base)
    except Exception:  # noqa: BLE001 — one awkward frame must not kill the call
        return None
    _materialised[name] = (df, rel_base)
    return rel_base


# The reader is shared by the bootstrap and by collect_published, so a frame
# makes the same round trip in both directions.
_READER_SRC = '''
def _swarn_read(base):
    with open(base + '.meta.json', encoding='utf-8') as f:
        meta = json.load(f)
    dt_cols = [c for c in meta.get('datetime_cols') or []]
    df = pd.read_csv(base + '.csv',
                     index_col=0 if meta.get('index') else None)
    for col in dt_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    for col, want in (meta.get('dtypes') or {}).items():
        if col not in df.columns or col in dt_cols:
            continue
        if str(df[col].dtype) == want:
            continue
        try:
            df[col] = df[col].astype(want)
        except Exception:
            pass          # a dtype CSV cannot round-trip is not worth failing on
    return df
'''


def build_bootstrap(names: list) -> tuple:
    """Write the module that binds the datasets, and return (import line, bound).

    Kept to a SINGLE import line so a traceback's line numbers still line up
    with the code the model wrote.
    """
    pipe = _pipeline()
    out_rel = _out_dir()
    bound = []
    lines = [
        "# generated by agent/data_bridge.py — binds loaded datasets for the sandbox",
        "import json",
        "import os",
        "import pandas as pd",
        "import numpy as np",
        "",
        f"_SWARN_OUT = {out_rel!r}",
        "_SWARN_PATHS = {}",
        _READER_SRC,
        "",
    ]
    for name in names:
        df = pipe.datasets.get(name)
        if df is None:
            continue
        rel_base = _materialise(name, df)
        if rel_base is None:
            continue
        lines.append(f"_SWARN_PATHS[{name!r}] = {rel_base!r}")
        if _is_identifier(name):
            lines.append(f"{name} = _swarn_read({rel_base!r})")
            bound.append(name)
    lines += [
        "",
        "",
        "def load_dataset(name):",
        "    \"\"\"Fetch a loaded dataset by name — for names that are not valid",
        "    Python identifiers ('sales-2024').\"\"\"",
        "    if name not in _SWARN_PATHS:",
        "        raise KeyError(",
        "            f\"dataset {name!r} was not made available to this call. \"",
        "            f\"Available here: {sorted(_SWARN_PATHS)}. Mention it by name in \"",
        "            f\"your code, or load it first with load_csv.\")",
        "    return _swarn_read(_SWARN_PATHS[name])",
        "",
        "",
        "def publish_dataset(name, df):",
        "    \"\"\"Hand a derived frame BACK to the agent's registry, so later tool",
        "    calls (analyze_dataset, rank_by, write_report) can use it.\"\"\"",
        "    os.makedirs(_SWARN_OUT, exist_ok=True)",
        "    safe = ''.join(c if (c.isalnum() or c in '_.-') else '_' for c in str(name))",
        "    base = os.path.join(_SWARN_OUT, safe)",
        "    index = not (isinstance(df.index, pd.RangeIndex)",
        "                 and df.index.start == 0 and df.index.step == 1)",
        "    meta = {'dtypes': {str(c): str(t) for c, t in df.dtypes.items()},",
        "            'datetime_cols': [str(c) for c, t in df.dtypes.items()",
        "                              if pd.api.types.is_datetime64_any_dtype(t)],",
        "            'index': index}",
        "    df.to_csv(base + '.csv', index=index)",
        "    with open(base + '.meta.json', 'w', encoding='utf-8') as f:",
        "        json.dump(meta, f)",
        "    print(f\"[published dataset {name!r}: {df.shape[0]:,} rows x {df.shape[1]} cols]\")",
        "    return name",
        "",
    ]
    with open(_abs(BOOTSTRAP_NAME), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return f"from {BOOTSTRAP_NAME[:-3]} import *  # noqa\n", bound


# ── collect anything the code published back ──────────────────────────────────

def collect_published() -> list:
    """Register frames the sandbox published, and report what came back."""
    import pandas as pd
    out_rel = _out_dir()
    out_abs = _abs(out_rel)
    registered = []
    for entry in sorted(os.listdir(out_abs)):
        if not entry.endswith(".csv"):
            continue
        stem = entry[:-4]
        base = os.path.join(out_abs, stem)
        try:
            with open(base + ".meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            df = pd.read_csv(base + ".csv", index_col=0 if meta.get("index") else None)
            for col in meta.get("datetime_cols") or []:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            for col, want in (meta.get("dtypes") or {}).items():
                if col in df.columns and str(df[col].dtype) != want:
                    try:
                        df[col] = df[col].astype(want)
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001 — a half-written frame is not fatal
            continue
        finally:
            for suffix in (".csv", ".meta.json"):
                try:
                    os.remove(base + suffix)
                except OSError:
                    pass
        pipe = _pipeline()
        pipe.datasets[stem] = df
        pipe.sources[stem] = "sandbox:run_python"
        _materialised.pop(stem, None)          # force a re-write next call
        registered.append((stem, df.shape))
    return registered


def cleanup() -> None:
    """Remove the bootstrap module once the call is over."""
    try:
        os.remove(_abs(BOOTSTRAP_NAME))
    except OSError:
        pass


def reset() -> None:
    """Drop every materialised copy — used by tests and by 'clear'."""
    _materialised.clear()
    shutil.rmtree(_abs(SCRATCH_DIRNAME), ignore_errors=True)


# ── traceback line numbers ────────────────────────────────────────────────────

_LINE_RE = re.compile(r'(File "[^"]*_exec_[^"]*\.py", line )(\d+)')


def _shift_tracebacks(text: str, offset: int) -> str:
    """The bootstrap import occupies line 1, so every reported line is one high."""
    def fix(m):
        return f"{m.group(1)}{max(1, int(m.group(2)) - offset)}"
    return _LINE_RE.sub(fix, text or "")


# ── the entry point tools.run_python uses ─────────────────────────────────────

def run_with_datasets(code: str, timeout: Optional[int] = None) -> str:
    """Execute `code` in the sandbox with the datasets it mentions bound to
    variables of the same name.

    Returns the sandbox's stdout/stderr, prefixed with a short note saying which
    datasets were bound. The note matters: without it, a model that referenced a
    name we did not recognise cannot tell why the variable was undefined.
    """
    from agent.runtime.sandbox import get_sandbox
    pipe = _pipeline()
    names = referenced_datasets(code, pipe.datasets.keys())
    if not names:
        return get_sandbox().exec_python(code, timeout=timeout)

    header, bound = build_bootstrap(names)
    try:
        raw = get_sandbox().exec_python(header + code, timeout=timeout)
    finally:
        cleanup()
    output = _shift_tracebacks(raw, offset=header.count("\n"))

    published = collect_published()
    notes = []
    if bound:
        shapes = ", ".join(
            f"{n} ({pipe.datasets[n].shape[0]:,}x{pipe.datasets[n].shape[1]})"
            for n in bound if n in pipe.datasets)
        notes.append(f"[datasets available in this call: {shapes}]")
    unbound = [n for n in names if n not in bound]
    if unbound:
        notes.append(f"[reachable only via load_dataset('<name>'): {unbound}]")
    if published:
        back = ", ".join(f"'{n}' ({r:,}x{c})" for n, (r, c) in published)
        notes.append(f"[registered back into the dataset registry: {back} — "
                     f"usable now by analyze_dataset / rank_by / write_report]")
    return "\n".join(notes + [output]) if notes else output
