"""
Phase 6: Data Ingestion & Validation

Gives the agent connectors for the common tabular data sources (CSV,
Excel, Parquet, SQL, cloud storage) and an automated validation layer
that profiles a loaded dataset and flags schema/null/outlier problems
*before* Phase 7's feature engineering or Phase 8's training ever sees it.

Design mirrors context_engine.py / sandbox.py: a singleton manager class
(`DataPipeline`) holds the in-memory dataset registry for the current
process, and tools.py exposes thin @tool wrappers around its methods.

Storage model
─────────────
Datasets are loaded into a small in-memory registry keyed by a name the
agent chooses (e.g. "train", "raw_sales"). Nothing is silently persisted
to disk — if the agent wants a cleaned copy saved, it calls
save_dataset() explicitly. This keeps the registry the single source of
truth within a session and avoids stale CSVs littering the workspace.

Validation philosophy
──────────────────────
validate_dataset() never raises and never blocks — it always returns a
structured report (schema, nulls, dtypes, outliers, duplicates) as text
the agent can read and act on. This is the same "errors as strings"
contract as tools.py, extended into "diagnostics as strings."

Phase boundary
───────────────
This module stops at "is the data sound and loaded into memory." Phase 7
(feature_engineering.py) consumes DataPipeline's registry to transform
columns; Phase 8 (model_training.py) consumes Phase 7's output to train.
None of these phases need to change agent_loop.py — same pattern as
Phase 2/3's sandbox.py and context_engine.py.
"""

import io
import os
from typing import Optional

import pandas as pd

WORKSPACE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "workspace")
)

MAX_PREVIEW_ROWS = 10
OUTLIER_Z_THRESHOLD = 3.0


def _safe_path(path: str) -> str:
    """Same path guard as tools.py — datasets are loaded from inside the workspace."""
    full = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not (full == WORKSPACE_DIR or full.startswith(WORKSPACE_DIR + os.sep)):
        raise ValueError(f"Path '{path}' escapes the workspace directory")
    return full


class DataPipeline:
    """
    Holds the in-memory dataset registry for the current process and
    implements every Phase 6 connector + the validation report.

    One instance per process (see get_data_pipeline() below), matching
    the Sandbox/ContextEngine singleton pattern already used in this
    codebase.
    """

    def __init__(self):
        self.datasets: dict[str, pd.DataFrame] = {}

    # ───────────────────────────────────────────────── ingestion connectors

    def load_csv(self, path: str, name: str, **read_kwargs) -> str:
        full = _safe_path(path)
        if not os.path.exists(full):
            return f"Error: file not found: {path}"
        try:
            df = pd.read_csv(full, **read_kwargs)
        except Exception as e:
            return f"Error reading CSV '{path}': {type(e).__name__}: {e}"
        return self._register(name, df, source=f"csv:{path}")

    def load_excel(self, path: str, name: str, sheet_name=0, **read_kwargs) -> str:
        full = _safe_path(path)
        if not os.path.exists(full):
            return f"Error: file not found: {path}"
        try:
            df = pd.read_excel(full, sheet_name=sheet_name, **read_kwargs)
        except Exception as e:
            return f"Error reading Excel '{path}': {type(e).__name__}: {e}"
        if isinstance(df, dict):
            sheets = ", ".join(df.keys())
            return (
                f"'{path}' has multiple sheets: {sheets}. "
                "Pass sheet_name explicitly to load one."
            )
        return self._register(name, df, source=f"excel:{path}")

    def load_parquet(self, path: str, name: str) -> str:
        full = _safe_path(path)
        if not os.path.exists(full):
            return f"Error: file not found: {path}"
        try:
            df = pd.read_parquet(full)
        except Exception as e:
            return f"Error reading Parquet '{path}': {type(e).__name__}: {e}"
        return self._register(name, df, source=f"parquet:{path}")

    def load_sql(self, connection_string: str, query: str, name: str) -> str:
        try:
            from sqlalchemy import create_engine
        except ImportError:
            return "Error: SQL support requires 'pip install sqlalchemy'."
        try:
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine)
        except Exception as e:
            return f"Error querying database: {type(e).__name__}: {e}"
        return self._register(name, df, source="sql")

    def load_cloud_storage(self, uri: str, name: str) -> str:
        """
        Load a CSV/Parquet object from S3 (s3://) or GCS (gs://) directly
        into the registry. Requires boto3 (S3) or gcsfs (GCS) and valid
        credentials in the environment — this function does not manage
        credentials itself.
        """
        try:
            if uri.startswith("s3://"):
                if uri.endswith(".parquet"):
                    df = pd.read_parquet(uri)
                else:
                    df = pd.read_csv(uri)
            elif uri.startswith("gs://"):
                if uri.endswith(".parquet"):
                    df = pd.read_parquet(uri)
                else:
                    df = pd.read_csv(uri)
            else:
                return "Error: uri must start with 's3://' or 'gs://'."
        except Exception as e:
            return (
                f"Error reading '{uri}': {type(e).__name__}: {e}\n"
                "Check credentials (AWS env vars / GOOGLE_APPLICATION_CREDENTIALS) "
                "and that boto3/gcsfs is installed."
            )
        return self._register(name, df, source=f"cloud:{uri}")

    # ───────────────────────────────────────────────── validation

    def validate_dataset(self, name: str) -> str:
        """
        Profile a loaded dataset and return a structured diagnostic report:
        shape, dtypes, null counts, duplicate rows, and per-numeric-column
        outliers (|z-score| > 3). Tries pandera first for schema-level
        checks (inferred-then-validated) and falls back to a hand-rolled
        profile if pandera isn't installed.
        """
        df = self.datasets.get(name)
        if df is None:
            return self._unknown_dataset_error(name)

        lines = [f"Validation report for '{name}'  (shape: {df.shape[0]} rows × {df.shape[1]} cols)"]

        # ── dtypes ──────────────────────────────────────────────
        lines.append("\ndtypes:")
        for col, dt in df.dtypes.items():
            lines.append(f"  {col}: {dt}")

        # ── nulls ───────────────────────────────────────────────
        null_counts = df.isnull().sum()
        nulls_present = null_counts[null_counts > 0]
        if len(nulls_present):
            lines.append("\nnulls:")
            for col, n in nulls_present.items():
                pct = 100 * n / len(df)
                lines.append(f"  {col}: {n} ({pct:.1f}%)")
        else:
            lines.append("\nnulls: none")

        # ── duplicates ──────────────────────────────────────────
        dup_count = int(df.duplicated().sum())
        lines.append(f"\nduplicate rows: {dup_count}")

        # ── outliers (numeric columns only, z-score method) ─────
        numeric_cols = df.select_dtypes(include="number").columns
        outlier_report = []
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 8 or series.std(ddof=0) == 0:
                continue
            z = (series - series.mean()) / series.std(ddof=0)
            n_outliers = int((z.abs() > OUTLIER_Z_THRESHOLD).sum())
            if n_outliers:
                outlier_report.append(f"  {col}: {n_outliers} values with |z| > {OUTLIER_Z_THRESHOLD}")
        lines.append("\noutliers (z-score method, numeric columns):")
        lines.extend(outlier_report if outlier_report else ["  none detected"])

        # ── pandera schema check, if available ───────────────────
        try:
            import pandera as pa
            schema = pa.infer_schema(df)
            try:
                schema.validate(df, lazy=True)
                lines.append("\npandera inferred-schema check: passed")
            except pa.errors.SchemaErrors as se:
                lines.append("\npandera inferred-schema check: violations found")
                lines.append(str(se.failure_cases.head(20)))
        except ImportError:
            lines.append(
                "\n(pandera not installed — install_package('pandera') for "
                "stricter schema validation; profile above used pandas only)"
            )

        lines.append(
            f"\nNext step suggestion: if nulls/outliers/duplicates look "
            f"problematic, handle them before calling profile_dataset/engineer_features "
            f"on '{name}'."
        )
        return "\n".join(lines)

    # ───────────────────────────────────────────────── inspection helpers

    def preview_dataset(self, name: str, n: int = MAX_PREVIEW_ROWS) -> str:
        df = self.datasets.get(name)
        if df is None:
            return self._unknown_dataset_error(name)
        buf = io.StringIO()
        df.head(n).to_string(buf)
        return (
            f"'{name}'  shape={df.shape[0]}×{df.shape[1]}\n\n"
            f"{buf.getvalue()}"
        )

    def list_datasets(self) -> str:
        if not self.datasets:
            return "No datasets loaded. Use load_csv / load_excel / load_parquet / load_sql first."
        lines = ["Loaded datasets:"]
        for name, df in self.datasets.items():
            lines.append(f"  {name}: {df.shape[0]} rows × {df.shape[1]} cols")
        return "\n".join(lines)

    def save_dataset(self, name: str, path: str) -> str:
        df = self.datasets.get(name)
        if df is None:
            return self._unknown_dataset_error(name)
        full = _safe_path(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        try:
            if path.endswith(".parquet"):
                df.to_parquet(full, index=False)
            else:
                df.to_csv(full, index=False)
        except Exception as e:
            return f"Error saving dataset: {type(e).__name__}: {e}"
        return f"Saved '{name}' ({df.shape[0]} rows) to {path}"

    # ───────────────────────────────────────────────── internals

    def _register(self, name: str, df: pd.DataFrame, source: str) -> str:
        self.datasets[name] = df
        return (
            f"Loaded '{name}' from {source}  "
            f"({df.shape[0]} rows × {df.shape[1]} cols)\n"
            f"Columns: {', '.join(str(c) for c in df.columns)}\n"
            f"Run validate_dataset('{name}') before using it downstream."
        )

    def _unknown_dataset_error(self, name: str) -> str:
        known = ", ".join(self.datasets.keys()) or "(none loaded)"
        return f"Error: no dataset named '{name}' is loaded. Loaded datasets: {known}"


# ─── singleton, matching sandbox.py / context_engine.py ───────────────────────

_pipeline: Optional[DataPipeline] = None


def get_data_pipeline() -> DataPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DataPipeline()
    return _pipeline
