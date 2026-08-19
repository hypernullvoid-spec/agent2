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

from agent.paths import WORKSPACE_DIR, safe_path as _safe_path

MAX_PREVIEW_ROWS = 10
OUTLIER_Z_THRESHOLD = 3.0

# Two category columns overlapping is only a defect for one SHAPE of overlap.
# Tunable because "small minority" is a judgement call that differs by domain.
CATEGORY_SAME_DOMAIN_JACCARD = float(os.environ.get("SWARN_CATEGORY_SAME_DOMAIN", "0.6"))
CATEGORY_LEAK_MAX_SHARE = float(os.environ.get("SWARN_CATEGORY_LEAK_SHARE", "0.5"))


# _safe_path lives in agent/paths.py — imported above under its original name.


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
        self.sources: dict[str, str] = {}      # name → where it was loaded from
        self.saved: dict[str, tuple] = {}      # name → (abs path, shape when written)
        self._cleaner_cache: dict[str, object] = {}

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

    @staticmethod
    def _overlapping_categories(df, max_levels: int = 60) -> list:
        """Two category columns sharing values usually means one leaked into the other.

        A real products file had 'Raksha Bandhan' — an OCCASION — sitting in the
        Category column alongside Sweets and Cake. Every check here passed it: the
        value is a non-null string, the schema infers cleanly, and no z-score
        applies to text. It was then reported as a legitimate 8.4%-of-revenue
        product category. Columns whose vocabularies overlap are worth a sentence,
        because nothing else in this report can see it.
        """
        candidates = []
        for col in df.columns:
            series = df[col]
            if not (series.dtype == object or str(series.dtype) in ("string", "str", "category")):
                continue
            # Single-character values count: 'M'/'F', grade bands and currency
            # codes are ordinary categoricals, and excluding them meant whole
            # columns were never checked at all.
            values = {str(v).strip() for v in series.dropna().unique()}
            values = {v for v in values if v}
            if 1 < len(values) <= max_levels:
                candidates.append((str(col), values))

        notes = []
        for i, (col_a, vals_a) in enumerate(candidates):
            for col_b, vals_b in candidates[i + 1:]:
                shared = vals_a & vals_b
                if not shared:
                    continue
                # WHICH KIND of overlap this is decides whether anything is wrong.
                # 'origin'/'destination' or 'billing_country'/'shipping_country'
                # share their WHOLE vocabulary by design — two roles of one entity
                # type, nothing to report. A stray value appearing in a column it
                # does not belong to shows up as a small minority of BOTH
                # vocabularies. Overlap shape separates them without needing to
                # know what the columns mean.
                jaccard = len(shared) / len(vals_a | vals_b)
                minority = (len(shared) / len(vals_a) < CATEGORY_LEAK_MAX_SHARE
                            and len(shared) / len(vals_b) < CATEGORY_LEAK_MAX_SHARE)
                sample = ", ".join(sorted(shared)[:4])
                more = f" (+{len(shared) - 4} more)" if len(shared) > 4 else ""

                if jaccard >= CATEGORY_SAME_DOMAIN_JACCARD:
                    # Same value domain. Only worth a word if the columns are also
                    # row-for-row identical, which makes one of them redundant.
                    try:
                        identical = bool(df[col_a].equals(df[col_b]))
                    except Exception:  # noqa: BLE001
                        identical = False
                    if identical:
                        notes.append(f"  '{col_a}' and '{col_b}' are identical row for row — "
                                     f"one is a redundant copy, not a second dimension.")
                    continue
                if minority:
                    notes.append(f"  '{col_a}' and '{col_b}' share {len(shared)} value(s): "
                                 f"{sample}{more}. That is a small minority of both columns, which "
                                 f"usually means a value belonging to one has leaked into the "
                                 f"other — verify it is a real '{col_a}' before reporting it as "
                                 f"one. (If these columns legitimately draw on the same list, "
                                 f"ignore this.)")
        return notes

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

        # ── categorical consistency ─────────────────────────────
        overlaps = self._overlapping_categories(df)
        lines.append("\ncategory consistency:")
        lines.extend(overlaps if overlaps else
                     ["  nothing to flag — no value looks out of place in its column"])

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
        # the file now matches the registry copy, so reading it back is safe —
        # record that so tooling does not warn about a correct action
        self.saved[name] = (full, df.shape)
        return f"Saved '{name}' ({df.shape[0]} rows) to {path}"

    # ───────────────────────────────────────────────── internals

    def _register(self, name: str, df: pd.DataFrame, source: str) -> str:
        self.datasets[name] = df
        # remembering the origin lets tooling warn when code re-reads the raw
        # file after a cleaned version already exists in the registry
        self.sources[name] = source
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
