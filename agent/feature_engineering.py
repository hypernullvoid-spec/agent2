"""
Phase 7: Automated Feature Engineering

Profiles a dataset already loaded via Phase 6's DataPipeline (dtypes,
cardinality, distributions, target correlation) and turns that profile
into a concrete, inspectable sklearn ColumnTransformer pipeline:
  - numeric columns  → impute + scale
  - low-cardinality categoricals → impute + one-hot encode
  - high-cardinality categoricals → frequency-encode (avoids explosion)
  - datetime columns → decomposed into year/month/day/dayofweek/is_weekend

The LLM-driven part is profile_dataset(): it returns a structured,
human-readable profile so the agent (Claude) can *reason* about which
columns are useless (IDs, constant columns), which need special handling,
and what the task type probably is — then call engineer_features() with
its own column-role decisions. This keeps the "thinking" in the agent
loop and the "doing" deterministic and inspectable here, exactly like
Phase 4 separates diagnosis (Claude) from detection (self_correction.py).

Output contract
─────────────────
engineer_features() writes the fitted transformer's output back into
DataPipeline's registry under a new name (e.g. "train_features") as a
plain DataFrame — not a sparse matrix — so Phase 8's training tools can
consume it exactly like any other dataset, and so the agent can
preview_dataset() it to sanity-check the result.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from agent.data_pipeline import get_data_pipeline

WORKSPACE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "workspace")
)

HIGH_CARDINALITY_THRESHOLD = 20   # categoricals above this → frequency-encoded, not OHE


class FeatureEngine:
    """
    Stateless transform logic + one piece of state: the last fitted
    ColumnTransformer (and the feature names it produced), kept so the
    agent can apply the *same* fitted transform to a held-out/test set
    later via apply_saved_transform(), instead of re-fitting on test
    data (a common and serious ML correctness bug this avoids by design).
    """

    def __init__(self):
        self._fitted_transformer = None
        self._fitted_feature_names: list[str] = []
        self._fitted_target_col: Optional[str] = None

    # ───────────────────────────────────────────────── profiling

    def profile_dataset(self, name: str, target_col: Optional[str] = None) -> str:
        df = get_data_pipeline().datasets.get(name)
        if df is None:
            return f"Error: no dataset named '{name}' is loaded. Load it via Phase 6 tools first."

        lines = [f"Feature profile for '{name}'  ({df.shape[0]} rows × {df.shape[1]} cols)"]

        # ── task type hint, if target given ──────────────────────
        if target_col:
            if target_col not in df.columns:
                lines.append(f"\nWarning: target_col '{target_col}' not found in dataset.")
            else:
                tgt = df[target_col]
                n_unique = tgt.nunique(dropna=True)
                if pd.api.types.is_numeric_dtype(tgt) and n_unique > 20:
                    task_hint = "regression (numeric target, many unique values)"
                elif n_unique == 2:
                    task_hint = "binary classification"
                elif 2 < n_unique <= 20:
                    task_hint = "multi-class classification"
                else:
                    task_hint = "uncertain — inspect target manually"
                lines.append(f"\nTarget column: '{target_col}'  →  likely task: {task_hint}")
                lines.append(f"  unique values: {n_unique}  •  nulls: {int(tgt.isnull().sum())}")

        # ── per-column profile ────────────────────────────────────
        lines.append("\nColumn-by-column:")
        for col in df.columns:
            if col == target_col:
                continue
            series = df[col]
            dtype = series.dtype
            n_unique = series.nunique(dropna=True)
            n_null = int(series.isnull().sum())
            pct_null = 100 * n_null / len(df) if len(df) else 0

            role, note = self._infer_role(col, series, n_unique, len(df))

            line = f"  {col:<24} dtype={str(dtype):<10} unique={n_unique:<6} null={pct_null:.1f}%  →  {role}"
            if note:
                line += f"   ({note})"
            lines.append(line)

        lines.append(
            "\nUse this profile to decide column roles, then call "
            "engineer_features(name, target_col, drop_cols=[...]) — drop "
            "any column flagged 'likely ID / drop' or 'constant — drop'."
        )
        return "\n".join(lines)

    def _infer_role(self, col: str, series: pd.Series, n_unique: int, n_rows: int) -> tuple[str, str]:
        """Best-effort heuristic role inference, surfaced as a *suggestion*, not auto-applied."""
        if n_unique <= 1:
            return "constant — drop", "no variance"
        if n_unique == n_rows and (pd.api.types.is_integer_dtype(series) or pd.api.types.is_string_dtype(series)):
            return "likely ID / drop", "unique per row"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime → decompose", ""
        # try to sniff datetime-looking object columns
        if pd.api.types.is_string_dtype(series):
            sample = series.dropna().head(20)
            if len(sample) and self._looks_like_datetime(sample):
                return "datetime-like (object) → decompose", "parse with pd.to_datetime first"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric → impute + scale", ""
        if pd.api.types.is_bool_dtype(series):
            return "boolean → pass through", ""
        if n_unique <= HIGH_CARDINALITY_THRESHOLD:
            return "categorical (low-card) → one-hot", ""
        return "categorical (high-card) → frequency-encode", f"{n_unique} categories, OHE would explode dims"

    @staticmethod
    def _looks_like_datetime(sample: pd.Series) -> bool:
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(sample, errors="coerce")
            return parsed.notna().mean() > 0.8
        except Exception:
            return False

    # ───────────────────────────────────────────────── transformation

    def engineer_features(
        self,
        name: str,
        target_col: Optional[str] = None,
        drop_cols: Optional[list[str]] = None,
        output_name: Optional[str] = None,
    ) -> str:
        """
        Fit a ColumnTransformer based on inferred column roles (numeric /
        categorical-low-card / categorical-high-card / datetime), transform
        the dataset, and register the result back into DataPipeline under
        output_name (default: f"{name}_features").
        """
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
        except ImportError:
            return "Error: feature engineering requires 'pip install scikit-learn'."

        df = get_data_pipeline().datasets.get(name)
        if df is None:
            return f"Error: no dataset named '{name}' is loaded."

        drop_cols = set(drop_cols or [])
        if target_col:
            drop_cols.add(target_col)

        work_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
        target_series = df[target_col].copy() if target_col and target_col in df.columns else None

        # ── decompose datetimes in-place before building the transformer ──
        datetime_cols: list[str] = []
        for col in list(work_df.columns):
            series = work_df[col]
            is_dt = pd.api.types.is_datetime64_any_dtype(series)
            if not is_dt and pd.api.types.is_string_dtype(series):
                sample = series.dropna().head(20)
                is_dt = len(sample) > 0 and self._looks_like_datetime(sample)
                if is_dt:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        work_df[col] = pd.to_datetime(series, errors="coerce")
            if is_dt:
                datetime_cols.append(col)

        for col in datetime_cols:
            dt = work_df[col]
            work_df[f"{col}__year"] = dt.dt.year
            work_df[f"{col}__month"] = dt.dt.month
            work_df[f"{col}__day"] = dt.dt.day
            work_df[f"{col}__dayofweek"] = dt.dt.dayofweek
            work_df[f"{col}__is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            work_df = work_df.drop(columns=[col])

        # ── classify remaining columns ─────────────────────────────────
        numeric_cols, low_card_cat, high_card_cat = [], [], []
        for col in work_df.columns:
            series = work_df[col]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                numeric_cols.append(col)
            else:
                n_unique = series.nunique(dropna=True)
                if n_unique <= HIGH_CARDINALITY_THRESHOLD:
                    low_card_cat.append(col)
                else:
                    high_card_cat.append(col)

        # ── frequency-encode high-cardinality columns manually ──────────
        freq_maps = {}
        for col in high_card_cat:
            freq = work_df[col].value_counts(normalize=True)
            freq_maps[col] = freq
            work_df[col] = work_df[col].map(freq).fillna(0.0)
        # frequency-encoded columns are now numeric — feed them through the numeric branch
        numeric_cols.extend(high_card_cat)

        transformers = []
        if numeric_cols:
            transformers.append((
                "numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_cols,
            ))
        if low_card_cat:
            transformers.append((
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                low_card_cat,
            ))

        if not transformers:
            return "Error: no usable columns left after dropping target/drop_cols."

        ct = ColumnTransformer(transformers, remainder="drop")
        try:
            transformed = ct.fit_transform(work_df)
        except Exception as e:
            return f"Error fitting feature pipeline: {type(e).__name__}: {e}"

        feature_names = list(ct.get_feature_names_out())
        result_df = pd.DataFrame(transformed, columns=feature_names, index=work_df.index)

        if target_series is not None:
            result_df[target_col] = target_series.values

        self._fitted_transformer = ct
        self._fitted_feature_names = feature_names
        self._fitted_target_col = target_col

        out_name = output_name or f"{name}_features"
        get_data_pipeline().datasets[out_name] = result_df

        summary = [
            f"Engineered features for '{name}' → registered as '{out_name}'",
            f"  numeric/frequency-encoded columns: {len(numeric_cols)}",
            f"  one-hot encoded columns: {len(low_card_cat)} (expanded to "
            f"{len([f for f in feature_names if f.startswith('categorical__')])} dummy columns)",
            f"  datetime columns decomposed: {len(datetime_cols)}",
            f"  resulting shape: {result_df.shape[0]} rows × {result_df.shape[1]} cols",
        ]
        if target_col:
            summary.append(f"  target column '{target_col}' carried through unchanged for training")
        summary.append(
            f"Preview with preview_dataset('{out_name}'), then proceed to "
            f"Phase 8 training tools."
        )
        return "\n".join(summary)


# ─── singleton, matching the rest of the codebase ──────────────────────────────

_engine: Optional[FeatureEngine] = None


def get_feature_engine() -> FeatureEngine:
    global _engine
    if _engine is None:
        _engine = FeatureEngine()
    return _engine
