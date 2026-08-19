"""
Data Cleaner — human-in-the-loop data cleaning + analysis (standalone).

A self-contained data-quality + analysis module: it diagnoses a dataset
against the standard cleaning checklist (missing values, redundancy,
structural errors, types/formats, anomalies), presents every finding to a
human for approval, applies ONLY the approved operations, then produces
analyst-facing summaries and grouped aggregations.

Modes
─────
1. Standalone CLI (no LLM, no swarn imports):
     python -m agent.data_cleaner path/to/data.csv [--auto-approve]
   Loads CSV/XLSX/PARQUET → prints a numbered cleaning plan → asks the
   human which ops to approve → applies them → saves "<name>_clean.csv"
   (original untouched) → prints a verification summary + describe().

2. Swarn-ready (registered into the agent's tool registry):
   register_into_swarn() lazily imports agent.tools and pushes 5 tools
   (clean_dataset / apply_cleaning / ask_human / describe_dataset /
   group_dataset) into TOOL_REGISTRY so the Swarn agent can run the same
   human-in-the-loop flow against DataPipeline registry datasets. In Swarn
   mode, ask_human prompts in the interactive REPL; in non-interactive
   contexts (swarn run, dashboard, CI) it returns "none" (report-only,
   nothing changes) unless SWARN_AUTO_APPROVE=1.

The module itself is pure pandas + stdlib — no LLM inside. The model (when
registered) only orchestrates: clean_dataset → ask_human → apply_cleaning.

Canonical apply order (safe, deterministic):
  flag_missing → cast/boolean/date/units → trim/case/syntax/split
  → impute → deduplicate → drop columns → drop rows → cap outliers
  → cross-field + rule checks (validation-only: flag columns, never delete)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd

# ─── thresholds (env-overridable, matching SWARN_* env conventions) ───────────

COL_NULL_DROP_THRESHOLD = float(os.environ.get("SWARN_CLEAN_COL_NULL_DROP", "0.5"))
ROW_NULL_DROP_THRESHOLD = float(os.environ.get("SWARN_CLEAN_ROW_NULL_DROP", "0.5"))
OUTLIER_Z_THRESHOLD = float(os.environ.get("SWARN_CLEAN_OUTLIER_Z", "3.0"))
MIN_OUTLIER_ROWS = 8

EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHONE_PATTERN = r"[+]?[0-9][0-9\s\-()]{7,}"

_BOOL_VOCAB = {"yes", "no", "y", "n", "true", "false", "t", "f", "1", "0"}
_BOOL_MAP = {
    "yes": True, "y": True, "true": True, "t": True, "1": True,
    "no": False, "n": False, "false": False, "f": False, "0": False,
}
_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
}
_UNIT_HINTS = ("price", "cost", "amount", "salary", "revenue", "weight",
               "distance", "height", "length", "rate", "fee")
_CURRENCY_SYMBOLS = ("$", "₹", "€", "£", "¥")

# canonical execution order per action (see module docstring)
_ACTION_ORDER = {
    "flag_missing": 1,
    "cast_types": 2, "convert_booleans": 2, "uniform_dates": 2, "harmonize_units": 2,
    "parse_years": 2,
    "trim_text": 3, "normalize_case": 3, "fix_syntax": 3, "split_column": 3,
    "standardize_categories": 3.5,   # after case/punctuation fixes, before dedup
    "fuzzy_merge": 3.6,
    "impute": 4,
    "deduplicate": 5,
    "drop_columns": 6,
    "drop_rows": 7,
    "filter_rows": 7,
    "cap_outliers": 8,
    "cross_field": 9, "rule_validate": 9,
}


# ─── small helpers ─────────────────────────────────────────────────────────────

def _is_numeric(series: pd.Series) -> bool:
    return series.dtype.kind in "iufc"


def _numlike(value) -> bool:
    try:
        float(str(value).replace(",", "").replace("$", "").replace("₹", ""))
        return True
    except (TypeError, ValueError):
        return False


# A clock reading carries no date. pd.to_datetime('23:48:13') still succeeds —
# it silently attaches TODAY — so a time-only column sails through any parse-based
# date test and then loses everything but the run date. Detected from the VALUES
# (not the column name) so it holds for any language or naming convention.
_CLOCK_ONLY = re.compile(
    r"^\s*\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*(?:[AaPp]\.?[Mm]\.?)?\s*$")


def _looks_like_clock_time(samples: pd.Series) -> bool:
    """True when the column is times-of-day with no date part ('23:48:13')."""
    if not len(samples):
        return False
    if not (samples.dtype == object or pd.api.types.is_string_dtype(samples)):
        return False
    strings = [v for v in samples.dropna() if isinstance(v, str)]
    if not strings:
        return False
    hits = sum(1 for v in strings if _CLOCK_ONLY.match(v))
    return hits / len(strings) >= 0.8


def _is_temporal_text(samples: pd.Series) -> bool:
    """Date OR time-of-day — used to keep text ops off any temporal column."""
    return _looks_like_datetime(samples) or _looks_like_clock_time(samples)


def _looks_like_datetime(samples: pd.Series) -> bool:
    if not len(samples):
        return False
    if not (samples.dtype == object or pd.api.types.is_string_dtype(samples)):
        return False  # numeric Series would be misread as nanosecond-epoch dates
    if _looks_like_clock_time(samples):
        return False  # times-of-day are not dates; converting them destroys them
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(samples, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(samples, errors="coerce")
    return parsed.notna().mean() >= 0.8


# ── date-convention helpers ────────────────────────────────────────────────────
# '05-03-2023' is March 5th or May 3rd depending on a convention the value itself
# cannot reveal. These decide it ONCE PER COLUMN rather than per value, which is
# what keeps a cleaned date column internally consistent.

_DMY_SHAPE = re.compile(r"^\s*(\d{1,2})\D(\d{1,2})\D(\d{2,4})\s*$")


def _detect_dayfirst(series: pd.Series):
    """True/False when the column PROVES its own layout, None when it cannot.

    Delegates to the analysis module so there is ONE proof rule in the codebase;
    falls back to 'unproven' if that module is unavailable.
    """
    try:
        from agent.data_analysis import _detect_dayfirst as _proof
    except Exception:  # noqa: BLE001 — never let a cleaning op die on an import
        return None
    try:
        return _proof(series)
    except Exception:  # noqa: BLE001
        return None


def _parse_uniform(series: pd.Series, dayfirst: bool) -> pd.Series:
    """Parse with ONE day/month convention for every value in the column.

    format="mixed" still allows each value its own LAYOUT (ISO here, D-M-Y
    there, which real extracts do contain) but `dayfirst` pins how every
    ambiguous value is read — that combination is what the original code was
    missing when it let pandas default to month-first per value.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=dayfirst)
        except (TypeError, ValueError):
            return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)


def _time_is_empty(parsed: pd.Series) -> bool:
    """True when every parsed value sits at midnight — i.e. there is no clock
    component to lose by formatting as a plain date."""
    valid = parsed.dropna()
    if not len(valid):
        return True
    try:
        return bool((valid.dt.hour.eq(0) & valid.dt.minute.eq(0) & valid.dt.second.eq(0)).all())
    except (AttributeError, TypeError):
        return True


def _dmy_shaped_any(series: pd.Series) -> bool:
    """Whether the column contains any value whose layout is ambiguous at all —
    ISO '2023-02-24' is not, so it never earns a convention warning."""
    if pd.api.types.is_datetime64_any_dtype(series) or _is_numeric(series):
        return False
    return any(_DMY_SHAPE.match(str(v)) for v in series.dropna().head(2000))


def _looks_like_email(strings: list) -> bool:
    if not strings:
        return False
    hits = sum(1 for s in strings if re.fullmatch(EMAIL_PATTERN, s))
    return hits / len(strings) >= 0.8


_YEAR_RE = re.compile(r"(?:\b|[(\s-])((?:19|20)\d{2})")


def _looks_like_year(samples: pd.Series) -> bool:
    """True if ≥80% of the column's strings embed a 4-digit year
    (e.g. '2021', '-2021', '(2010–2022)', '(2021– )')."""
    if not len(samples):
        return False
    if not (samples.dtype == object or pd.api.types.is_string_dtype(samples)):
        return False
    if _looks_like_datetime(samples):
        return False  # full dates like '2023-01-05' are dates, not years
    strings = [s for s in samples.dropna() if isinstance(s, str)]
    if not strings:
        return False
    hits = sum(1 for s in strings if _YEAR_RE.search(s))
    return hits / len(strings) >= 0.8


_ID_NAME_TOKENS = {
    "id", "ids", "uuid", "guid", "key", "code", "no", "num", "number",
    "index", "idx", "serial", "sr", "sno", "ref", "pk",
}


def _name_tokens(name) -> set:
    """Whole words in a column name — 'start_date' → {'start','date'},
    'orderId' → {'order','id'}. Substring matching would find 'to' inside
    'customer' and 'history', pairing unrelated columns."""
    return {t.lower() for t in
            re.split(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])", str(name)) if t}


def _name_looks_like_id(name: str) -> bool:
    """True if the column name carries an identifier token — 'order_id',
    'orderId', 'customer code'. 'units_sold' does not."""
    return bool(_name_tokens(name) & _ID_NAME_TOKENS)


_RANGE_START_TOKENS = {"start", "from", "min", "begin", "opened", "first", "since"}
_RANGE_END_TOKENS = {"end", "to", "max", "finish", "closed", "last", "until", "through"}


def _looks_like_row_counter(series: pd.Series) -> bool:
    """True if the values form a near-perfect 1-step run (0,1,2… / 1,2,3…).

    A row counter is safe to drop; a column of merely-distinct integers
    (units_sold, amounts in paise) is real data and must be kept.
    """
    values = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(values) < 3:
        return False
    diffs = values.sort_values().diff().dropna()
    return bool((diffs == 1).mean() >= 0.99)


def _looks_zero_padded(values) -> bool:
    """True if any value is a zero-padded all-digit code ('07001', '0042').

    Such columns must never be cast to numbers: the leading zero is
    unrecoverable, and they are identifiers (PIN/ZIP, account, SKU), not measures.
    """
    for v in values:
        if isinstance(v, str):
            s = v.strip()
            if len(s) > 1 and s[0] == "0" and s.isdigit():
                return True
    return False


def _iqr_outlier_fraction(series: pd.Series) -> float:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((series < lo) | (series > hi)).mean())


def _looks_log_distributed(series: pd.Series) -> bool:
    """True when a column spans orders of magnitude — vote counts, revenues,
    populations, city sizes.

    For these the long tail is REAL structure, not contamination, and IQR
    capping is the wrong instrument: it flattens the top decile into a single
    tied value and destroys the ranking. The test is whether taking logs makes
    the 'outliers' largely disappear — if it does, the raw tail is shape.
    """
    positive = series[series > 0].dropna()
    if len(positive) < MIN_OUTLIER_ROWS or positive.nunique() < 5:
        return False
    median = float(positive.median())
    if median <= 0 or float(positive.max()) / median < 50:
        return False
    raw = _iqr_outlier_fraction(positive)
    logged = _iqr_outlier_fraction(positive.map(math.log10))
    return raw > 0.02 and logged < raw / 2


def _canonical_label(value) -> str:
    """Aggressively normalised form of a category label, used ONLY to spot
    spellings of the same thing: 'Acme Corp.' / 'ACME  Corp' → 'acme corp'.

    Deliberately not fuzzy: 'Store 1' and 'Store 2' stay distinct. Only
    case, punctuation and whitespace are ignored.
    """
    s = re.sub(r"[^\w\s]+", "", str(value).lower())
    return re.sub(r"\s+", " ", s).strip()


# long legal-entity forms → their short form, so 'Acme Corporation' and
# 'Acme Corp' reduce to the same key. Deterministic and explainable — unlike
# raw edit distance, this cannot merge two genuinely different companies.
_ENTITY_SUFFIXES = {
    "corporation": "corp", "corporated": "corp", "incorporated": "inc",
    "limited": "ltd", "company": "co", "private": "pvt", "brothers": "bros",
    "international": "intl", "associates": "assoc", "manufacturing": "mfg",
    "department": "dept", "industries": "ind", "enterprises": "ent",
}
FUZZY_MAX_LABELS = 500          # similarity pass is O(n²); refuse beyond this


def _entity_key(value) -> str:
    """Canonical label with long legal-entity words folded to short forms."""
    return " ".join(_ENTITY_SUFFIXES.get(t, t) for t in _canonical_label(value).split())


def _digit_signature(value) -> tuple:
    """The numbers inside a label. 'Store 1' and 'Store 2' differ here, so they
    are never merged however similar their letters look."""
    return tuple(re.findall(r"\d+", str(value)))


def _profile(df: pd.DataFrame) -> dict:
    """Compact per-column statistics — cheap enough to keep for a before/after
    comparison without holding a second copy of the data."""
    profile = {"rows": int(len(df)), "columns": {}}
    for col in df.columns:
        s = df[col]
        nn = s.dropna()
        entry = {
            "dtype": str(s.dtype),
            "non_null": int(len(nn)),
            "nulls": int(s.isnull().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if _is_numeric(s) and len(nn):
            entry.update(
                mean=float(nn.mean()),
                std=float(nn.std()) if len(nn) > 1 else 0.0,
                min=float(nn.min()), median=float(nn.median()), max=float(nn.max()),
            )
        elif len(nn):
            top = nn.value_counts()
            entry.update(top=str(top.index[0]), top_share=float(top.iloc[0] / len(nn)))
        profile["columns"][str(col)] = entry
    return profile


def compare_profiles(before: dict, after: dict, sigma_tol: float = 0.25) -> list[str]:
    """Human-readable before/after diff. Shifts larger than `sigma_tol` standard
    deviations are marked ⚠ — that is how you catch a cleaning step that quietly
    moved your data rather than tidying it."""
    lines: list[str] = []
    rows_b, rows_a = before.get("rows", 0), after.get("rows", 0)
    if rows_b != rows_a:
        pct = 100 * (rows_a - rows_b) / max(rows_b, 1)
        lines.append(f"rows {rows_b} → {rows_a}  ({rows_a - rows_b:+d}, {pct:+.1f}%)")
    cols_b, cols_a = before.get("columns", {}), after.get("columns", {})
    removed = [c for c in cols_b if c not in cols_a]
    added = [c for c in cols_a if c not in cols_b]
    if removed:
        lines.append(f"columns removed: {removed}")
    if added:
        lines.append(f"columns added:   {added}")
    for col, b in cols_b.items():
        a = cols_a.get(col)
        if a is None:
            continue
        notes = []
        if b["dtype"] != a["dtype"]:
            notes.append(f"dtype {b['dtype']}→{a['dtype']}")
        if b["nulls"] != a["nulls"]:
            notes.append(f"nulls {b['nulls']}→{a['nulls']}")
        if b["unique"] != a["unique"]:
            notes.append(f"distinct {b['unique']}→{a['unique']}")
        flag = ""
        if "mean" in b and "mean" in a:
            shift = a["mean"] - b["mean"]
            if abs(shift) > 1e-12:
                sigma = abs(shift) / b["std"] if b.get("std") else 0.0
                notes.append(f"mean {b['mean']:.6g}→{a['mean']:.6g} ({shift:+.3g}"
                             + (f", {sigma:.2f}σ)" if sigma else ")"))
                if sigma >= sigma_tol:
                    flag = "⚠ "
            for edge in ("min", "max"):
                if edge in b and edge in a and b[edge] != a[edge]:
                    notes.append(f"{edge} {b[edge]:.6g}→{a[edge]:.6g}")
        elif "top_share" in b and "top_share" in a:
            if b["top"] != a["top"]:
                notes.append(f"most common '{b['top']}'→'{a['top']}'")
            elif abs(a["top_share"] - b["top_share"]) >= 0.05:
                notes.append(f"top share {b['top_share']:.0%}→{a['top_share']:.0%}")
        if notes:
            lines.append(f"{flag}{col}: " + "; ".join(notes))
    return lines


def _row_preview(df: pd.DataFrame, indices: list, limit: int = 3, width: int = 4) -> str:
    """Compact preview of the actual rows an op would delete, so a human is
    never asked to approve a deletion sight-unseen."""
    if not indices:
        return ""
    try:
        sub = df.loc[indices[:limit]]
    except (KeyError, TypeError):
        return ""
    cols = list(sub.columns)[:width]
    rows = []
    for idx, row in sub.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            text = "∅" if pd.isna(v) else str(v)
            cells.append(f"{c}={text[:18]}")
        more = "…" if len(sub.columns) > width else ""
        rows.append(f"[{idx}] " + ", ".join(cells) + more)
    return " | ".join(rows)


def _jsonable(value):
    """Convert numpy/pandas scalars and containers into JSON-safe Python."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if pd.isna(value):
        return None
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


class _OpSkipped(Exception):
    """An operation cannot run — e.g. its column was removed by an earlier op."""


def parse_approval(answer: str, op_ids: list[str],
                   optional_ids: Optional[list] = None) -> tuple[list[str], dict]:
    """Parse a human approval answer into (approved op ids, per-op params).

    Supported forms:
      "all"                       → every RECOMMENDED op (see optional_ids)
      "none" / ""                 → nothing
      "op1 op3 op5"               → those ops, recommended or not
      "op4:factor=83.2 op9:rule=x" → those ops with per-op key=value params

    `optional_ids` are operations the tool marked NOT RECOMMENDED — capping a
    naturally long tail, imputing a structurally-missing block. "all" must not
    sweep those in: an operation the tool advises against cannot also be part of
    "approve everything". Naming one explicitly still approves it.
    """
    answer = (answer or "").strip()
    if not answer:
        return [], {}
    if answer.lower() == "none":
        return [], {}
    approved: list[str] = []
    params: dict = {}
    tokens = [t.strip() for t in re.split(r"[,\s]+", answer) if t.strip()]
    # 'all' may be combined with per-op params — "all op4:factor=83.2" approves
    # everything AND parameterises op4, rather than silently approving only op4.
    approve_all = any(t.lower() in ("all", "approve", "yes") for t in tokens)
    for token in tokens:
        op_id, _, param_str = token.partition(":")
        if op_id not in op_ids:
            continue
        if op_id not in approved:
            approved.append(op_id)
        if param_str:
            params[op_id] = {}
            for kv in re.split(r"[;,]+\s*", param_str):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    params[op_id][k.strip()] = v.strip()
    if approve_all:
        skip = set(optional_ids or ())
        named = set(approved)
        approved = [o for o in op_ids if o not in skip or o in named]
    return approved, params


def ask_human(question: str, options: Optional[list[str]] = None) -> str:
    """Ask a human a question on the terminal.

    Non-interactive (piped/stdin not a TTY): returns "approve all" when
    SWARN_AUTO_APPROVE is enabled, else "none" (report-only). Never raises
    on EOF — treated as "none".
    """
    interactive = bool(sys.stdin.isatty()) and bool(sys.stdout.isatty())
    if not interactive:
        auto = os.environ.get("SWARN_AUTO_APPROVE", "").strip().lower()
        return "approve all" if auto in ("1", "true", "yes") else "none"
    prompt = question
    if options:
        prompt += "\n  " + " / ".join(options)
    try:
        return input(prompt + "\n> ").strip()
    except EOFError:
        return "none"


# ───────────────────────────────────────────────────────────────────────────────

class DataCleaner:
    """Detects and applies cleaning operations under human approval."""

    def __init__(self, df: Optional[pd.DataFrame] = None, source: str = "",
                 target_col: Optional[str] = None):
        self.df = df
        self.source = source
        self.target_col = target_col
        self.ops: dict[str, dict] = {}
        self.plan: str = ""
        self.applied_ops: list[dict] = []      # audit trail / replayable recipe
        self.shape_before: Optional[tuple] = None
        self.profile_before: Optional[dict] = None

    # ── loading / saving ──────────────────────────────────────────────

    def load_file(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".xlsx" or ext == ".xls":
                self.df = pd.read_excel(path)
            elif ext == ".parquet":
                self.df = pd.read_parquet(path)
            elif ext == ".json":
                self.df = pd.read_json(path)
            else:
                self.df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            return f"Error reading '{path}': {type(e).__name__}: {e}"
        self.source = path
        return (f"Loaded {path}: {self.df.shape[0]} rows × {self.df.shape[1]} cols. "
                "Run diagnose() before any change.")

    def save_file(self, path: str) -> None:
        if self.df is None:
            raise ValueError("No data to save.")
        if path.endswith(".parquet"):
            self.df.to_parquet(path, index=False)
        else:
            self.df.to_csv(path, index=False)

    def op_ids(self) -> list[str]:
        return list(self.ops)

    def optional_op_ids(self) -> list[str]:
        """Ops the plan advises against — 'all' must not include these."""
        return [i for i, op in self.ops.items() if not op.get("recommended", True)]

    # ── registration ──────────────────────────────────────────────────

    def _register_op(self, action: str, label: str, columns: Optional[list] = None,
                     extra: Optional[dict] = None, needs_param: bool = False,
                     recommended: bool = True) -> str:
        op_id = f"op{len(self.ops) + 1}"
        self.ops[op_id] = {
            "id": op_id, "action": action, "label": label,
            "columns": columns or [], "extra": extra or {},
            "needs_param": needs_param, "recommended": recommended,
        }
        if isinstance(self.plan, list):
            hint = "  (needs params, e.g. 'opX: ...')" if needs_param else ""
            tag = "" if recommended else "  (optional)"
            self.plan.append(f"  {op_id}: {label}{hint}{tag}")
        return op_id

    # ── diagnosis ─────────────────────────────────────────────────────

    def diagnose(self) -> str:
        df = self.df
        self.ops, self.plan = {}, []
        if df is None:
            return "Error: no data loaded."
        if self.target_col and self.target_col not in df.columns:
            return (f"Error: target column '{self.target_col}' is not in the data — "
                    f"nothing was diagnosed. Available columns: {list(df.columns)}")
        if not df.index.is_unique:
            # row-dropping ops address rows by index label; a repeated label would
            # take good rows down with the bad ones.
            df = df.reset_index(drop=True)
            self.df = df
        n_rows, n_cols = df.shape
        self.shape_before = (n_rows, n_cols)
        self.profile_before = _profile(df)

        self.plan.append(
            f"CLEANING PLAN — '{self.source}' ({n_rows} rows × {n_cols} cols)\n"
            "Human approval required before any change."
        )

        null_cols = [c for c in df.columns if int(df[c].isnull().sum()) > 0]
        col_null_frac = df.isnull().mean()
        row_null_frac = df.isnull().mean(axis=1)

        # columns whose blanks land in the SAME rows describe one incomplete
        # GROUP of records; imputing each separately invents whole rows
        missing_together: dict[str, tuple] = {}
        gappy = [c for c in null_cols if col_null_frac[c] < 1.0]
        if len(gappy) >= 2 and n_rows:
            null_corr = df[gappy].isnull().corr()
            for col in gappy:
                partners = [o for o in gappy if o != col
                            and pd.notna(null_corr.loc[col, o])
                            and null_corr.loc[col, o] >= 0.9]
                if partners:
                    block = [col] + partners
                    missing_together[col] = (block,
                                             int(df[block].isnull().all(axis=1).sum()))

        # ── A. MISSING VALUES ─────────────────────────────────────
        self.plan.append("\n[A] MISSING VALUES")
        high_null = [c for c in df.columns if col_null_frac.get(c, 0) >= COL_NULL_DROP_THRESHOLD]
        for col in high_null:
            n = int(df[col].isnull().sum())
            self._register_op(
                "drop_columns",
                f"drop '{col}' — {n} nulls ({100 * n / n_rows:.1f}%, ≥ "
                f"{int(COL_NULL_DROP_THRESHOLD * 100)}% missing)",
                columns=[col],
            )
        low_null = [c for c in null_cols if c not in high_null]
        for col in low_null:
            n = int(df[col].isnull().sum())
            if self.target_col and col == self.target_col:
                idx = df.index[df[col].isnull()].tolist()
                self._register_op(
                    "drop_rows",
                    f"drop {len(idx)} rows missing target '{col}' "
                    f"(imputing the target would shrink its variance)\n"
                    f"        rows: {_row_preview(df, idx)}",
                    extra={"indices": idx, "rule": "target_null", "column": col},
                )
                continue
            strategy = "median" if _is_numeric(df[col]) else "most_frequent"
            if col in missing_together:
                # Fabricating an entire record is at least as destructive as
                # flattening a heavy tail, so it gets the same treatment: demoted
                # so a blanket 'all' does NOT sweep it in. A warning alone was not
                # enough — it printed, and 'all' imputed anyway.
                block, rows_affected = missing_together[col]
                others = [c for c in block if c != col]
                self._register_op(
                    "impute",
                    f"NOT RECOMMENDED for '{col}' ({n} nulls, {100 * n / n_rows:.1f}%): blank in "
                    f"the SAME {rows_affected:,} rows as {others}. Those rows are one incomplete "
                    f"GROUP, not scattered gaps — filling these columns invents a whole record, "
                    f"and any per-period average over them is the fill value rather than data. "
                    f"Prefer dropping those rows. Approve only if you need this column and "
                    f"accept the invented values.",
                    columns=[col], extra={"strategy": strategy, "co_missing": others},
                    recommended=False,
                )
                continue
            self._register_op(
                "impute",
                f"impute '{col}' ({n} nulls, {100 * n / n_rows:.1f}%) with {strategy}",
                columns=[col], extra={"strategy": strategy},
            )
        flag_cols = [c for c in low_null if not (self.target_col and c == self.target_col)]
        if flag_cols:
            self._register_op(
                "flag_missing",
                f"add missing-indicator columns for {flag_cols}",
                columns=flag_cols,
            )
        less_info = df.index[row_null_frac >= ROW_NULL_DROP_THRESHOLD].tolist()
        if less_info:
            self._register_op(
                "drop_rows",
                f"drop {len(less_info)} 'less-info' rows (≥ {int(ROW_NULL_DROP_THRESHOLD * 100)}% "
                f"fields missing)\n        rows: {_row_preview(df, less_info)}",
                extra={"indices": less_info, "rule": "row_null_frac",
                       "threshold": ROW_NULL_DROP_THRESHOLD},
            )

        # ── B. REDUNDANCY & IRRELEVANT ──────────────────────────────
        self.plan.append("\n[B] REDUNDANCY & IRRELEVANT DATA")
        dup_mask = df.duplicated()
        n_dups = int(dup_mask.sum())
        # rows that are duplicates only once text is trimmed/lower-cased would be
        # invisible to a raw duplicated() check, but dedup runs AFTER those fixes.
        n_norm_dups = n_dups
        obj_cols = [c for c in df.columns
                    if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
        if obj_cols and n_rows:
            norm = df.copy()
            for c in obj_cols:
                norm[c] = df[c].map(lambda v: _canonical_label(v) if isinstance(v, str) else v)
            n_norm_dups = int(norm.duplicated().sum())
        if n_dups or n_norm_dups:
            samples = df.index[dup_mask][:3].tolist()
            if n_norm_dups > n_dups:
                label = (f"drop duplicate rows — {n_dups} exact, {n_norm_dups} once text is "
                         f"trimmed/standardised (approve the [C] text ops too, or only "
                         f"{n_dups} will go)")
            else:
                label = f"drop {n_dups} duplicate rows (e.g. indices {samples})"
            self._register_op("deduplicate", label)
        drop_reasons: dict[str, str] = {}
        for col in df.columns:
            if col in high_null:
                continue
            nun = df[col].nunique(dropna=True)
            if nun <= 1:
                drop_reasons[col] = "constant"
            elif nun == n_rows:
                # Unique-per-row alone is NOT enough: real measures (units_sold,
                # amounts in paise) are often all-distinct. Demand a second signal.
                s = df[col].dropna().head(50)
                name_id = _name_looks_like_id(col)
                if df[col].dtype.kind in "iu":
                    is_id = name_id or _looks_like_row_counter(df[col])
                elif df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                    is_id = (
                        all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_\-]+", v) for v in s)
                        and not _is_temporal_text(s)
                        and (name_id or not all(_numlike(v) for v in s))
                    )
                else:
                    is_id = False
                if is_id:
                    drop_reasons[col] = ("ID-like (name looks like an identifier)"
                                         if name_id else "ID-like (row counter)")
        # one op per column, so a human can keep one and drop another
        for col, reason in drop_reasons.items():
            self._register_op(
                "drop_columns",
                f"drop irrelevant column '{col}' ({reason})",
                columns=[col],
            )
        dt_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dt_cols.append(col)
            elif _looks_like_datetime(df[col].dropna()):
                dt_cols.append(col)
        if dt_cols:
            self._register_op(
                "filter_rows",
                f"filter by date range on {dt_cols}",
                columns=dt_cols, needs_param=True, recommended=False,
            )

        # ── C. STRUCTURAL ERRORS ────────────────────────────────────
        self.plan.append("\n[C] STRUCTURAL ERRORS")
        str_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
        trim_cols, trim_total = [], 0
        case_cols, case_total = [], 0
        syntax_cols, syntax_total = [], 0
        split_cols, split_total = [], 0
        for col in str_cols:
            nonnull = df[col].dropna()
            strings = [v for v in nonnull if isinstance(v, str)]
            if not strings:
                continue
            if _is_temporal_text(nonnull) or _looks_like_email(strings):
                continue
            bad_trim = sum(1 for v in strings if v != v.strip())
            bad_case = sum(1 for v in strings if any(c.islower() for c in v) and any(c.isupper() for c in v))
            bad_syntax = sum(1 for v in strings if re.search(r"[\x00-\x1f]|&[a-z#0-9]+;| {2,}", v))
            comma_split = [v for v in strings if v.count(", ") == 1]
            if bad_trim:
                trim_cols.append(col); trim_total += bad_trim
            if bad_case:
                case_cols.append(col); case_total += bad_case
            if bad_syntax:
                syntax_cols.append(col); syntax_total += bad_syntax
            if len(comma_split) >= max(2, int(0.8 * len(strings))):
                split_cols.append(col); split_total += len(comma_split)
        if trim_cols:
            self._register_op("trim_text", f"trim leading/trailing whitespace in {trim_cols} ({trim_total} cells)", columns=trim_cols)
        if case_cols:
            self._register_op("normalize_case", f"normalize case in {case_cols} ({case_total} cells)", columns=case_cols)
        if syntax_cols:
            self._register_op("fix_syntax", f"fix bad chars/HTML entities/multi-space in {syntax_cols} ({syntax_total} cells)", columns=syntax_cols)
        if split_cols:
            self._register_op(
                "split_column", f"split 'City, State'-style values in {split_cols}",
                columns=split_cols, needs_param=True, recommended=False,
            )
        # category labels that are the same thing spelled differently
        for col in str_cols:
            strings = [v for v in df[col].dropna() if isinstance(v, str)]
            if not strings or _looks_like_email(strings) or _is_temporal_text(df[col].dropna()):
                continue
            uniques = set(strings)
            if not 2 <= len(uniques) <= 200:
                continue
            groups: dict[str, list] = {}
            for v in uniques:
                groups.setdefault(_canonical_label(v), []).append(v)
            collapsible = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
            if not collapsible:
                continue
            shown = "; ".join(f"{v} → one" for v in list(collapsible.values())[:2])
            self._register_op(
                "standardize_categories",
                f"merge {sum(len(v) for v in collapsible.values())} spelling variant(s) of "
                f"{len(collapsible)} label(s) in '{col}' (case/punctuation/spacing only, "
                f"most common spelling wins) — e.g. {shown}",
                columns=[col],
            )
        # near-matches beyond pure punctuation: 'Acme Corporation' vs 'Acme Corp'.
        # needs_param means a blanket 'approve all' will NOT silently run this.
        for col in str_cols:
            strings = [v for v in df[col].dropna() if isinstance(v, str)]
            if not strings or _looks_like_email(strings):
                continue
            uniques = set(strings)
            if not 2 <= len(uniques) <= FUZZY_MAX_LABELS:
                continue
            keyed: dict[str, set] = {}
            for v in uniques:
                keyed.setdefault(_entity_key(v), set()).add(_canonical_label(v))
            beyond = {k: v for k, v in keyed.items()
                      if len(v) > 1 and len({_digit_signature(x) for x in v}) == 1}
            if not beyond:
                continue
            shown = "; ".join(" ≈ ".join(sorted(v)) for v in list(beyond.values())[:2])
            self._register_op(
                "fuzzy_merge",
                f"merge near-identical labels in '{col}' — {len(beyond)} group(s), e.g. {shown}"
                f"\n        reply '<op>:apply=1' for entity-suffix merging only, or add"
                f" ';threshold=0.9' to also merge typos (never merges labels whose digits differ)",
                columns=[col], needs_param=True, recommended=False,
            )

        # ── D. TYPES & FORMATS ─────────────────────────────────────
        self.plan.append("\n[D] TYPES & FORMATS")
        year_cols = []
        for col in str_cols:
            nonnull = df[col].dropna()
            if len(nonnull) and _looks_like_year(nonnull):
                year_cols.append(col)
        if year_cols:
            self._register_op(
                "parse_years",
                f"parse year/year-range strings → year in {year_cols}",
                columns=year_cols,
            )
        cast_cols, bool_cols, date_cols, unit_cols = [], [], [], []
        for col in str_cols:
            if col in year_cols:
                continue
            nonnull = df[col].dropna()
            if not len(nonnull):
                continue
            vals = {str(v).strip().lower() for v in nonnull.unique()}
            if vals <= _BOOL_VOCAB and len(vals) <= 8:
                bool_cols.append(col)
                continue
            if all(_numlike(v) for v in nonnull):
                if _looks_zero_padded(nonnull):
                    continue  # '07001' is a code — casting would eat the leading zero
                cast_cols.append(col)
                continue
            if _looks_like_datetime(nonnull):
                date_cols.append(col)
                continue
            hint = col.lower()
            if any(k in hint for k in _UNIT_HINTS) and any(
                    isinstance(v, str) and (v.startswith(_CURRENCY_SYMBOLS) or re.search(r"[A-Za-z]{2,3}$", v))
                    for v in nonnull):
                unit_cols.append(col)
        if cast_cols:
            self._register_op("cast_types", f"cast text→number for {cast_cols}", columns=cast_cols)
        if bool_cols:
            self._register_op("convert_booleans", f"map Yes/No, 1/0 → boolean for {bool_cols}", columns=bool_cols)
        if date_cols:
            self._register_op("uniform_dates", f"convert mixed date formats → YYYY-MM-DD for {date_cols}", columns=date_cols)
        for col in unit_cols:
            self._register_op(
                "harmonize_units",
                f"harmonise units/currency in '{col}'",
                columns=[col], needs_param=True, recommended=False,
            )

        # ── E. ANOMALIES ───────────────────────────────────────────
        self.plan.append("\n[E] ANOMALIES")
        outlier_cols = []
        for col in df.columns:
            if _is_numeric(df[col]):
                s = df[col].dropna()
            elif col in cast_cols:
                # numbers stored as text (very common in CSV/Excel exports) are
                # about to become numeric — analyse them now or they never get checked
                s = pd.to_numeric(
                    df[col].map(lambda x: str(x).replace(",", "") if isinstance(x, str) else x),
                    errors="coerce",
                ).dropna()
            else:
                continue
            if len(s) < MIN_OUTLIER_ROWS:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((s < lo) | (s > hi)).sum())
            if n_out:
                # Cap to the SAME fence that detected them. Percentile bounds are
                # computed from the contaminated column, so a handful of sentinel
                # values (99999, -1) end up defining their own limit and survive
                # capping — while the plan still claims all n_out were handled.
                bounds = (float(lo), float(hi))
                outlier_cols.append(col)
                if _looks_log_distributed(s):
                    # heavy tail = real data. Do not recommend flattening it.
                    span = float(s.max()) / max(float(s.median()), 1e-9)
                    self._register_op(
                        "cap_outliers",
                        f"NOT RECOMMENDED for '{col}': {n_out} values sit beyond the IQR "
                        f"bounds, but this column spans {span:,.0f}x from its middle to its "
                        f"largest value — a naturally long tail (counts, money, popularity). "
                        f"Those extremes are almost certainly real, and clipping would tie "
                        f"{100 * n_out / len(s):.0f}% of rows at {bounds[1]:.3g}, destroying the "
                        f"ranking. Approve only if you know they are errors — and prefer "
                        f"'mode=null' to blank them. Otherwise leave it and read it on a log "
                        f"scale: plot_column(..., log_scale=true).",
                        columns=[col], extra={"bounds": bounds, "heavy_tail": True},
                        recommended=False,
                    )
                else:
                    self._register_op(
                        "cap_outliers",
                        f"cap outliers in '{col}' ({n_out} values beyond IQR bounds) → clip to "
                        f"[{bounds[0]:.3g}, {bounds[1]:.3g}]",
                        columns=[col], extra={"bounds": bounds},
                    )
        if not outlier_cols:
            self.plan.append("  none detected")

        cross_pairs = []
        for a in df.columns:
            if not (_name_tokens(a) & _RANGE_START_TOKENS):
                continue
            for b in df.columns:
                if b != a and (_name_tokens(b) & _RANGE_END_TOKENS):
                    cross_pairs.append((a, b))
        if cross_pairs:
            for a, b in cross_pairs:
                self._register_op(
                    "cross_field",
                    f"validate '{b}' ≥ '{a}' (cross-field date check)",
                    columns=[a, b], extra={"rule": f"{b} >= {a}"},
                    recommended=False,
                )

        rule_cols = []
        for col in df.columns:
            hint = col.lower()
            if re.search(r"\be?mail", hint):
                rule_cols.append((col, "email", EMAIL_PATTERN))
            elif re.search(r"\b(?:phone|telephone|mobile|tel|contact)\b", hint):
                rule_cols.append((col, "phone", PHONE_PATTERN))
        for col, kind, pattern in rule_cols:
            self._register_op(
                "rule_validate",
                f"validate '{col}' against {kind} pattern",
                columns=[col], extra={"pattern": pattern}, recommended=False,
            )

        self.plan.append(
            "\nReply to ask_human with: 'all' (every RECOMMENDED op — those marked NOT "
            "RECOMMENDED are skipped unless you name them) | 'none' | op ids ('op1 op3 op5') | "
            "or ops with params ('op4:factor=83.2'; separate several with ';').\n"
            "  impute      strategy=median|mean|mode|constant|ffill|bfill|interpolate, "
            "value=<x>, by=<col> (fill within groups)\n"
            "  cap_outliers  mode=null to blank extreme values instead of rewriting them\n"
            "  fuzzy_merge   apply=1, threshold=0.9 (opt-in; 'all' will NOT run it)\n"
            "NOTE: cross-field/rule ops only add validation flag columns — they never delete data.\n"
            "NOTE: capping REWRITES values; each affected row is marked in '<col>_was_capped'."
        )
        self.plan = "\n".join(self.plan)
        return self.plan

    # ── apply ────────────────────────────────────────────────────────

    def apply_cleaning(self, operations: list[str], params: Optional[dict] = None) -> str:
        if self.df is None:
            return "Error: no data loaded."
        if not self.ops:
            self.diagnose()
        params = params or {}
        df = self.df.copy()
        unknown = [o for o in operations if o not in self.ops]
        selected = [self.ops[o] for o in operations if o in self.ops]
        selected.sort(key=lambda op: _ACTION_ORDER.get(op["action"], 99))

        applied: list[str] = []
        skipped_param: list[str] = []
        skipped_done: list[str] = []
        skipped_stale: list[str] = []
        failed: list[str] = []
        violations: list[str] = []
        for op in selected:
            if op.get("applied"):
                skipped_done.append(op["label"])
                continue
            op_params = params.get(op["id"], {})
            if op["needs_param"] and not op_params:
                skipped_param.append(op["label"])
                continue
            # each op runs on its own copy: a failure can never leave the frame
            # half-transformed, and the remaining approved ops still run.
            try:
                work, notes = self._run_op(df.copy(), op, op_params)
            except _OpSkipped as e:
                skipped_stale.append(f"{op['label']} — {e}")
                continue
            except Exception as e:  # noqa: BLE001
                failed.append(f"{op['label']} — {type(e).__name__}: {e}")
                continue
            df = work
            applied.append(op["label"])
            violations.extend(notes)
            if op["id"] in self.ops:
                self.ops[op["id"]]["applied"] = True
            self.applied_ops.append({
                "action": op["action"],
                "label": op["label"].split("\n")[0],
                "columns": list(op["columns"]),
                "extra": _jsonable({k: v for k, v in op["extra"].items()
                                    if k not in ("indices", "_replay")}),
                "params": _jsonable(op_params),
            })

        self.df = df
        lines = [
            f"Cleaned '{self.source}' → {df.shape[0]} rows × {df.shape[1]} cols",
            f"Applied {len(applied)} approved operation(s):",
        ]
        for label in applied:
            lines.append(f"  ✓ {label}")
        if skipped_param:
            lines.append(f"  (skipped {len(skipped_param)} op(s) needing parameters — pass them via params):")
            for label in skipped_param:
                lines.append(f"    - {label}")
        if skipped_stale:
            lines.append(f"  (skipped {len(skipped_stale)} op(s) that could not run):")
            for label in skipped_stale:
                lines.append(f"    - {label}")
        if skipped_done:
            lines.append(f"  (skipped {len(skipped_done)} op(s) already applied earlier):")
            for label in skipped_done:
                lines.append(f"    - {label}")
        if failed:
            lines.append(f"  (FAILED {len(failed)} op(s) — data left unchanged for these):")
            for label in failed:
                lines.append(f"    ✗ {label}")
        if unknown:
            lines.append(f"  (skipped unknown ops: {unknown})")
        skipped = [o["id"] for o in self.ops.values() if o["id"] not in operations]
        if skipped:
            lines.append(f"  (skipped unapproved ops: {', '.join(skipped)})")
        lines.append(f"  duplicates remaining: {int(df.duplicated().sum())}")
        lines.append(f"  null cells remaining: {int(df.isnull().sum().sum())}")
        if violations:
            lines.append("Findings (no rows deleted):")
            lines.extend(f"  ⚠ {v}" for v in violations)
        return "\n".join(lines)

    # ── audit trail / replayable recipe ─────────────────────────────

    def drift_report(self) -> str:
        """What cleaning actually did to the data, column by column."""
        if self.profile_before is None or self.df is None:
            return "BEFORE/AFTER — unavailable (run diagnose() first)."
        changes = compare_profiles(self.profile_before, _profile(self.df))
        if not changes:
            return "BEFORE/AFTER — no measurable change."
        head = ("BEFORE/AFTER — what cleaning changed "
                "(⚠ marks a mean shift ≥ 0.25 of the original standard deviation)")
        return head + "\n" + "\n".join(f"  {line}" for line in changes)

    def manifest(self) -> dict:
        """A record of exactly what was done — auditable, and replayable on a
        later extract of the same dataset via replay()."""
        shape = self.df.shape if self.df is not None else (0, 0)
        return {
            "schema": "swarn.data_cleaner/1",
            "created": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": self.source,
            "target_col": self.target_col,
            "rows_before": (self.shape_before or shape)[0],
            "cols_before": (self.shape_before or shape)[1],
            "rows_after": shape[0],
            "cols_after": shape[1],
            "thresholds": {
                "col_null_drop": COL_NULL_DROP_THRESHOLD,
                "row_null_drop": ROW_NULL_DROP_THRESHOLD,
                "outlier_z": OUTLIER_Z_THRESHOLD,
            },
            "operations": self.applied_ops,
            "profile_before": self.profile_before,
            "profile_after": _profile(self.df) if self.df is not None else None,
        }

    def save_manifest(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.manifest(), fh, indent=2, ensure_ascii=False)
        return path

    def replay(self, manifest) -> str:
        """Re-apply a saved recipe to the currently loaded data.

        Operations are replayed by action + columns, never by op id — op ids are
        positional and would mean something different in a file whose problems
        differ. Row-drop ops re-derive their rows from the recorded rule rather
        than reusing last run's row positions. Nothing is asked or diagnosed:
        this is the unattended, reproducible path.
        """
        if self.df is None:
            return "Error: no data loaded."
        recorded = manifest.get("operations", []) if isinstance(manifest, dict) else list(manifest or [])
        if not recorded:
            return "Error: the recipe contains no operations."
        self.ops, self.applied_ops = {}, []
        self.shape_before = self.df.shape
        self.profile_before = _profile(self.df)
        params = {}
        for i, record in enumerate(recorded, 1):
            op_id = f"r{i}"
            extra = dict(record.get("extra") or {})
            extra["_replay"] = True
            self.ops[op_id] = {
                "id": op_id,
                "action": record.get("action", ""),
                "label": record.get("label") or record.get("action", ""),
                "columns": list(record.get("columns") or []),
                "extra": extra,
                "needs_param": False,
                "recommended": True,
            }
            params[op_id] = dict(record.get("params") or {})
        summary = self.apply_cleaning(list(self.ops), params=params)
        return f"Replayed recipe ({len(recorded)} recorded operation(s))\n{summary}"

    def _run_op(self, df: pd.DataFrame, op: dict, params: dict) -> tuple[pd.DataFrame, list[str]]:
        handler = getattr(self, f"_op_{op['action']}", None)
        if handler is None:
            return df, [f"unknown action {op['action']}"]
        # the plan was built against the pre-clean frame; an earlier approved op
        # may have dropped a column this one targets.
        if op["action"] != "drop_columns":
            gone = [c for c in op["columns"] if c not in df.columns]
            if gone:
                raise _OpSkipped(f"column(s) {gone} were removed by an earlier operation")
        return handler(df, op, params)

    # ── operation implementations (each returns df, notes) ──────────

    def _op_flag_missing(self, df, op, params):
        for col in op["columns"]:
            df[f"{col}_was_missing"] = df[col].isnull().astype(int)
        return df, []

    IMPUTE_STRATEGIES = ("median", "mean", "mode", "most_frequent", "constant",
                         "ffill", "bfill", "interpolate")

    def _op_impute(self, df, op, params):
        """Fill nulls. params:
             strategy=median|mean|mode|constant|ffill|bfill|interpolate
             value=<x>     for strategy=constant
             by=region     (or 'region|store') to fill within groups first
        Group-aware filling matters when the groups differ: one global median
        across regions whose sales differ 10x is a poor estimate for both.
        Order-sensitive strategies (ffill/bfill/interpolate) assume the rows are
        already in a meaningful order — sort before using them.
        """
        by = [c.strip() for c in str(params.get("by", "")).split("|") if c.strip()]
        unknown_by = [c for c in by if c not in df.columns]
        if unknown_by:
            raise _OpSkipped(f"group column(s) {unknown_by} not in the data")
        notes = []
        for col in op["columns"]:
            strategy = str(params.get("strategy", op["extra"].get("strategy", "median"))).lower()
            if strategy not in self.IMPUTE_STRATEGIES:
                raise _OpSkipped(f"unknown strategy '{strategy}' — choose from "
                                 f"{', '.join(self.IMPUTE_STRATEGIES)}")
            s = df[col]
            before = int(s.isnull().sum())

            if strategy in ("ffill", "bfill", "interpolate"):
                if strategy == "interpolate" and not _is_numeric(s):
                    raise _OpSkipped(f"'{col}' is not numeric — interpolate needs numbers")
                if by:
                    grouped = df.groupby(by, dropna=False)[col]
                    if strategy == "interpolate":
                        s = grouped.transform(lambda g: g.interpolate(limit_direction="both"))
                    else:
                        s = grouped.transform(lambda g: getattr(g, strategy)())
                else:
                    s = (s.interpolate(limit_direction="both") if strategy == "interpolate"
                         else getattr(s, strategy)())
                notes.append(f"'{col}': {before - int(s.isnull().sum())}/{before} null(s) "
                             f"filled by {strategy}"
                             + (f" within {by} groups" if by else "")
                             + ("; remainder filled globally" if s.isnull().any() else ""))
            elif by:
                grouped = df.groupby(by, dropna=False)[col]
                if _is_numeric(s) and strategy in ("median", "mean"):
                    s = s.fillna(grouped.transform(strategy))
                elif not _is_numeric(s) and strategy in ("mode", "most_frequent"):
                    s = s.fillna(grouped.transform(
                        lambda g: g.mode().iloc[0] if not g.mode().empty else None))
                notes.append(f"'{col}': {before - int(s.isnull().sum())}/{before} null(s) "
                             f"filled within {by} groups"
                             + ("; remainder filled globally" if s.isnull().any() else ""))

            if s.isnull().any():                       # global fallback
                if strategy == "constant":
                    fill = params.get("value", "Unknown" if not _is_numeric(s) else 0)
                    fill = pd.to_numeric(fill, errors="coerce") if _is_numeric(s) else fill
                elif _is_numeric(s):
                    fill = s.mean() if strategy == "mean" else s.median()
                else:
                    fill = s.mode().iloc[0] if not s.mode().empty else "Unknown"
                s = s.fillna(fill)
            df[col] = s
        return df, notes

    def _op_drop_columns(self, df, op, params):
        return df.drop(columns=[c for c in op["columns"] if c in df.columns]), []

    def _op_drop_rows(self, df, op, params):
        extra = op["extra"]
        if extra.get("_replay") and extra.get("rule"):
            # row positions from the recorded run mean nothing in a new file —
            # re-derive them from the rule that produced them.
            indices = self._rows_matching(df, extra)
        else:
            indices = extra.get("indices") or params.get("indices")
        return df.drop(index=indices, errors="ignore"), []

    def _rows_matching(self, df: pd.DataFrame, extra: dict) -> list:
        rule = extra.get("rule")
        if rule == "target_null":
            col = extra.get("column")
            return df.index[df[col].isnull()].tolist() if col in df.columns else []
        if rule == "row_null_frac":
            threshold = float(extra.get("threshold", ROW_NULL_DROP_THRESHOLD))
            return df.index[df.isnull().mean(axis=1) >= threshold].tolist()
        return []

    def _op_filter_rows(self, df, op, params):
        cols = [c for c in op["columns"] if c in df.columns]
        if not cols:
            return df, []
        col = cols[0]
        s = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if params.get("start"):
            df = df[s >= pd.to_datetime(params["start"], errors="coerce")]
        if params.get("end"):
            df = df[s <= pd.to_datetime(params["end"], errors="coerce")]
        return df, []

    def _op_deduplicate(self, df, op, params):
        keep = params.get("keep", "first")
        return df.drop_duplicates(keep=keep), []

    def _op_trim_text(self, df, op, params):
        for col in op["columns"]:
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        return df, []

    def _op_normalize_case(self, df, op, params):
        target = params.get("case", "lower")
        for col in op["columns"]:
            df[col] = df[col].map(
                lambda x: getattr(x, target)() if isinstance(x, str) else x
            )
        return df, []

    def _op_fix_syntax(self, df, op, params):
        def fix(value):
            if not isinstance(value, str):
                return value
            s = value
            for k, v in _HTML_ENTITIES.items():
                s = s.replace(k, v)
            s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
            s = re.sub(r"[ \t]{2,}", " ", s)
            return s.strip()
        for col in op["columns"]:
            df[col] = df[col].map(fix)
        return df, []

    def _op_split_column(self, df, op, params):
        col = op["columns"][0]
        sep = params.get("sep", ", ")
        names = params.get("into") or [f"{col}_1", f"{col}_2"]
        split = df[col].map(
            lambda x: x.split(sep, 1) if isinstance(x, str) and sep in x else None
        )
        for i, name in enumerate(names):
            df[name] = split.map(lambda parts: parts[i].strip() if parts and i < len(parts) else None)
        df = df.drop(columns=[col])
        return df, []

    def _op_cast_types(self, df, op, params):
        for col in op["columns"]:
            df[col] = pd.to_numeric(
                df[col].map(lambda x: str(x).replace(",", "") if isinstance(x, str) else x),
                errors="coerce",
            )
        return df, []

    def _op_parse_years(self, df, op, params):
        def parse(value):
            if not isinstance(value, str):
                return value
            m = _YEAR_RE.search(value)
            return int(m.group(1)) if m else None
        for col in op["columns"]:
            df[col] = df[col].map(parse)
        return df, []

    def _op_uniform_dates(self, df, op, params):
        """Rewrite date columns to one format — ONE convention per column.

        format="mixed" decides per VALUE, so '07-11-2023' is read month-first
        while '14-07-2023' (14 cannot be a month) is read day-first — two
        conventions inside one column, and roughly a third of rows silently
        wrong. The convention is therefore settled once for the whole column,
        preferring what the column can PROVE about itself, and is reported back
        whenever it had to be assumed.
        """
        fmt = params.get("format")
        notes = []
        for col in op["columns"]:
            series = df[col]
            forced = params.get("dayfirst")
            if forced is not None:
                dayfirst = str(forced).strip().lower() in ("1", "true", "yes", "day", "dayfirst")
                proven = False
            else:
                detected = _detect_dayfirst(series)
                proven = detected is not None
                dayfirst = bool(detected)
            parsed = _parse_uniform(series, dayfirst)
            other = _parse_uniform(series, not dayfirst)
            # A convention that reads strictly fewer rows is the wrong one, even
            # when nothing in the column could prove the layout outright.
            if not proven and forced is None and other.notna().sum() > parsed.notna().sum():
                dayfirst = not dayfirst
                parsed = other
            # Only drop the clock if there is no clock to drop.
            out_fmt = fmt or ("%Y-%m-%d" if _time_is_empty(parsed) else "%Y-%m-%d %H:%M:%S")
            df[col] = parsed.dt.strftime(out_fmt).where(parsed.notna())
            if _dmy_shaped_any(series):
                how = "day-first (DD-MM-YYYY)" if dayfirst else "month-first (MM-DD-YYYY)"
                notes.append(
                    f"'{col}' was read as {how} — "
                    + ("proven by the column itself (it contains values above 12 in that position)."
                       if proven else
                       "ASSUMED: every value in this column is ambiguous, so the layout could not "
                       "be proven. If that is the wrong convention, re-run with "
                       f"params={{'{op['id']}': {{'dayfirst': {not dayfirst}}}}}."))
            unread = int(parsed.isna().sum() - series.isna().sum())
            if unread > 0:
                notes.append(f"'{col}': {unread:,} value(s) could not be read as a date and are now blank.")
        return df, notes

    def _op_convert_booleans(self, df, op, params):
        for col in op["columns"]:
            df[col] = df[col].map(
                lambda x: _BOOL_MAP.get(str(x).strip().lower(), x) if pd.notna(x) else x
            )
        return df, []

    def _op_harmonize_units(self, df, op, params):
        col = op["columns"][0]
        factor = float(params.get("factor", op["extra"].get("factor", 1.0)))
        def convert(x):
            if isinstance(x, str):
                m = re.search(r"[+-]?[\d.,]+", x)
                if m:
                    return float(m.group(0).replace(",", "")) * factor
            return x
        df[col] = df[col].map(convert)
        return df, []

    def _op_cap_outliers(self, df, op, params):
        col = op["columns"][0]
        s = df[col]
        if not _is_numeric(s):
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().mean() < 0.8:
                raise _OpSkipped(f"'{col}' is not numeric — approve its cast op first")
            s = coerced
        bounds = params.get("bounds") or op["extra"].get("bounds")
        if bounds is None:
            lo, hi = s.quantile(0.01), s.quantile(0.99)
        else:
            try:
                lo, hi = (float(b) for b in bounds)
            except (TypeError, ValueError):
                raise _OpSkipped(f"bad bounds {bounds!r} — expected two numbers")
        lo, hi = float(lo), float(hi)
        outside = ((s < lo) | (s > hi)) & s.notna()
        # capping REWRITES real values (a sentinel like 99999 or a genuine whale
        # look identical here), so always leave a trail of which rows were touched.
        mode = str(params.get("mode", "clip")).strip().lower()
        if mode in ("null", "nan", "none", "drop"):
            df[col], verb = s.mask(outside), "nulled"
        else:
            df[col], verb = s.clip(lower=lo, upper=hi), "capped"
        df[f"{col}_was_capped"] = outside.astype(int)
        n = int(outside.sum())
        note = (f"'{col}': {n} value(s) {verb} to [{lo:.6g}, {hi:.6g}] — original values "
                f"replaced, rows marked in '{col}_was_capped'"
                + ("" if mode.startswith(("null", "nan", "none", "drop"))
                   else "; pass mode=null to blank them instead"))
        return df, [note]

    def _op_fuzzy_merge(self, df, op, params):
        """Merge labels that mean the same thing but are spelled differently.

        Two passes, both refusing to merge labels whose digits differ:
          1. entity-suffix folding ('Acme Corporation' → 'Acme Corp') — always
          2. similarity, only when an explicit threshold is supplied — typos
        The most common spelling always wins, so counts move to the majority form.
        """
        threshold = float(params.get("threshold", 0) or 0)
        notes = []
        for col in op["columns"]:
            strings = [v for v in df[col].dropna() if isinstance(v, str)]
            if not strings:
                continue
            counts = pd.Series(strings).value_counts()   # most common first
            labels = list(counts.index)
            if len(labels) > FUZZY_MAX_LABELS:
                notes.append(f"'{col}': {len(labels)} distinct labels exceeds the "
                             f"{FUZZY_MAX_LABELS} limit — skipped, not silently truncated")
                continue
            mapping: dict = {}
            grouped: dict[str, list] = {}
            for v in labels:
                grouped.setdefault(_entity_key(v), []).append(v)
            for variants in grouped.values():
                if len(variants) > 1 and len({_digit_signature(v) for v in variants}) == 1:
                    mapping.update({v: variants[0] for v in variants[1:]})
            if threshold > 0:
                remaining = [v for v in labels if v not in mapping]
                for i, a in enumerate(remaining):
                    if a in mapping:
                        continue
                    for b in remaining[i + 1:]:
                        if b in mapping or _digit_signature(a) != _digit_signature(b):
                            continue
                        ka, kb = _entity_key(a), _entity_key(b)
                        if ka and kb and ka[0] == kb[0] and \
                                SequenceMatcher(None, ka, kb).ratio() >= threshold:
                            mapping[b] = a
            if mapping:
                df[col] = df[col].map(lambda x: mapping.get(x, x) if isinstance(x, str) else x)
                sample = "; ".join(f"'{k}' → '{v}'" for k, v in list(mapping.items())[:3])
                notes.append(f"'{col}': merged {len(mapping)} near-identical label(s) — {sample}")
        return df, notes

    def _op_standardize_categories(self, df, op, params):
        notes = []
        for col in op["columns"]:
            strings = [v for v in df[col].dropna() if isinstance(v, str)]
            groups: dict[str, list] = {}
            for v in strings:
                groups.setdefault(_canonical_label(v), []).append(v)
            mapping = {}
            for variants in groups.values():
                counts = pd.Series(variants).value_counts()
                if len(counts) > 1:
                    winner = counts.index[0]      # most common spelling wins
                    mapping.update({v: winner for v in counts.index if v != winner})
            if mapping:
                df[col] = df[col].map(lambda x: mapping.get(x, x) if isinstance(x, str) else x)
                sample = "; ".join(f"'{k}' → '{v}'" for k, v in list(mapping.items())[:3])
                notes.append(f"'{col}': merged {len(mapping)} spelling variant(s) — {sample}")
        return df, notes

    def _op_cross_field(self, df, op, params):
        a, b = op["columns"][0], op["columns"][1]
        rule = params.get("rule", op["extra"].get("rule", f"{b} >= {a}"))
        da, db = df[a], df[b]
        try:
            pa, pb = (pd.to_datetime(da, errors="coerce", format="mixed"),
                      pd.to_datetime(db, errors="coerce", format="mixed"))
            if pa.notna().mean() >= 0.9 and pb.notna().mean() >= 0.9:
                da, db = pa, pb
        except (TypeError, ValueError):
            pass
        if ">=" in rule:
            violation = df[db < da]
        else:
            violation = df[db > da]
        flag = f"{b}_vs_{a}_valid"
        df[flag] = 1
        df.loc[violation.index, flag] = 0
        note = f"'{rule}': {len(violation)} violating row(s) flagged in '{flag}'"
        return df, [note]

    def _op_rule_validate(self, df, op, params):
        col = op["columns"][0]
        pattern = params.get("pattern", op["extra"].get("pattern", ""))
        if not pattern:
            return df, [f"'rule_validate' for {col}: no pattern given — skipped"]
        rx = re.compile(pattern)
        def ok(x):
            if pd.isna(x):
                return None
            return 1 if rx.fullmatch(str(x).strip()) else 0
        flag = f"{col}_valid"
        df[flag] = df[col].map(ok)
        n_bad = int((df[flag] == 0).sum())
        note = f"'{col}': {n_bad} value(s) fail the pattern (flagged in '{flag}')"
        return df, [note]


# ───────────────────────────────────────────────────────────────────────────────

class DataAnalyzer:
    """Analyst-facing summaries and aggregations for a cleaned dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def describe(self) -> str:
        df = self.df
        lines = [f"DATA SUMMARY — {df.shape[0]} rows × {df.shape[1]} cols", ""]
        for col in df.columns:
            s = df[col]
            nn = s.dropna()
            nulls = int(s.isnull().sum())
            uniq = int(s.nunique())
            lines.append(f"■ {col}  ({s.dtype})  non-null={len(nn)}  nulls={nulls}  unique={uniq}")
            if _is_numeric(s) and len(nn):
                q = nn.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
                lines.append(
                    f"    min={q[0]:.3g}  q1={q[1]:.3g}  median={q[2]:.3g}  "
                    f"q3={q[3]:.3g}  max={q[4]:.3g}  mean={nn.mean():.3g}"
                )
            elif len(nn) and uniq <= 15:
                top = nn.value_counts().head(5)
                lines.append("    top: " + ", ".join(f"{k} ({v})" for k, v in top.items()))
        return "\n".join(lines)

    def group(self, keys: list[str], aggregations: dict[str, str],
              output_name: str = "") -> pd.DataFrame:
        agg: dict[str, str] = {}
        unknown_aggs = []
        for col, how in (aggregations or {}).items():
            if col in self.df.columns:
                agg[col] = how
            else:
                unknown_aggs.append(col)
        missing_keys = [k for k in keys if k not in self.df.columns]
        if missing_keys:
            raise ValueError(f"Group keys not in data: {missing_keys}")
        if unknown_aggs:
            # a typo here used to be swallowed, returning bare distinct keys
            raise ValueError(f"Columns to aggregate not in data: {unknown_aggs}. "
                             f"Available: {list(self.df.columns)}")
        if not agg:
            result = self.df[keys].drop_duplicates().reset_index(drop=True)
        else:
            grouped = self.df.groupby(keys, dropna=False)
            result = grouped.agg(agg).reset_index()
            # An average is meaningless without knowing how many rows it rests
            # on — a 1-row "yearly average" reads exactly like a 1,600-row one.
            # Always attach the count rather than hoping the caller asks.
            count_col = "n_rows"
            while count_col in result.columns:
                count_col += "_"
            result[count_col] = grouped.size().reset_index(drop=True)
        return result


# ─── Swarn tool registration (lazy — only touches agent.tools) ─────────────────

def _swarn_clean_dataset(name: str, target_col: Optional[str] = None) -> str:
    from agent.ml.data_pipeline import get_data_pipeline
    pipe = get_data_pipeline()
    df = pipe.datasets.get(name)
    if df is None:
        return f"Error: no dataset named '{name}' is loaded."
    cleaner = DataCleaner(df.copy(), source=name, target_col=target_col)
    plan = cleaner.diagnose()
    pipe._cleaner_cache[name] = cleaner
    return plan


def _swarn_apply_cleaning(name: str, operations: list, params: Optional[dict] = None,
                          output_name: Optional[str] = None,
                          target_col: Optional[str] = None) -> str:
    from agent.ml.data_pipeline import get_data_pipeline
    pipe = get_data_pipeline()
    df = pipe.datasets.get(name)
    if df is None:
        return f"Error: no dataset named '{name}' is loaded."
    cleaner = pipe._cleaner_cache.get(name)
    if cleaner is None or not cleaner.ops:
        cleaner = DataCleaner(df.copy(), source=name, target_col=target_col)
        cleaner.diagnose()
        pipe._cleaner_cache[name] = cleaner
    answer = ask_human(
        f"HUMAN APPROVAL REQUIRED — apply cleaning operations to '{name}'?\n\n"
        f"{cleaner.plan}\n\n"
        "Reply 'all' | 'none' | op ids ('op1 op3 op5') | ops with params "
        "('op4: factor=83.2')."
    )
    approved, approved_params = parse_approval(answer, cleaner.op_ids(),
                                               cleaner.optional_op_ids())
    if not approved:
        return (f"No operations approved for '{name}' — nothing was applied. "
                f"Original data kept as-is.")
    params = {**(params or {}), **approved_params}
    summary = cleaner.apply_cleaning(approved, params)
    out = output_name or f"{name}_clean"
    pipe.datasets[out] = cleaner.df
    recipe = json.dumps(cleaner.manifest()["operations"], ensure_ascii=False)
    return (f"{summary}\nRegistered result as '{out}'. Original '{name}' untouched.\n"
            f"IMPORTANT: '{out}' lives in memory ONLY — it is NOT a file. Refer to it by name "
            f"in describe_dataset / analyze_dataset / plot_* / pivot_dataset. Do NOT "
            f"pd.read_csv('{out}.csv') — any file with that name is a different, probably stale "
            f"copy. If you genuinely need it on disk, call save_dataset('{out}', '{out}.csv') first.\n"
            f"Cleaning record (replayable on a later extract): {recipe}")


def _swarn_describe_dataset(name: str) -> str:
    from agent.ml.data_pipeline import get_data_pipeline
    df = get_data_pipeline().datasets.get(name)
    if df is None:
        return f"Error: no dataset named '{name}' is loaded."
    return DataAnalyzer(df).describe()


def _swarn_group_dataset(name: str, keys: list, aggregations: dict,
                         output_name: Optional[str] = None) -> str:
    from agent.ml.data_pipeline import get_data_pipeline
    pipe = get_data_pipeline()
    df = pipe.datasets.get(name)
    if df is None:
        return f"Error: no dataset named '{name}' is loaded."
    try:
        result = DataAnalyzer(df).group(keys, aggregations)
    except ValueError as e:
        return f"Error: {e}"
    out = output_name or f"{name}_grouped"
    pipe.datasets[out] = result
    head = result.head(20).to_string(index=False)
    lines = [f"Grouped '{name}' by {keys} → registered as '{out}' ({len(result)} rows)", head]
    lines.extend(_group_imputation_warnings(df, keys, aggregations or {}))
    return "\n".join(lines)


def _group_imputation_warnings(df, keys: list, aggregations: dict) -> list:
    """Warn when a group's average is built from values the cleaner invented.

    Without this, a year whose every vote count was filled in with the column
    median shows up as a clean data point and gets reported as a trend.
    """
    notes = []
    for col in aggregations:
        marker = f"{col}_was_missing"
        if col not in df.columns or marker not in df.columns:
            continue
        share = df.groupby(keys, dropna=False)[marker].mean()
        fully = share[share >= 0.999]
        mostly = share[(share >= 0.5) & (share < 0.999)]
        if not len(fully) and not len(mostly):
            continue
        if len(fully):
            try:    # remember it, so a trend claim resting on these gets caught
                from agent.data_analysis import note_imputed_groups
                note_imputed_groups(col, list(fully.index))
            except Exception:  # noqa: BLE001
                pass
        note = f"⚠ '{col}' contains values that cleaning invented: "
        if len(fully):
            examples = ", ".join(str(k) for k in list(fully.index)[:6])
            note += (f"{len(fully)} group(s) are 100% imputed ({examples}"
                     f"{'…' if len(fully) > 6 else ''}) — their figure IS the fill value, "
                     f"not a measurement. Do not report those as findings. ")
        if len(mostly):
            note += f"{len(mostly)} further group(s) are over half imputed. "
        note += f"Check '{marker}' before drawing conclusions."
        notes.append(note)
    return notes


def _swarn_ask_human(question: str, options: Optional[list] = None) -> str:
    return ask_human(question, options=options)


_SWARN_TOOLS = {
    "clean_dataset": (
        "Run a full human-in-the-loop data-quality diagnosis on a loaded dataset. "
        "Returns a numbered CLEANING PLAN covering missing values, redundancy, structural "
        "errors, types/formats, and anomalies. Changes NOTHING. After loading any dataset, "
        "ALWAYS call this so the plan exists, then call apply_cleaning — it will ask the "
        "human for approval and apply only the approved ops. Do NOT call ask_human for "
        "cleaning approval; apply_cleaning does that itself. If a target/label column will "
        "be used for training, pass it as target_col so the plan drops rows with a missing "
        "target instead of imputing it.",
        {"type": "object",
         "properties": {
             "name": {"type": "string", "description": "Name of a loaded dataset."},
             "target_col": {"type": "string", "description": "Optional target/label column (nulls in it are dropped, not imputed)."},
         },
         "required": ["name"]},
        _swarn_clean_dataset,
    ),
    "apply_cleaning": (
        "Apply cleaning operations to a dataset (the plan from clean_dataset). IMPORTANT: "
        "this tool BLOCKS and asks the human to approve the plan — 'all', 'none', 'op1 op3 "
        "op5', or 'op4: factor=83.2' — and WAITS for their answer before changing anything. "
        "It applies ONLY the ops the human approves. Registers the result under "
        "'<name>_clean'; the original is never modified. Cross-field/rule ops only add "
        "validation flag columns, never delete.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "operations": {"type": "array", "items": {"type": "string"},
                            "description": "Proposed op ids from the cleaning plan, e.g. ['op1','op3']. The human's approval reply overrides this."},
             "params": {"type": "object",
                        "description": "Optional per-op parameters keyed by op id, e.g. {'op4': {'factor': 83.2}}."},
             "output_name": {"type": "string", "description": "Result dataset name (default '<name>_clean')."},
             "target_col": {"type": "string", "description": "Optional target column (must match clean_dataset's target_col)."},
         },
         "required": ["name", "operations"]},
        _swarn_apply_cleaning,
    ),
    "ask_human": (
        "Ask the human for approval/input and return their answer. Use this BEFORE any "
        "destructive step (deleting rows/columns, dedup, imputation, etc.) — the project "
        "requires human-in-the-loop for data changes. Call this tool ALONE and wait for the "
        "answer; do not batch it with other tool calls.",
        {"type": "object",
         "properties": {
             "question": {"type": "string", "description": "The question to ask, with any evidence (counts, sample values)."},
             "options": {"type": "array", "items": {"type": "string"}, "description": "Optional answer choices."},
         },
         "required": ["question"]},
        _swarn_ask_human,
    ),
    "describe_dataset": (
        "Analyst-style summary of a loaded dataset: per-column dtype, non-null/null counts, "
        "cardinality, and for numeric columns min/q1/median/q3/max/mean (or top values for "
        "low-cardinality categoricals).",
        {"type": "object",
         "properties": {"name": {"type": "string"}},
         "required": ["name"]},
        _swarn_describe_dataset,
    ),
    "group_dataset": (
        "SQL-style GROUP BY on a loaded dataset (e.g. keys=['region','month'], "
        "aggregations={'sales':'sum','orders':'count'}). Registers the result as "
        "'<name>_grouped' and returns a preview.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "keys": {"type": "array", "items": {"type": "string"}, "description": "Columns to group by."},
             "aggregations": {"type": "object", "description": "Mapping column → agg function (sum/mean/count/min/max)."},
             "output_name": {"type": "string"},
         },
         "required": ["name", "keys", "aggregations"]},
        _swarn_group_dataset,
    ),
}


def register_into_swarn() -> str:
    """Register the 5 data-cleaning/analysis tools into agent.tools.TOOL_REGISTRY."""
    try:
        from agent.runtime.tools import TOOL_REGISTRY
    except Exception as e:  # noqa: BLE001
        return f"swarn registration skipped: {e}"
    registered = 0
    for name, (desc, schema, fn) in _SWARN_TOOLS.items():
        if name in TOOL_REGISTRY:
            continue
        TOOL_REGISTRY[name] = {"description": desc, "schema": schema, "func": fn}
        registered += 1
    return f"registered {registered} data-cleaning tool(s) into swarn"


# ─── standalone CLI ────────────────────────────────────────────────────────────

def _default_output(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_clean{ext or '.csv'}"


def _parse_group_spec(spec: str) -> tuple[list[str], dict[str, str]]:
    keys_part, _, aggs_part = spec.partition(":")
    keys = [k.strip() for k in keys_part.split(",") if k.strip()]
    aggs: dict[str, str] = {}
    for tok in aggs_part.split(","):
        m = re.match(r"\s*(\w+)\s*\(\s*([^()]+?)\s*\)\s*$", tok)
        if m:
            aggs[m.group(2).strip()] = m.group(1).lower()
    return keys, aggs


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="data_cleaner",
        description="Human-in-the-loop data cleaning + analysis (standalone). "
                    "Loads CSV/XLSX/PARQUET, proposes cleaning operations, asks for "
                    "approval, applies only approved ops, saves '<name>_clean.*', and "
                    "prints a data summary.",
    )
    parser.add_argument("path", help="Path to a CSV/XLSX/PARQUET file.")
    parser.add_argument("-o", "--output", help="Output path (default '<name>_clean.<ext>').")
    parser.add_argument("--target-col", default=None,
                        help="Target/label column: rows with a missing value in it are dropped, not imputed.")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Non-interactive: approve all recommended ops (no prompts).")
    parser.add_argument("--approvals", default=None,
                        help="Explicit approval, e.g. 'all', 'op1 op3 op5', or 'none' (skips prompting).")
    parser.add_argument("--params", default=None,
                        help='JSON params keyed by op id, e.g. \'{"op4":{"factor":83.2}}\'.')
    parser.add_argument("--describe", action="store_true", default=True,
                        help="Print the data summary after cleaning (default on).")
    parser.add_argument("--group", action="append", default=[],
                        help="GROUP BY spec, e.g. 'region:sum(sales),mean(price)'. Repeatable.")
    parser.add_argument("--log", default=None,
                        help="Where to write the cleaning record (default '<output>.cleaning.json').")
    parser.add_argument("--no-log", action="store_true",
                        help="Do not write a cleaning record.")
    parser.add_argument("--replay", default=None,
                        help="Re-apply a saved cleaning record (from --log) instead of "
                             "diagnosing and prompting. Unattended and reproducible.")
    parser.add_argument("--no-drift", action="store_true",
                        help="Skip the before/after comparison of what cleaning changed.")
    args = parser.parse_args(argv)

    cleaner = DataCleaner(target_col=args.target_col)
    print(cleaner.load_file(args.path))

    if args.replay:
        try:
            with open(args.replay, encoding="utf-8") as fh:
                recipe = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error: cannot read recipe '{args.replay}': {e}")
            return 2
        print("\n" + cleaner.replay(recipe))
        if not args.no_drift:
            print("\n" + cleaner.drift_report())
        out = args.output or _default_output(args.path)
        cleaner.save_file(out)
        print(f"\nSaved cleaned data → {out}")
        if not args.no_log:
            print(f"Cleaning record → {cleaner.save_manifest(args.log or out + '.cleaning.json')}")
        if args.describe:
            print("\n" + DataAnalyzer(cleaner.df).describe())
        return 0

    plan = cleaner.diagnose()
    print("\n" + plan + "\n")
    if plan.startswith("Error:"):
        return 2

    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"Error: --params is not valid JSON: {e}")
            return 2
    else:
        params = {}

    if args.approvals is not None:
        approved, approved_params = parse_approval(args.approvals, cleaner.op_ids(),
                                                  cleaner.optional_op_ids())
        params.update(approved_params)
    elif args.auto_approve or os.environ.get("SWARN_AUTO_APPROVE", "").strip().lower() in ("1", "true", "yes"):
        recommended = set(cleaner.op_ids()) - set(cleaner.optional_op_ids())
        approved = [o for o in cleaner.op_ids() if o in recommended]
    else:
        answer = ask_human(
            "Approve cleaning operations?  ('all', 'none', 'op1 op3 op5', or 'op4: factor=83.2')"
        )
        approved, approved_params = parse_approval(answer, cleaner.op_ids(),
                                               cleaner.optional_op_ids())
        params.update(approved_params)

    print(cleaner.apply_cleaning(approved, params=params))

    if not approved:
        print("No operations approved — original data kept as-is.")

    if not args.no_drift:
        print("\n" + cleaner.drift_report())

    out = args.output or _default_output(args.path)
    cleaner.save_file(out)
    print(f"\nSaved cleaned data → {out}")

    if not args.no_log:
        log_path = cleaner.save_manifest(args.log or out + ".cleaning.json")
        print(f"Cleaning record → {log_path}  (re-run with --replay {log_path})")

    if args.describe:
        print("\n" + DataAnalyzer(cleaner.df).describe())

    for spec in args.group:
        keys, aggs = _parse_group_spec(spec)
        if not keys:
            print(f"\n(skipped invalid group spec: {spec})")
            continue
        try:
            result = DataAnalyzer(cleaner.df).group(keys, aggs)
        except ValueError as e:
            print(f"\n(skip: {e})")
            continue
        print(f"\nGROUP BY {keys} → {len(result)} rows\n{result.head(20).to_string(index=False)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
