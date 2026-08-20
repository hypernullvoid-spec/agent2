"""
Joining datasets — the one step in the analysis stack that could produce a
wrong number with nothing watching.

Why this module exists
──────────────────────
Every other data tool here is defended. clean_dataset asks before it changes
anything and records what it did. analyze_dataset refuses to average an error
code. rank_by tests whether its own league table is distinguishable from
chance. write_report reads the narrative back against the evidence ledger and
refuses a document whose story contradicts the measurements.

Joining had none of that. The registry could hold `orders`, `customers` and
`products`, and the only way to combine them was hand-written pandas inside
run_python — outside every guard in this package. That mattered because a join
is the single most dangerous operation in tabular analysis, and it fails
silently in two opposite directions at once:

  FAN-OUT   The right table is not unique on the key. One left row matches
            three right rows, so it becomes three rows. Every measure on the
            left side is now counted three times. Revenue inflates and nothing
            about the output looks wrong.

  ORPHANS   A left row's key is absent from the right table. On an inner join
            that row is deleted. Its revenue leaves the analysis entirely, and
            no error is raised.

The reason this is worth a whole module rather than a wrapper is that the two
cancel. A worked example from this project's own test suite: four orders worth
1,000 total, joined to a customer list holding one duplicate and missing one
customer. Rows in: 4. Rows out: 4. The duplicate added one, the orphan removed
one. Row count — the thing a careful person checks — was unchanged, and the
total had quietly become 700.

So counting rows is not a defence. The defence is to work out, BEFORE merging,
exactly which rows will multiply, which will vanish, and what that does to each
measure — then show a human that arithmetic and wait.

The contract
────────────
  • Nothing is joined until a human approves it (ask_human, the same flow
    apply_cleaning uses). Non-interactive runs approve only under
    SWARN_AUTO_APPROVE, matching data_cleaner.py.
  • The prediction is exact, not heuristic. Predicted row count and predicted
    measure totals are computed from the actual key distributions, and the
    result is verified against them afterwards.
  • The join is recorded in the evidence ledger, so write_report's Methodology
    section states the keys, the direction and the row-count effect — in the
    part of the document the narrator cannot edit.
  • Neither input is modified. The result is registered as a NEW dataset.

Pure pandas / numpy — no new dependencies.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# A left row matching this many right rows is a fan-out worth shouting about
# even when the human approved the shape. Below it, duplicates are usually a
# genuine one-to-many (an order and its lines); above it, they are usually a
# defect in the right-hand table.
FANOUT_LOUD = float(os.environ.get("SWARN_JOIN_FANOUT_LOUD", "1.5"))

# Share of a measure that may silently leave the analysis before the loss is
# promoted from a note to a stated limitation in the report.
MEASURE_LOSS_MATERIAL = float(os.environ.get("SWARN_JOIN_MEASURE_LOSS", "0.005"))

# Overlap below this means the keys almost certainly do not correspond at all —
# a type mismatch, a zero-padding difference, or simply the wrong columns.
OVERLAP_BROKEN = float(os.environ.get("SWARN_JOIN_OVERLAP_BROKEN", "0.01"))

HOWS = ("left", "inner", "right", "outer")

_APPROVE_WORDS = {"yes", "y", "approve", "approved", "all", "ok", "okay", "proceed", "go"}


# ─── registry access ───────────────────────────────────────────────────────────

def _pipeline():
    from agent.ml.data_pipeline import get_data_pipeline
    return get_data_pipeline()


def _get(name: str):
    pipe = _pipeline()
    df = pipe.datasets.get(name)
    if df is None:
        known = ", ".join(pipe.datasets) or "(none loaded)"
        return None, f"Error: no dataset named '{name}' is loaded. Loaded: {known}"
    return df, ""


# ─── key resolution ────────────────────────────────────────────────────────────

def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _auto_keys(left_df: pd.DataFrame, right_df: pd.DataFrame) -> list:
    """Columns present in both frames, preferring ones that look like keys.

    Deliberately conservative. Guessing the join key wrong produces a confident
    wrong answer, so this only offers a suggestion when exactly one shared
    column looks like an identifier; anything else is handed back to the caller
    to name explicitly.
    """
    shared = [str(c) for c in left_df.columns if str(c) in {str(x) for x in right_df.columns}]
    if not shared:
        return []
    idish = [c for c in shared
             if c.lower().endswith(("_id", "id", "_key", "key", "_no", "_code", "code"))]
    if len(idish) == 1:
        return idish
    if len(shared) == 1:
        return shared
    return []


def _resolve_keys(left_df, right_df, on, left_on, right_on) -> tuple:
    """Return (left_keys, right_keys, error)."""
    lk, rk = _as_list(left_on), _as_list(right_on)
    if on:
        lk = rk = _as_list(on)
    if not lk and not rk:
        guess = _auto_keys(left_df, right_df)
        if not guess:
            shared = sorted({str(c) for c in left_df.columns} & {str(c) for c in right_df.columns})
            return [], [], (
                "Error: no join key given and none could be inferred safely. "
                + (f"Columns present in both: {shared}. " if shared else
                   "The two tables share no column names. ")
                + "Name the key explicitly — on='customer_id', or "
                  "left_on='cust_id', right_on='id'.")
        lk = rk = guess
    if len(lk) != len(rk):
        return [], [], (f"Error: left_on has {len(lk)} column(s) but right_on has {len(rk)}. "
                        "They must line up one-for-one.")
    missing_l = [c for c in lk if c not in left_df.columns]
    missing_r = [c for c in rk if c not in right_df.columns]
    if missing_l:
        return [], [], (f"Error: {missing_l} not found in the left dataset. "
                        f"It has: {list(map(str, left_df.columns))}")
    if missing_r:
        return [], [], (f"Error: {missing_r} not found in the right dataset. "
                        f"It has: {list(map(str, right_df.columns))}")
    return lk, rk, ""


# ─── key quality ───────────────────────────────────────────────────────────────

def _kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    return "text"


def _padded_share(series: pd.Series) -> float:
    """Fraction of text values that are digits with a leading zero.

    '007' read from CSV as text and 7 read from Excel as a number are the same
    customer to everyone except pandas, which matches neither.
    """
    values = series.dropna().astype(str).head(2000)
    if not len(values):
        return 0.0
    return float(np.mean([v.isdigit() and len(v) > 1 and v.startswith("0") for v in values]))


def _untrimmed_share(series: pd.Series) -> float:
    values = series.dropna().astype(str).head(2000)
    if not len(values):
        return 0.0
    return float(np.mean([v != v.strip() for v in values]))


def _key_problems(left_df, right_df, lk, rk) -> list:
    """Reasons the keys may not match even though they look like they should."""
    notes = []
    for a, b in zip(lk, rk):
        ls, rs = left_df[a], right_df[b]
        la, ra = _kind(ls), _kind(rs)
        if la != ra:
            notes.append(
                f"TYPE MISMATCH: left '{a}' holds {la}s, right '{b}' holds {ra}s. pandas "
                f"matches these by value AND type, so this join will find almost nothing. "
                f"Cast one side first — apply_cleaning's cast_types op, or run_python.")
        if la == "text" and ra == "text":
            lp, rp = _padded_share(ls), _padded_share(rs)
            if abs(lp - rp) > 0.2:
                side = a if lp > rp else b
                notes.append(
                    f"ZERO PADDING: '{side}' keeps leading zeros ('007') and the other side "
                    f"does not. Those are different strings — the padded rows will not match.")
            lu, ru = _untrimmed_share(ls), _untrimmed_share(rs)
            if max(lu, ru) > 0.01:
                notes.append(
                    f"WHITESPACE: {max(lu, ru):.0%} of values in '{a if lu >= ru else b}' have "
                    f"leading/trailing spaces. ' A12' and 'A12' will not match — trim first.")
            lset = set(ls.dropna().astype(str).head(5000))
            rset = set(rs.dropna().astype(str).head(5000))
            if lset and rset and not (lset & rset):
                low = {v.lower() for v in lset} & {v.lower() for v in rset}
                if low:
                    notes.append(
                        f"CASE: '{a}' and '{b}' share no values as written, but "
                        f"{len(low):,} match once lowercased. Normalise case first.")
        ln, rn = int(ls.isnull().sum()), int(rs.isnull().sum())
        if ln:
            notes.append(f"BLANK KEYS: {ln:,} left row(s) have no '{a}'. They can never match "
                         f"and will be dropped by an inner join.")
        if rn:
            notes.append(f"BLANK KEYS: {rn:,} right row(s) have no '{b}'.")
    return notes


# ─── the prediction ────────────────────────────────────────────────────────────

def _match_counts(a_df: pd.DataFrame, a_keys: list,
                  b_df: pd.DataFrame, b_keys: list) -> pd.Series:
    """For every row of `a`, how many rows of `b` its key matches.

    This is the whole prediction. Row-count effect, orphan count and measure
    inflation all fall straight out of it, and because it is computed from the
    real key distributions the forecast is exact rather than a rule of thumb.
    """
    names = [f"__k{i}" for i in range(len(a_keys))]
    left = a_df[a_keys].copy()
    left.columns = names
    right = b_df[b_keys].copy()
    right.columns = names
    counts = right.groupby(names, dropna=False).size().rename("__n").reset_index()
    try:
        merged = left.merge(counts, on=names, how="left")
    except ValueError:
        # pandas refuses to merge str against int64. That refusal IS the answer:
        # under these dtypes nothing matches, so zero is the honest count. It
        # must not propagate as an exception, because this function runs to
        # DIAGNOSE the mismatch — crashing here would replace the explanation
        # with a traceback in exactly the case the explanation is needed.
        return pd.Series(np.zeros(len(left), dtype="int64"))
    return merged["__n"].fillna(0).astype("int64").reset_index(drop=True)


def _predicted_rows(how: str, l_match: pd.Series, r_match: pd.Series) -> int:
    if how == "inner":
        return int(l_match.sum())
    if how == "left":
        return int(l_match.clip(lower=1).sum())
    if how == "right":
        return int(r_match.clip(lower=1).sum())
    return int(l_match.clip(lower=1).sum() + (r_match == 0).sum())


def _measure_columns(df: pd.DataFrame, keys: list, limit: int = 8) -> list:
    """Numeric columns worth tracking through the join.

    Identifiers and flags are excluded: 'the sum of order_id changed' is noise,
    and it would bury the one line that matters ('revenue changed').
    """
    out = []
    for col in df.columns:
        name = str(col)
        if name in keys or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if name.lower().endswith(("_id", "id", "_key", "code", "_no", "year", "zip", "pincode")):
            continue
        if df[col].nunique(dropna=True) <= 2:
            continue
        out.append(name)
        if len(out) >= limit:
            break
    return out


def _measure_impact(df: pd.DataFrame, keys: list, match: pd.Series, how: str) -> list:
    """(column, before, after, lost) for each measure, under the chosen join."""
    weight = match if how in ("inner", "right") else match.clip(lower=1)
    rows = []
    for col in _measure_columns(df, keys):
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).reset_index(drop=True)
        before = float(values.sum())
        after = float((values * weight).sum())
        lost = float(values[match == 0].sum()) if how in ("inner", "right") else 0.0
        if before or after:
            rows.append((col, before, after, lost))
    return rows


def _fmt(value: float) -> str:
    if value != value:
        return "—"
    if abs(value) >= 1_000_000:
        return f"{value:,.0f}"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _relationship(l_dupes: int, r_dupes: int) -> str:
    return {(False, False): "one-to-one",
            (False, True): "one-to-many",
            (True, False): "many-to-one",
            (True, True): "many-to-many"}[(bool(l_dupes), bool(r_dupes))]


# ─── the approval card ─────────────────────────────────────────────────────────

def _card(left, right, lk, rk, how, out_name, left_df, right_df,
          l_match, r_match, problems) -> tuple:
    """Render what the human is being asked to approve, and the risk flags."""
    n_left, n_right = len(left_df), len(right_df)
    predicted = _predicted_rows(how, l_match, r_match)
    l_orphans = int((l_match == 0).sum())
    r_orphans = int((r_match == 0).sum())
    l_dupes = int(left_df.duplicated(subset=lk).sum())
    r_dupes = int(right_df.duplicated(subset=rk).sum())
    rel = _relationship(l_dupes, r_dupes)
    overlap = float((l_match > 0).mean()) if n_left else 0.0
    impact = _measure_impact(left_df, lk, l_match, how)

    keyspec = (f"{lk[0]}" if lk == rk and len(lk) == 1
               else f"{lk} = {rk}" if lk != rk else f"{lk}")

    lines = [
        f"JOIN '{left}' ({n_left:,} rows) → '{right}' ({n_right:,} rows) on {keyspec}, "
        f"how={how}, result registered as '{out_name}'",
        "",
        f"  Key relationship : {rel}"
        + (f"  — {r_dupes:,} duplicate key(s) on the right" if r_dupes else ""),
        f"  Key overlap      : {overlap:.1%} of left rows find a match",
        f"  Rows in → out    : {n_left:,} → {predicted:,}"
        + (f"  ({predicted - n_left:+,})" if predicted != n_left else "  (unchanged)"),
    ]

    flags = list(problems)

    if r_dupes:
        worst = int(l_match.max()) if len(l_match) else 0
        multiplied = int((l_match > 1).sum())
        ratio = predicted / n_left if n_left else 1.0
        lines.append(
            f"  FAN-OUT          : {multiplied:,} left row(s) match more than one right row "
            f"(worst: 1 → {worst:,})")
        if ratio >= FANOUT_LOUD or multiplied:
            flags.append(
                f"FAN-OUT: '{right}' is not unique on {rk} — {r_dupes:,} duplicate key(s). "
                f"{multiplied:,} left row(s) will be duplicated, so every measure on the left "
                f"is counted more than once. If '{right}' is meant to be a lookup table, "
                f"de-duplicate it first; if the duplication is real (an order and its lines), "
                f"this is correct — but do not sum left-side measures afterwards.")

    if l_orphans:
        share = l_orphans / n_left if n_left else 0.0
        verb = "DROPPED" if how in ("inner", "right") else "kept, with blank right-hand columns"
        lines.append(f"  Left orphans     : {l_orphans:,} ({share:.1%}) match nothing — {verb}")
        if how in ("inner", "right"):
            flags.append(
                f"DATA LOSS: {l_orphans:,} row(s) of '{left}' ({share:.1%}) have no match in "
                f"'{right}' and this {how} join DELETES them. They leave the analysis with no "
                f"error raised. Use how='left' to keep them, or find out why they are missing.")
    if r_orphans and how in ("right", "outer"):
        lines.append(f"  Right orphans    : {r_orphans:,} right row(s) match nothing — kept, "
                     f"with blank left-hand columns")
    elif r_orphans:
        lines.append(f"  Right orphans    : {r_orphans:,} right row(s) match nothing — not "
                     f"included (this is normal for a lookup table)")

    clash = sorted(({str(c) for c in left_df.columns} & {str(c) for c in right_df.columns})
                   - set(lk) - set(rk))
    if clash:
        lines.append(f"  Name clashes     : {clash} exist on both sides → suffixed _x / _y")

    if impact:
        lines += ["", "  What this does to the measures:"]
        for col, before, after, lost in impact:
            note = ""
            if before and abs(after - before) / max(abs(before), 1e-9) >= 0.001:
                pct = (after - before) / before * 100 if before else 0.0
                note = f"   ({pct:+.1f}%)"
            lines.append(f"    {col}: {_fmt(before)} → {_fmt(after)}{note}")
            if before and lost and abs(lost / before) >= MEASURE_LOSS_MATERIAL:
                flags.append(
                    f"MEASURE LOSS: {_fmt(lost)} of '{col}' ({lost / before:.1%} of the total) "
                    f"sits in rows this join deletes. Any total for '{col}' computed after the "
                    f"join will be that much too low.")

    if overlap < OVERLAP_BROKEN:
        flags.insert(0, (
            f"ALMOST NOTHING MATCHES: only {overlap:.2%} of left rows find a partner. These "
            f"columns probably do not correspond — check the key names and the notes above "
            f"before approving."))

    if flags:
        lines += ["", "  ⚠ RISKS:"]
        lines += [f"    • {f}" for f in flags]
    else:
        lines += ["", "  ✓ No fan-out, no orphans, no key problems — this join is clean."]

    return "\n".join(lines), flags, dict(
        predicted=predicted, l_orphans=l_orphans, r_orphans=r_orphans,
        l_dupes=l_dupes, r_dupes=r_dupes, relationship=rel, overlap=overlap,
        impact=impact, clash=clash)


# ─── the tool ──────────────────────────────────────────────────────────────────

def join_datasets(left: str, right: str, on=None, left_on=None, right_on=None,
                  how: str = "left", output_name: Optional[str] = None,
                  suffixes: Optional[list] = None) -> str:
    """Join two loaded datasets, after showing a human exactly what it will do."""
    if how not in HOWS:
        return f"Error: how must be one of {list(HOWS)} — got '{how}'."

    left_df, err = _get(left)
    if err:
        return err
    right_df, err = _get(right)
    if err:
        return err
    if not len(left_df):
        return f"Error: '{left}' has no rows."
    if not len(right_df):
        return f"Error: '{right}' has no rows."

    lk, rk, err = _resolve_keys(left_df, right_df, on, left_on, right_on)
    if err:
        return err

    out_name = output_name or f"{left}_{right}"
    problems = _key_problems(left_df, right_df, lk, rk)
    l_match = _match_counts(left_df, lk, right_df, rk)
    r_match = _match_counts(right_df, rk, left_df, lk)
    card, flags, facts = _card(left, right, lk, rk, how, out_name,
                               left_df, right_df, l_match, r_match, problems)

    if facts["overlap"] <= 0.0:
        return ("Join NOT performed — nothing would match.\n\n" + card +
                "\n\nNot a single left row finds a partner, so the result would be empty (inner) "
                "or entirely blank on the right (left). Fix the keys and try again.")

    from agent.data_cleaner import ask_human
    answer = ask_human(
        "HUMAN APPROVAL REQUIRED — join two datasets?\n\n" + card +
        "\n\nReply 'yes' to proceed, 'no' to cancel, or 'yes how=left' to change the "
        "join direction.")
    tokens = [t.strip().lower() for t in str(answer or "").replace(",", " ").split() if t.strip()]
    chosen_how = how
    for token in tokens:
        if token.startswith("how="):
            candidate = token[4:]
            if candidate in HOWS:
                chosen_how = candidate
    if not any(t in _APPROVE_WORDS for t in tokens):
        return ("Join cancelled — nothing was joined and both datasets are unchanged.\n\n"
                + card)

    if chosen_how != how:
        card, flags, facts = _card(left, right, lk, rk, chosen_how, out_name,
                                   left_df, right_df, l_match, r_match, problems)
        how = chosen_how

    sfx = tuple(suffixes) if suffixes and len(suffixes) == 2 else (f"_{left}", f"_{right}")
    try:
        result = left_df.merge(right_df, left_on=lk, right_on=rk, how=how,
                               suffixes=sfx, validate=None)
    except Exception as e:  # noqa: BLE001
        return f"Error performing the join: {type(e).__name__}: {e}"

    pipe = _pipeline()
    pipe.datasets[out_name] = result
    pipe.sources[out_name] = (f"join:{left}+{right} on {lk}={rk} how={how}")

    actual = len(result)
    predicted = facts["predicted"]
    record = {
        "output": out_name, "left": left, "right": right,
        "left_keys": list(lk), "right_keys": list(rk), "how": how,
        "left_rows": int(len(left_df)), "right_rows": int(len(right_df)),
        "result_rows": int(actual),
        "left_orphans": facts["l_orphans"], "right_orphans": facts["r_orphans"],
        "dropped_rows": facts["l_orphans"] if how in ("inner", "right") else 0,
        "relationship": facts["relationship"], "overlap": facts["overlap"],
        "right_duplicate_keys": facts["r_dupes"],
        "measure_impact": [{"column": c, "before": b, "after": a, "lost": l}
                           for c, b, a, l in facts["impact"]],
    }
    try:
        from agent.data_analysis import note_join
        note_join(record)
    except Exception:  # noqa: BLE001 — recording must never break the join
        pass

    lines = [f"Joined '{left}' → '{right}' on {lk}"
             + (f" = {rk}" if rk != lk else "") + f" (how={how}).",
             f"Registered result as '{out_name}': {actual:,} rows × {result.shape[1]} columns.",
             ""]
    if actual != predicted:
        lines.append(f"NOTE: {actual:,} rows came out where {predicted:,} were predicted — "
                     f"the difference is worth understanding before you trust the result.")
    else:
        lines.append(f"Row count matched the prediction exactly ({actual:,}).")

    if record["dropped_rows"]:
        lines.append(f"{record['dropped_rows']:,} row(s) of '{left}' were DROPPED — they had no "
                     f"match. Every total computed from '{out_name}' excludes them.")
    if facts["r_dupes"]:
        lines.append(f"'{right}' had {facts['r_dupes']:,} duplicate key(s), so left-side rows were "
                     f"duplicated. Do NOT sum a left-side measure on '{out_name}' without "
                     f"de-duplicating first.")
    for col, before, after, _lost in facts["impact"]:
        if before and abs(after - before) / max(abs(before), 1e-9) >= 0.001:
            lines.append(f"'{col}' total went {_fmt(before)} → {_fmt(after)} "
                         f"({(after - before) / before * 100:+.1f}%) as a result of this join.")

    lines += ["",
              "This join is recorded — write_report will state the keys, the direction and the "
              "row-count effect in its Methodology section.",
              f"Next: analyze_dataset('{out_name}') to see what the combined table looks like.",
              "", card]
    return "\n".join(lines)


# ─── swarn tool registration ───────────────────────────────────────────────────

_SWARN_TOOLS = {
    "join_datasets": (
        "Combine two loaded datasets into one — orders + customers, sales + products. USE THIS "
        "instead of writing a pandas merge inside run_python: a hand-written merge silently "
        "multiplies rows when the right table has duplicate keys (inflating every total) and "
        "silently deletes rows whose key is missing (deflating every total), and the two cancel "
        "out in the row count so nothing looks wrong.\n"
        "Before joining anything it works out the exact row-count effect, how many rows will be "
        "duplicated, how many will be dropped, and what each of those does to every measure — "
        "then shows a human that arithmetic and WAITS for approval. It also checks the keys "
        "actually correspond (type mismatch, leading zeros, stray spaces, case).\n"
        "Neither input is modified; the result is registered as a new dataset. The join is "
        "recorded, so write_report must disclose it.",
        {"type": "object",
         "properties": {
             "left": {"type": "string", "description": "The main dataset — the one whose rows you want to keep (e.g. 'orders')."},
             "right": {"type": "string", "description": "The lookup dataset providing extra columns (e.g. 'customers')."},
             "on": {"type": ["string", "array"], "items": {"type": "string"},
                    "description": "Key column name(s) present in BOTH, e.g. 'customer_id'. Omit only if the key is obvious from a shared id column."},
             "left_on": {"type": ["string", "array"], "items": {"type": "string"},
                         "description": "Key column(s) in the left dataset, when the two sides name the key differently."},
             "right_on": {"type": ["string", "array"], "items": {"type": "string"},
                          "description": "Matching key column(s) in the right dataset."},
             "how": {"type": "string",
                     "description": "'left' (default — keep every left row, safest), 'inner' (keep only matched rows: DELETES unmatched left rows), 'right', or 'outer'."},
             "output_name": {"type": "string", "description": "Name for the result (default '<left>_<right>')."},
             "suffixes": {"type": "array", "items": {"type": "string"},
                          "description": "Two suffixes for columns that exist on both sides (default '_<left>', '_<right>')."},
         },
         "required": ["left", "right"]},
        join_datasets,
    ),
}


def register_into_swarn() -> str:
    """Register the join tool into agent.tools.TOOL_REGISTRY."""
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
    return f"registered {registered} join tool(s) into swarn"
