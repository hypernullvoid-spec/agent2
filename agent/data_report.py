"""
Findings report — the document you actually hand to people.

Structure is fixed and deliberate: Background → Key figures → Key takeaways →
Methodology → Appendix, carrying a Situation / Complication / Resolution story.
Key figures is generated wholly from the evidence ledger — see _kpi_rows.

  Situation     why you were asked to look — restated in the reader's terms
  Complication  what the data turned out to complicate about that question
  Resolution    the 2nd-degree analysis you propose next

Who writes what
───────────────
The narrative is YOURS (the analyst's). Nobody is talking to the dashboard;
they are talking to you, so the story has to be told in words.

But the numbers, the charts, what cleaning did to the data, and the limitations
are generated here from the recorded evidence — NOT retyped by the narrator.
That split is the whole point: a polished report is trusted more than terminal
output, so an unverified claim inside one does more damage, not less. Before
anything is written the narrative is checked against what was actually
measured, and the report is REFUSED if it contradicts it.

Outputs (both, side by side in the workspace):
  <name>_report.md     for git, review, diffing
  <name>_report.html    self-contained — charts embedded, opens anywhere, emailable
"""

from __future__ import annotations

import base64
import datetime
import html
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace"))
PLOTS_SUBDIR = "plots"
MAX_CHARTS = 12
CHART_MAX_BYTES = 2_000_000      # skip embedding anything absurd


# ─── gathering the verified material ──────────────────────────────────────────

def _pipeline():
    from agent.ml.data_pipeline import get_data_pipeline
    return get_data_pipeline()


def _charts_for(dataset: str) -> list:
    """Charts this dataset produced. Tool filenames start '<dataset>__', which
    is why the naming convention exists."""
    directory = os.path.join(WORKSPACE_DIR, PLOTS_SUBDIR)
    if not os.path.isdir(directory):
        return []
    prefix = f"{dataset}__"
    found = sorted(f for f in os.listdir(directory)
                   if f.startswith(prefix) and f.endswith(".png"))
    return [os.path.join(PLOTS_SUBDIR, f) for f in found[:MAX_CHARTS]]


def _chart_caption(filename: str, dataset: str) -> str:
    stem = os.path.basename(filename)[len(dataset) + 2:].rsplit(".", 1)[0]
    words = stem.replace("__", " — ").replace("_", " ")
    return words[:1].upper() + words[1:] if words else stem


def _cleaning_record(dataset: str) -> tuple:
    """(manifest, source_name) for whichever dataset this one was derived from."""
    pipe = _pipeline()
    cache = getattr(pipe, "_cleaner_cache", {}) or {}
    for candidate in (dataset, dataset.replace("_clean", "")):
        cleaner = cache.get(candidate)
        if cleaner is not None and getattr(cleaner, "applied_ops", None):
            try:
                return cleaner.manifest(), candidate
            except Exception:  # noqa: BLE001
                return None, candidate
    return None, None


def _limitations(df: pd.DataFrame, dataset: str) -> list:
    """What this data cannot support — read off the cleaner's own markers, so it
    cannot be softened or left out by whoever writes the narrative."""
    notes = []
    for column in df.columns:
        base = None
        for suffix in ("_was_missing", "_was_capped"):
            if column.endswith(suffix):
                base = column[: -len(suffix)]
                break
        if base is None or base not in df.columns:
            continue
        share = float(df[column].mean()) if df[column].dtype.kind in "iufb" else 0.0
        if share < 0.01:
            continue
        if column.endswith("_was_missing"):
            # The marker records that a value WAS missing, not that it was filled.
            # If the impute step was never approved the column is still blank, and
            # claiming it was filled would be a falsehood in the one section the
            # narrator cannot edit.
            still_blank = int(df[base].isnull().sum())
            if still_blank:
                notes.append(f"**{base}** — {still_blank:,} rows ({still_blank / len(df):.0%}) are "
                             f"still blank; this was deliberately not filled in. Any figure for "
                             f"{base} describes only the rows that have a value, so state the "
                             f"count alongside it.")
                continue
            filled = df.loc[df[column] == 1, base]
            value = ""
            if len(filled) and not filled.mode().empty:
                value = f" (all set to {filled.mode().iloc[0]})"
            notes.append(f"**{base}** — {share:.0%} of these values were blank and were filled "
                         f"in during cleaning{value}. Any average or trend involving {base} is "
                         f"partly an estimate, not a measurement.")
        else:
            notes.append(f"**{base}** — {share:.0%} of these values were extreme and were "
                         f"rewritten to the edge of the normal range, so the highest and lowest "
                         f"figures are capped rather than actual.")
    if int(df.duplicated().sum()):
        notes.append(f"{int(df.duplicated().sum()):,} duplicate rows remain in the analysed set.")
    return (notes + _truncation_limitations(dataset) + _grain_limitations(dataset)
            + _join_limitations(dataset) + _effect_limitations(dataset)
            + _reconciliation_limitations(dataset)
            + _intrinsic_limitations(df, dataset))


# Lorem-ipsum stems, used to spot generated filler text.
#
# Deliberately EXCLUDES words that are ordinary in English or common in other
# languages — 'error', 'sit', 'no', 'rem', 'nam', 'modi', 'natus', 'total'. A log
# table whose level column reads 'error' is real data, and saying otherwise is a
# far worse failure than staying quiet.
#
# Presence alone is not the test either: genuine Latin exists in legal, medical
# and botanical datasets. What separates Faker's filler from real Latin prose is
# COVERAGE — Faker draws every token from a closed ~180-word list, so almost the
# whole column is in this set, whereas real Latin carries domain terms and proper
# nouns that are not. See _placeholder_columns.
_LOREM = {
    "accusam", "accusamus", "accusantium", "adipisci", "alias", "aliqua", "aliquid",
    "aliquyam", "animi", "aperiam", "architecto", "asperiores", "aspernatur", "assumenda",
    "atque", "autem", "beatae", "blanditiis", "clita", "commodi", "consectetur",
    "consequatur", "consequuntur", "consetetur", "corporis", "corrupti", "culpa", "cumque",
    "cupiditate", "debitis", "delectus", "deleniti", "deserunt", "dicta", "dignissimos",
    "distinctio", "dolor", "dolore", "dolorem", "doloremque", "dolores", "doloribus",
    "dolorum", "ducimus", "eaque", "earum", "eirmod", "eius", "eiusmod", "eligendi",
    "enim", "erat", "eveniet", "excepturi", "exercitationem", "expedita", "explicabo",
    "facere", "facilis", "fuga", "fugiat", "fugit", "gubergren", "harum", "illo", "illum",
    "impedit", "incididunt", "incidunt", "inventore", "invidunt", "ipsa", "ipsam", "ipsum",
    "iste", "itaque", "iure", "iusto", "justo", "kasd", "labore", "laboriosam", "laborum",
    "laudantium", "libero", "lorem", "magnam", "magni", "maiores", "maxime", "minus",
    "molestiae", "molestias", "mollitia", "natus", "necessitatibus", "nemo", "neque",
    "nesciunt", "nihil", "nisi", "nobis", "nonumy", "nostrum", "nulla", "numquam",
    "occaecati", "odio", "odit", "officia", "officiis", "omnis", "optio", "pariatur",
    "perferendis", "perspiciatis", "placeat", "porro", "possimus", "praesentium",
    "provident", "quae", "quaerat", "quam", "quas", "quasi", "quia", "quibusdam", "quidem",
    "quis", "quisquam", "quod", "quos", "ratione", "rebum", "recusandae", "reiciendis",
    "repellat", "repellendus", "reprehenderit", "repudiandae", "rerum", "saepe", "sanctus",
    "sapiente", "sequi", "similique", "sint", "soluta", "stet", "sunt", "suscipit",
    "takimata", "tempor", "tempora", "tempore", "temporibus", "tenetur", "totam", "ullam",
    "unde", "velit", "veniam", "veritatis", "vitae", "voluptas", "voluptate", "voluptatem",
    "voluptates", "voluptatibus", "voluptatum", "voluptua",
}

# Share of a column's TOKENS that must come from the filler vocabulary before it
# is called placeholder text. High on purpose: a false "your data is fake" is
# worse than missing a synthetic column.
PLACEHOLDER_TOKEN_SHARE = float(os.environ.get("SWARN_PLACEHOLDER_SHARE", "0.75"))

# Name-based detection is a heuristic and cannot be anything else: nothing in a
# float column says "this is money". Both lists are extendable without touching
# code (SWARN_MONEY_TOKENS / SWARN_COST_TOKENS, comma-separated) because column
# vocabularies are domain- and language-specific — a Spanish schema calls it
# `ingresos`, a German one `umsatz`. A miss here costs a limitation note, never a
# wrong number, so the defaults stay conservative rather than guessing widely.
_MONEY_TOKENS = tuple(t.strip().lower() for t in os.environ.get(
    "SWARN_MONEY_TOKENS",
    "revenue,sales,amount,turnover,gmv,billing,income,ingresos,umsatz,chiffre,fatturato"
).split(",") if t.strip())
_COST_TOKENS = tuple(t.strip().lower() for t in os.environ.get(
    "SWARN_COST_TOKENS",
    "cost,cogs,margin,profit,expense,spend,discount,refund,landed,coste,costo,kosten"
).split(",") if t.strip())

# How thin a dimension may get before its league table is called sampling noise,
# and how narrow a date range may be before month-on-month stops being seasonal.
THIN_ROWS_PER_LEVEL = float(os.environ.get("SWARN_THIN_ROWS_PER_LEVEL", "20"))
SEASONALITY_MIN_DAYS = int(os.environ.get("SWARN_SEASONALITY_MIN_DAYS", "400"))
STALE_DATA_DAYS = int(os.environ.get("SWARN_STALE_DATA_DAYS", "365"))


def _truncation_limitations(dataset: str) -> list:
    """A partial read stated where the narrator cannot soften it.

    This one is listed first among the limitations on purpose. Every other
    caveat qualifies a figure; this one says the figures describe a different,
    smaller dataset than the reader believes they are looking at.
    """
    cut = _truncation(dataset)
    read, total = int(cut.get("rows_read") or 0), int(cut.get("rows_total") or 0)
    if not total or read >= total:
        return []
    return [
        f"**This is not the whole file.** Only {read:,} of the {total:,} rows "
        f"({read / total:.1%}) were read; {total - read:,} were not. Every count, total and "
        f"average in this report describes that subset alone — none of them is a figure for "
        f"the full dataset, and the rows left out are the ones at the END of the file, so an "
        f"ordered extract is missing its most recent period entirely."
    ]


def _grain_limitations(dataset: str) -> list:
    """Rows that duplicate an identifier, stated where it cannot be edited out.

    A doubled order is indistinguishable from two orders once it is in the
    frame, so this is the last point at which anyone can know.
    """
    grain = _grain(dataset)
    source = dataset
    if not grain.get("extra_rows"):
        # A defect found in a source table does not stop mattering because the
        # table was then joined. The duplicate rows travel into the result and
        # inflate its totals exactly as they did before, so a check run on the
        # inputs has to reach a report written about the output — otherwise the
        # finding exists, and the reader never sees it.
        for join in _recorded_joins(dataset):
            for parent in (join.get("left"), join.get("right")):
                candidate = _grain(parent or "")
                if candidate.get("extra_rows"):
                    grain, source = candidate, parent
                    break
            if grain.get("extra_rows"):
                break
    extra = int(grain.get("extra_rows") or 0)
    if not extra:
        return []
    keys = "/".join(grain.get("keys") or [])
    rows = int(grain.get("rows") or 0)
    inherited = (f" These came from `{source}`, which this table was built from, so they are "
                 f"still here." if source != dataset else "")
    plural = "those records" if extra > 1 else "that record"
    note = (f"**Rows are duplicated.** {extra:,} row(s) repeat a `{keys}` that should appear "
            f"once, out of {rows:,}. Every count and total in this report counts {plural} "
            f"more than once and is overstated by that much.")
    if grain.get("conflicting_columns"):
        note += (f" They are not identical copies — they disagree on "
                 f"{grain['conflicting_columns'][:4]} — so they are conflicting records, and "
                 f"which one is correct has not been decided.")
    return [note + inherited]


def _effect_limitations(dataset: str) -> list:
    """Differences that passed a significance test but are too small to matter.

    This exists because 'statistically significant' reads to most people as
    'important', and with enough rows it means neither. Left unstated, a
    negligible gap gets written up as one group outperforming another.
    """
    notes = []
    for effect in _effects(dataset):
        if not (effect.get("significant") and effect.get("negligible")):
            continue
        notes.append(
            f"**A significant difference in {effect.get('value_col')} by "
            f"{effect.get('group_col')} is not a meaningful one.** The test rules out chance "
            f"(p = {effect.get('p', float('nan')):.3g}), but the effect size is negligible "
            f"({effect.get('effect_kind')} = {effect.get('effect'):+.3f}) — the groups are "
            f"practically the same. With enough rows almost any difference becomes "
            f"significant; nothing here supports describing one group as outperforming "
            f"another.")
    return notes


def _reconciliation_limitations(dataset: str) -> list:
    """A figure that failed to match the number the business already has."""
    notes = []
    for check in _reconciliations(dataset):
        if check.get("matches"):
            continue
        notes.append(
            f"**{check.get('column')} does not reconcile.** This analysis makes it "
            f"{check.get('actual', 0):,.2f}; {check.get('label')} says "
            f"{check.get('expected', 0):,.2f} — a difference of {check.get('gap', 0):+,.2f}. "
            f"The gap has not been explained, so this figure should not be presented as "
            f"authoritative.")
    return notes


def _join_limitations(dataset: str) -> list:
    """What a join took away from this table, stated where the narrator cannot edit it.

    Dropped rows and fanned-out rows are the two ways a join changes every
    total in a report while leaving the output looking entirely normal. Both
    belong here rather than in the narrative, for the same reason the cleaner's
    markers do: the section that admits a weakness must not be written by
    whoever is arguing the conclusion.
    """
    notes = []
    for join in _recorded_joins(dataset):
        left, right = join.get("left"), join.get("right")
        dropped = int(join.get("dropped_rows") or 0)
        left_rows = int(join.get("left_rows") or 0)
        if dropped and left_rows:
            notes.append(
                f"**Rows lost to a join.** {dropped:,} of the {left_rows:,} rows in `{left}` "
                f"({dropped / left_rows:.1%}) had no match in `{right}` and were removed by the "
                f"`{join.get('how')}` join. Every count and total in this report describes only "
                f"the rows that matched, so none of them is a figure for the whole of `{left}`.")
        for impact in join.get("measure_impact") or []:
            before, lost = impact.get("before") or 0.0, impact.get("lost") or 0.0
            if before and lost and abs(lost / before) >= 0.005:
                notes.append(
                    f"**{impact.get('column')} is understated.** {_num(lost)} "
                    f"({lost / before:.1%} of the pre-join total) sat in rows the join removed. "
                    f"Any figure for {impact.get('column')} here is that much too low.")
        if int(join.get("right_duplicate_keys") or 0):
            notes.append(
                f"**Rows were duplicated by a join.** `{right}` was not unique on "
                f"{join.get('right_keys')}, so rows from `{left}` appear more than once. Sums "
                f"and averages over `{left}`'s own measures count those rows repeatedly and are "
                f"overstated; de-duplicate before quoting them.")
    return notes


def _placeholder_columns(df: pd.DataFrame, sample: int = 300) -> list:
    """Text columns whose whole vocabulary is lorem-ipsum filler.

    Measured as a share of TOKENS, not of rows. A row-level test ("does this row
    contain a filler word?") flags genuine Latin — a legal maxim or a botanical
    name trips it on one word — and asserting that someone's real data is fake is
    the most damaging thing this module can say. Requiring most of the column's
    words to come from the closed filler list separates the two: Faker draws
    everything from that list, real Latin does not.
    """
    found = []
    for column in df.columns:
        series = df[column]
        if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
            continue
        values = series.dropna().astype(str).head(sample)
        if len(values) < 10:
            continue
        # Words under 4 letters are dropped from BOTH sides of the ratio: "in",
        # "ad", "et", "sed" appear in filler and in half the languages on earth,
        # so counting them tells you nothing and risks flagging real prose.
        tokens = [w for v in values for w in re.split(r"[^a-zA-Z]+", v.lower()) if len(w) >= 4]
        if len(tokens) < 30:
            continue
        share = sum(1 for t in tokens if t in _LOREM) / len(tokens)
        if share >= PLACEHOLDER_TOKEN_SHARE:
            found.append((str(column), share))
    return found


def _intrinsic_limitations(df: pd.DataFrame, dataset: str) -> list:
    """What this data cannot support REGARDLESS of what cleaning did to it.

    Limitations used to be read only off the cleaner's markers, so a file loaded
    straight from disk and never cleaned produced a report stating zero
    limitations — the most confident-looking report being precisely the one
    nobody had checked. These are properties of the data itself.
    """
    notes: list = []
    rows = max(1, len(df))
    lower = {str(c).lower(): c for c in df.columns}

    # ── revenue without cost is half a P&L ──────────────────────────────────
    money = [c for k, c in lower.items() if any(t in k for t in _MONEY_TOKENS)]
    has_cost = any(any(t in k for t in _COST_TOKENS) for k in lower)
    if money and not has_cost:
        notes.append(
            f"**No cost or margin data.** {', '.join(f'`{c}`' for c in money[:3])} measures money "
            f"coming in, and nothing here measures what it cost to earn. The highest-revenue "
            f"category may well be the least profitable one, so nothing above can rank anything "
            f"by profit and no recommendation should be read as one."
        )

    # ── placeholder / synthetic content ─────────────────────────────────────
    fake_cols = _placeholder_columns(df)
    if fake_cols:
        listed = ", ".join(f"`{c}` ({share:.0%} filler)" for c, share in fake_cols[:4])
        notes.append(
            f"**Some columns look like generated placeholder text.** {listed}. Nearly every word "
            f"in {'that column' if len(fake_cols) == 1 else 'those columns'} comes from the "
            f"lorem-ipsum vocabulary, which normally means the data is sample data rather than a "
            f"real extract. The mechanics of the analysis hold either way — but confirm the source "
            f"before acting on any figure or recommendation above."
        )

    # ── one season is not a seasonal pattern ────────────────────────────────
    spans = []
    for column in df.columns:
        series = df[column]
        try:
            if pd.api.types.is_datetime64_any_dtype(series):
                parsed = series
            elif series.dtype == object or pd.api.types.is_string_dtype(series):
                parsed = _da()._as_datetime(series) if _da() else None
                if parsed is None or parsed.notna().mean() < 0.8:
                    continue
            else:
                continue
            valid = parsed.dropna()
            if len(valid) > 10:
                spans.append((str(column), valid.min(), valid.max()))
        except Exception:  # noqa: BLE001
            continue
    if spans:
        column, lo, hi = max(spans, key=lambda s: (s[2] - s[1]).days)
        days = (hi - lo).days
        if days < SEASONALITY_MIN_DAYS:
            notes.append(
                f"**One {max(1, days // 30)}-month window only** (`{column}`: {lo:%b %Y} – {hi:%b %Y}). "
                f"Each calendar month appears exactly once, so a month-to-month rise or fall is a "
                f"single observation, not a demonstrated seasonal pattern. Establishing seasonality "
                f"needs a second year to compare against; treat every trend statement above as "
                f"describing what happened, not what recurs."
            )
        age = (pd.Timestamp.now().normalize() - hi).days
        if age > STALE_DATA_DAYS:
            notes.append(
                f"**The data is {age // 365} year(s) old** — it ends {hi:%B %Y}. Anything it says "
                f"about customer behaviour describes that period, not today."
            )

    # ── dimensions too thin to rank ─────────────────────────────────────────
    # Only the dimensions this report actually ranked. Every high-cardinality
    # column in the frame is technically "thin", but warning about all of them
    # buries the two that a reader is about to act on.
    ranked = {str(key[0]) for key in _evidence().get("rankings", {})}
    thin = []
    for column in ranked:
        if column not in df.columns:
            continue
        levels = int(df[column].nunique(dropna=True))
        if levels >= 15 and levels <= 0.9 * rows and rows / levels <= THIN_ROWS_PER_LEVEL:
            thin.append((column, levels, rows / levels))
    if thin:
        detail = "; ".join(f"`{c}` ({n:,} values, ~{per:.0f} rows each)" for c, n, per in
                           sorted(thin, key=lambda t: t[2]))
        notes.append(
            f"**Some league tables above are too thin to rank.** {detail}. At that depth the order "
            f"of the top few is largely sampling noise — a handful of rows moving would reshuffle "
            f"it. Read those tables as indicative, and do not describe positions 1 and 2 as a "
            f"meaningful lead."
        )

    # ── merge leftovers ─────────────────────────────────────────────────────
    pairs = [c for c in df.columns if str(c).endswith("_x") and f"{str(c)[:-2]}_y" in df.columns]
    for base in pairs:
        twin = f"{str(base)[:-2]}_y"
        try:
            identical = bool(df[base].equals(df[twin]))
        except Exception:  # noqa: BLE001
            identical = False
        detail = ("hold identical values in every row, so one is redundant"
                  if identical else
                  "DISAGREE on some rows — check which one the figures above actually used")
        notes.append(
            f"**`{base}` / `{twin}` are merge artefacts.** They {detail}. Column names ending "
            f"`_x`/`_y` mean a join produced a name clash that was never resolved, and they should "
            f"not appear in a finished report."
        )

    if not _cleaning_record(dataset)[0]:
        notes.append(
            "**Nothing was cleaned or corrected.** This dataset was analysed exactly as loaded, so "
            "every figure inherits whatever errors the source file contains. Absence of a cleaning "
            "record is not evidence that the data was clean."
        )
    return notes


def _da():
    """The analysis module, for its date parser — imported lazily to keep the
    report module usable on its own."""
    try:
        from agent import data_analysis
        return data_analysis
    except Exception:  # noqa: BLE001
        return None


def _provenance_lines(df: pd.DataFrame, dataset: str) -> list:
    """Where this frame came from and how its derived columns were computed.

    A methodology section that only describes cleaning leaves the reader unable to
    rebuild the numbers: a real report never said the three source files had been
    joined, nor that Revenue was Price × Quantity. Both are recoverable — the
    source is in the registry, and an exact arithmetic relationship between
    columns can simply be tested.
    """
    lines = ["### How the analysed table was built", ""]
    rows, cols = df.shape
    source = ""
    try:
        source = str(_pipeline().sources.get(dataset, "") or "")
    except Exception:  # noqa: BLE001
        source = ""
    origin = f" from `{source}`" if source else ""
    lines.append(f"- `{dataset}` as analysed: **{rows:,} rows × {cols} columns**{origin}.")

    cut = _truncation(dataset)
    if cut:
        read, total = int(cut.get("rows_read") or 0), int(cut.get("rows_total") or 0)
        if total and read < total:
            lines.append(
                f"- **Only part of the source file was read** — {read:,} of {total:,} rows "
                f"({read / total:.1%}), by {cut.get('strategy', 'a row cap')}. The "
                f"{total - read:,} rows not read are the last in the file, so if it is "
                f"ordered by date this table stops partway through the period.")

    recorded = _recorded_joins(dataset)
    for join in recorded:
        keys = ", ".join(f"`{k}`" for k in join.get("left_keys") or [])
        rkeys = join.get("right_keys") or []
        if rkeys and rkeys != (join.get("left_keys") or []):
            keys += " = " + ", ".join(f"`{k}`" for k in rkeys)
        how = str(join.get("how", "?"))
        article = "an" if how[:1] in "aeiou" else "a"
        lines.append(
            f"- This table is {article} **{how} join** of `{join.get('left')}` "
            f"({join.get('left_rows', 0):,} rows) to `{join.get('right')}` "
            f"({join.get('right_rows', 0):,} rows) on {keys} — a "
            f"{join.get('relationship', 'unknown')} relationship producing "
            f"{join.get('result_rows', 0):,} rows.")
        dropped = int(join.get("dropped_rows") or 0)
        if dropped:
            lines.append(
                f"  - **{dropped:,} row(s) of `{join.get('left')}` were dropped** by this join "
                f"because they had no match. Every total below excludes them.")
        dupes = int(join.get("right_duplicate_keys") or 0)
        if dupes:
            lines.append(
                f"  - `{join.get('right')}` held {dupes:,} duplicate key(s), so rows from "
                f"`{join.get('left')}` were repeated. Sums over left-hand measures count those "
                f"rows more than once.")
        for impact in join.get("measure_impact") or []:
            before, after = impact.get("before") or 0.0, impact.get("after") or 0.0
            if before and abs(after - before) / max(abs(before), 1e-9) >= 0.001:
                lines.append(
                    f"  - `{impact.get('column')}` totalled {_num(before)} before the join and "
                    f"{_num(after)} after ({(after - before) / before * 100:+.1f}%).")

    # An unrecorded merge — hand-written pandas inside run_python — leaves no
    # ledger entry, so the only trace is the name clash pandas suffixes. That
    # clue exists only when the two tables shared a column name, which is why
    # it is a fallback and not the mechanism.
    merged = [str(c) for c in df.columns
              if str(c).endswith("_x") and f"{str(c)[:-2]}_y" in df.columns]
    if merged and not recorded:
        lines.append(
            f"- This table is the result of a **join** — `{merged[0]}`/`{merged[0][:-2]}_y` are the "
            f"name clash it produced. The join keys, direction and row-count effect were NOT "
            f"recorded, so the merge cannot be reproduced or checked from this report. Rebuild it "
            f"with join_datasets, which records them."
        )
    for check in _reconciliations(dataset):
        if check.get("matches"):
            lines.append(
                f"- `{check.get('column')}` **reconciles** with {check.get('label')}: "
                f"{check.get('actual', 0):,.2f} against {check.get('expected', 0):,.2f}.")
    grain = _grain(dataset)
    if grain.get("keys") and not grain.get("extra_rows"):
        lines.append(f"- Verified as **one row per {'/'.join(grain['keys'])}**, so totals "
                     f"count each one exactly once.")
    for formula in _derived_formulas(df):
        lines.append(f"- {formula}")
    lines.append("")
    return lines


def _derived_formulas(df: pd.DataFrame, max_columns: int = 40) -> list:
    """Numeric columns that are an arithmetic function of two others.

    Two things this must get right on data it has never seen:

    * TOLERANCE is relative, not absolute. `notional = units × price` in the
      billions carries float error far larger than 1e-9, so an absolute test
      silently finds nothing on exactly the large-magnitude finance tables where
      documenting the formula matters most.
    * WIDE frames still get searched. Bailing above a column count meant a
      40-column table reported no derivations at all — indistinguishable, in the
      report, from a table that genuinely had none. The search runs on a small
      row sample and every candidate is then confirmed against the full column,
      which keeps an O(k³) scan affordable without weakening the claim.
    """
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) < 3:
        return []
    capped = ""
    if len(numeric) > max_columns:
        numeric = numeric[:max_columns]
        capped = (f" (searched the first {max_columns} numeric columns only — "
                  f"others were not checked)")

    full = df[numeric].dropna()
    if len(full) < 5:
        return []
    probe = full.head(200)

    def matches(target_values, candidate) -> bool:
        return bool(np.isclose(target_values, candidate, rtol=1e-9, atol=0.0,
                               equal_nan=False).all())

    found = []
    for target in numeric:
        hit = None
        for i, left in enumerate(numeric):
            if left == target or hit:
                continue
            for right in numeric[i + 1:]:
                if right == target:
                    continue
                for symbol, op in (("×", lambda a, b: a * b), ("+", lambda a, b: a + b)):
                    if not matches(probe[target].values, op(probe[left], probe[right]).values):
                        continue
                    # Confirm on every row before stating it as fact.
                    if matches(full[target].values, op(full[left], full[right]).values):
                        hit = (left, symbol, right)
                        break
                if hit:
                    break
        if hit:
            left, symbol, right = hit
            found.append(
                f"`{target}` is a derived column: **`{left}` {symbol} `{right}`** "
                f"(holds for all {len(full):,} rows). It was computed here, not measured at "
                f"source.{capped}"
            )
    return found


def _measured_numbers() -> list:
    """Relationships actually measured, straight from the evidence ledger."""
    try:
        from agent.data_analysis import _EVIDENCE, _strength
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for (a, b), r in sorted(_EVIDENCE.get("correlations", {}).items(),
                            key=lambda kv: -abs(kv[1])):
        rows.append((f"{a} ↔ {b}", f"{r:+.2f}", _strength(r)))
    return rows[:12]


def _grain(dataset: str) -> dict:
    try:
        from agent.data_analysis import grain_for
        return grain_for(dataset)
    except Exception:  # noqa: BLE001
        return {}


def _effects(dataset: str) -> list:
    try:
        from agent.data_analysis import effects_for
        return effects_for(dataset)
    except Exception:  # noqa: BLE001
        return []


def _reconciliations(dataset: str) -> list:
    try:
        from agent.data_analysis import reconciliations_for
        return reconciliations_for(dataset)
    except Exception:  # noqa: BLE001
        return []


def _truncation(dataset: str) -> dict:
    """What the loader left unread — see data_analysis.note_truncation."""
    try:
        from agent.data_analysis import truncation_for
        return truncation_for(dataset)
    except Exception:  # noqa: BLE001
        return {}


def _recorded_joins(dataset: str) -> list:
    """Joins the ledger says produced this table — see data_analysis.note_join."""
    try:
        from agent.data_analysis import joins_for
        return joins_for(dataset)
    except Exception:  # noqa: BLE001
        return []


def _invented_groups() -> dict:
    try:
        from agent.data_analysis import _EVIDENCE
        return {k: sorted(set(v)) for k, v in _EVIDENCE.get("imputed_groups", {}).items()}
    except Exception:  # noqa: BLE001
        return {}


def _evidence() -> dict:
    try:
        from agent.data_analysis import _EVIDENCE
        return _EVIDENCE
    except Exception:  # noqa: BLE001
        return {}


def _num(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:,.0f}" if abs(value - round(value)) < 1e-9 else f"{value:,.2f}"


def _kpi_rows(df: pd.DataFrame, dataset: str) -> list:
    """The headline numbers, derived from the ledger rather than retyped.

    A reader looks for these first and quotes them second, so they are the last
    thing that should pass through a narrator. Every row here is either a shape
    of the frame or something a tool actually measured.
    """
    rows = [("Rows analysed", f"{len(df):,}")]
    evidence = _evidence()

    # Totals, plus the per-ROW average. Deliberately not a per-group average:
    # which group to divide by would be an arbitrary pick between the rankings
    # that happen to have been run, and 'average revenue per product' reads as a
    # KPI while meaning almost nothing. Per-group figures live in the ranking
    # tables below, where the group is named.
    seen_totals = set()
    for (_group_col, value_col, how), record in evidence.get("rankings", {}).items():
        total = record.get("total")
        if not total or how != "sum" or value_col in seen_totals:
            continue
        seen_totals.add(value_col)
        rows.append((f"Total {value_col}", _num(total)))
        if len(df):
            rows.append((f"Average {value_col} per row", _num(total / len(df))))

    for (start_col, end_col), record in evidence.get("durations", {}).items():
        label = f"Average {start_col} → {end_col}"
        rows.append((label, f"{record['mean']:,.2f} {record['unit']}"))
        if record.get("negative"):
            rows.append((f"{label} — rows with an impossible gap",
                         f"{record['negative']:,} (excluded from any reading)"))
    return rows


def _ranking_tables(top: int = 5) -> list:
    """(heading, header-row, body-rows) for every ranking that was measured.

    These sections exist because the dimensional breakdowns — best products,
    biggest cities, revenue by category — were previously only ever bullets the
    narrator wrote by hand, which is precisely where a row gets dropped or two
    near-identical bars get swapped.
    """
    tables = []
    for (group_col, value_col, how), record in _evidence().get("rankings", {}).items():
        order, values = record.get("order") or [], record.get("values") or {}
        if not order:
            continue
        total = record.get("total")
        end = "Lowest" if record.get("ascending") else "Top"
        measure = "count" if value_col == "row count" else f"{how} of {value_col}"
        header = ["#", str(group_col), measure.capitalize()]
        if total:
            header.append("Share")
        body = []
        for rank, label in enumerate(order[:top], start=1):
            row = [str(rank), str(label), _num(values.get(label))]
            if total:
                row.append(f"{100 * float(values.get(label, 0)) / total:.1f}%")
            body.append(row)
        note = ""
        if total:
            covered = sum(float(values.get(l, 0)) for l in order[:top])
            note = (f"{len(order):,} {group_col} value(s) in total; these {len(body)} "
                    f"account for {100 * covered / total:.1f}% of {value_col}.")
        tables.append((f"{end} {len(body)} {group_col} by {measure}", header, body, note))
    return tables


# ─── the document ─────────────────────────────────────────────────────────────

def _build_markdown(*, title, dataset, df, situation, complication, takeaways,
                    next_steps, charts, manifest, source, dashboard_url) -> str:
    today = datetime.date.today().isoformat()
    rows, cols = df.shape
    lines = [f"# {title}", "", f"*{today} · prepared from `{dataset}` "
                              f"({rows:,} rows × {cols} columns)*", ""]

    # ── 1. Background — the Situation, in the reader's terms ──
    lines += ["## Background", "", situation.strip(), ""]
    if source:
        lines.append(f"Source data: `{source}`. ")
    if dashboard_url:
        lines.append(f"Live charts: {dashboard_url}")
    lines.append("")

    # ── 2. The measured figures — generated, never retyped by the narrator ──
    kpis = _kpi_rows(df, dataset)
    if kpis:
        lines += ["## Key figures", "", "| Metric | Value |", "| --- | --- |"]
        lines += [f"| {label} | {value} |" for label, value in kpis]
        lines.append("")

    for heading, header, body, note in _ranking_tables():
        lines += [f"### {heading}", "",
                  "| " + " | ".join(header) + " |",
                  "| " + " | ".join("---" for _ in header) + " |"]
        lines += ["| " + " | ".join(cell for cell in row) + " |" for row in body]
        lines.append("")
        if note:
            lines += [f"*{note}*", ""]

    # ── 3. Key takeaways — the story: complication then resolution ──
    lines += ["## Key takeaways", ""]
    if complication:
        lines += ["**The complication.** " + complication.strip(), ""]
    for point in takeaways:
        lines.append(f"- {point.strip()}")
    lines.append("")
    if next_steps:
        lines += ["**What I propose next.**", ""]
        for step in next_steps:
            lines.append(f"1. {step.strip()}")
        lines += ["", "*These next steps are my recommendation — judgement, not measurement. "
                      "Everything above this line comes from the data; this part is what I think "
                      "it means to do about it.*", ""]

    # ── 3. Methodology — including what we changed, which earns the trust ──
    lines += ["## Methodology", ""]
    if manifest:
        before = f"{manifest.get('rows_before', '?'):,}" if isinstance(
            manifest.get("rows_before"), int) else "?"
        lines += [f"The raw data had {before} rows. Cleaning was proposed by the tool, approved "
                  f"by a human, and applied in a fixed order. {len(manifest.get('operations', []))} "
                  f"operation(s) ran:", ""]
        for op in manifest.get("operations", [])[:20]:
            lines.append(f"- {op.get('label', op.get('action', ''))}")
        extra = len(manifest.get("operations", [])) - 20
        if extra > 0:
            lines.append(f"- …and {extra} more (full list in the appendix)")
        lines.append("")
    else:
        lines += ["No cleaning record was found for this dataset — it was analysed as loaded.", ""]

    lines += _provenance_lines(df, dataset)

    limits = _limitations(df, dataset)
    if limits:
        lines += ["### What this data cannot tell you", "",
                  "Read these before quoting any figure above:", ""]
        lines += [f"- {n}" for n in limits]
        lines.append("")
    invented = _invented_groups()
    if invented:
        for column, groups in invented.items():
            shown = ", ".join(groups[:6]) + ("…" if len(groups) > 6 else "")
            lines.append(f"- **{column}** — these groups are entirely filled-in values, so their "
                         f"figures are the fill value rather than anything observed: {shown}")
        lines.append("")

    # ── 4. Appendix — the technical detail ──
    lines += ["## Appendix", ""]
    measured = _measured_numbers()
    if measured:
        lines += ["### Relationships measured", "",
                  "| Pair | Correlation | Strength |", "| --- | --- | --- |"]
        lines += [f"| {p} | {r} | {s} |" for p, r, s in measured]
        lines += ["", "A correlation shows two things moving together. It is not evidence that "
                      "one causes the other.", ""]
    if charts:
        lines += ["### Charts", ""]
        for chart in charts:
            lines.append(f"**{_chart_caption(chart, dataset)}**")
            lines.append("")
            lines.append(f"![{_chart_caption(chart, dataset)}]({chart})")
            lines.append("")
    if manifest:
        lines += ["### Reproducing this", "",
                  "Every cleaning step was recorded, so this can be re-run on a later extract "
                  "without repeating any of the approvals:", "",
                  "```", f"python -m agent.data_cleaner <newfile.csv> --replay "
                         f"<{dataset}.cleaning.json>", "```", ""]
    lines += ["### Column summary", ""]
    lines += ["| Column | Type | Filled | Distinct |", "| --- | --- | --- | --- |"]
    for column in df.columns:
        series = df[column]
        lines.append(f"| {column} | {series.dtype} | {int(series.notna().sum()):,} | "
                     f"{int(series.nunique(dropna=True)):,} |")
    return "\n".join(lines) + "\n"


_HTML_CSS = """
:root { --ink:#1a1f26; --muted:#5c6673; --line:#dfe4e9; --bg:#ffffff;
        --accent:#14625e; --warn:#8a5520; --warnbg:#fdf4e8; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--ink); margin:0; padding:2.5rem 1.25rem 5rem;
       font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:46rem; margin:0 auto; }
h1 { font-size:2rem; line-height:1.15; margin:0 0 .4rem; letter-spacing:-.02em; }
h2 { font-size:1.3rem; margin:2.6rem 0 .9rem; padding-bottom:.35rem;
     border-bottom:1px solid var(--line); }
h3 { font-size:1.05rem; margin:1.8rem 0 .6rem; color:var(--accent); }
.meta { color:var(--muted); font-size:.9rem; margin-bottom:2rem; }
p, li { margin:0 0 .7rem; }
ul, ol { padding-left:1.3rem; }
strong { font-weight:650; }
table { border-collapse:collapse; width:100%; font-size:.92rem; margin:.6rem 0 1.2rem; }
th, td { text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--line); }
th { background:#f3f6f7; font-weight:600; }
code, pre { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88em; }
pre { background:#f3f6f7; border:1px solid var(--line); border-radius:6px;
      padding:.8rem 1rem; overflow-x:auto; }
figure { margin:1.2rem 0 1.8rem; }
figure img { width:100%; height:auto; border:1px solid var(--line); border-radius:6px; }
figcaption { color:var(--muted); font-size:.86rem; margin-top:.4rem; }
.caveat { background:var(--warnbg); border-left:3px solid var(--warn);
          padding:.9rem 1.1rem; border-radius:0 6px 6px 0; margin:1rem 0; }
.judgement { color:var(--muted); font-size:.9rem; font-style:italic;
             border-top:1px solid var(--line); padding-top:.7rem; margin-top:1rem; }
.scroll { overflow-x:auto; }
@media print { body { padding:0; } h2 { page-break-after:avoid; } figure { page-break-inside:avoid; } }
"""


def _embed(path: str) -> Optional[str]:
    full = os.path.join(WORKSPACE_DIR, path)
    try:
        if os.path.getsize(full) > CHART_MAX_BYTES:
            return None
        with open(full, "rb") as fh:
            return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")
    except OSError:
        return None


def _md_inline(text: str) -> str:
    """Minimal inline markdown → HTML: **bold** and `code`, everything escaped."""
    out = html.escape(str(text))
    while out.count("**") >= 2:
        out = out.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
    while out.count("`") >= 2:
        out = out.replace("`", "<code>", 1).replace("`", "</code>", 1)
    return out


def _build_html(*, title, dataset, df, situation, complication, takeaways, next_steps,
                charts, manifest, source, dashboard_url) -> str:
    today = datetime.date.today().isoformat()
    rows, cols = df.shape
    p = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         f"<title>{html.escape(title)}</title><style>{_HTML_CSS}</style></head><body><main>",
         f"<h1>{html.escape(title)}</h1>",
         f"<p class='meta'>{today} · prepared from <code>{html.escape(dataset)}</code> "
         f"({rows:,} rows × {cols} columns)</p>"]

    p.append("<h2>Background</h2>")
    p.append(f"<p>{_md_inline(situation)}</p>")
    if source:
        p.append(f"<p class='meta'>Source data: <code>{html.escape(str(source))}</code></p>")
    if dashboard_url:
        safe = html.escape(dashboard_url, quote=True)
        p.append(f"<p>Live charts: <a href='{safe}'>{safe}</a></p>")

    kpis = _kpi_rows(df, dataset)
    if kpis:
        p.append("<h2>Key figures</h2><div class='scroll'><table>"
                 "<tr><th>Metric</th><th>Value</th></tr>")
        p += [f"<tr><td>{html.escape(str(label))}</td><td>{html.escape(str(value))}</td></tr>"
              for label, value in kpis]
        p.append("</table></div>")

    for heading, header, body, note in _ranking_tables():
        p.append(f"<h3>{html.escape(heading)}</h3><div class='scroll'><table><tr>"
                 + "".join(f"<th>{html.escape(h)}</th>" for h in header) + "</tr>")
        p += ["<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
              for row in body]
        p.append("</table></div>")
        if note:
            p.append(f"<p class='meta'>{html.escape(note)}</p>")

    p.append("<h2>Key takeaways</h2>")
    if complication:
        p.append(f"<p><strong>The complication.</strong> {_md_inline(complication)}</p>")
    if takeaways:
        p.append("<ul>" + "".join(f"<li>{_md_inline(t)}</li>" for t in takeaways) + "</ul>")
    if next_steps:
        p.append("<h3>What I propose next</h3>")
        p.append("<ol>" + "".join(f"<li>{_md_inline(s)}</li>" for s in next_steps) + "</ol>")
        p.append("<p class='judgement'>These next steps are my recommendation — judgement, not "
                 "measurement. Everything above comes from the data; this is what I think it "
                 "means to do about it.</p>")

    p.append("<h2>Methodology</h2>")
    if manifest:
        before = manifest.get("rows_before", "?")
        before = f"{before:,}" if isinstance(before, int) else "?"
        ops = manifest.get("operations", [])
        p.append(f"<p>The raw data had {before} rows. Cleaning was proposed by the tool, approved "
                 f"by a human, and applied in a fixed order. {len(ops)} operation(s) ran:</p>")
        p.append("<ul>" + "".join(
            f"<li>{html.escape(str(o.get('label', o.get('action', ''))))}</li>"
            for o in ops[:20]) + "</ul>")
    else:
        p.append("<p>No cleaning record was found for this dataset — it was analysed as loaded.</p>")

    limits = _limitations(df, dataset)
    invented = _invented_groups()
    if limits or invented:
        p.append("<h3>What this data cannot tell you</h3><div class='caveat'>"
                 "<p>Read these before quoting any figure above.</p><ul>")
        p += [f"<li>{_md_inline(n)}</li>" for n in limits]
        for column, groups in invented.items():
            shown = html.escape(", ".join(groups[:6]) + ("…" if len(groups) > 6 else ""))
            p.append(f"<li><strong>{html.escape(column)}</strong> — these groups are entirely "
                     f"filled-in values, so their figures are the fill value rather than anything "
                     f"observed: {shown}</li>")
        p.append("</ul></div>")

    p.append("<h2>Appendix</h2>")
    measured = _measured_numbers()
    if measured:
        p.append("<h3>Relationships measured</h3><div class='scroll'><table>"
                 "<tr><th>Pair</th><th>Correlation</th><th>Strength</th></tr>")
        p += [f"<tr><td>{html.escape(a)}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in measured]
        p.append("</table></div><p class='meta'>A correlation shows two things moving together. "
                 "It is not evidence that one causes the other.</p>")
    embedded = 0
    if charts:
        p.append("<h3>Charts</h3>")
        for chart in charts:
            data = _embed(chart)
            caption = html.escape(_chart_caption(chart, dataset))
            if data:
                embedded += 1
                p.append(f"<figure><img alt='{caption}' src='{data}'>"
                         f"<figcaption>{caption}</figcaption></figure>")
            else:
                p.append(f"<p class='meta'>({caption} — chart file unavailable)</p>")
    if manifest:
        p.append("<h3>Reproducing this</h3><p>Every cleaning step was recorded, so this can be "
                 "re-run on a later extract without repeating any approvals:</p>"
                 f"<pre>python -m agent.data_cleaner &lt;newfile.csv&gt; --replay "
                 f"&lt;{html.escape(dataset)}.cleaning.json&gt;</pre>")
    p.append("<h3>Column summary</h3><div class='scroll'><table>"
             "<tr><th>Column</th><th>Type</th><th>Filled</th><th>Distinct</th></tr>")
    for column in df.columns:
        s = df[column]
        p.append(f"<tr><td>{html.escape(str(column))}</td><td>{s.dtype}</td>"
                 f"<td>{int(s.notna().sum()):,}</td>"
                 f"<td>{int(s.nunique(dropna=True)):,}</td></tr>")
    p.append("</table></div></main></body></html>")
    return "\n".join(p)


# ─── the tool ─────────────────────────────────────────────────────────────────

def write_report(dataset: str, situation: str, takeaways: list,
                 complication: str = "", next_steps: Optional[list] = None,
                 title: Optional[str] = None, dashboard_url: Optional[str] = None,
                 output_name: Optional[str] = None) -> str:
    """Write the findings report. Refuses if the narrative contradicts the data."""
    pipe = _pipeline()
    df = pipe.datasets.get(dataset)
    if df is None:
        known = ", ".join(pipe.datasets) or "(none loaded)"
        return f"Error: no dataset named '{dataset}' is loaded. Loaded: {known}"
    if not situation or not str(situation).strip():
        return ("Error: 'situation' is required — restate in one short paragraph why this was "
                "asked for, in the reader's own terms. Without it the report has no context and "
                "reads as a pile of statistics.")
    takeaways = [t for t in (takeaways or []) if str(t).strip()]
    if not takeaways:
        return ("Error: 'takeaways' is required — 3 to 5 sentences saying what you found and why "
                "it matters. Nobody is reading this to see the charts; they are reading it for "
                "your conclusion.")
    next_steps = [s for s in (next_steps or []) if str(s).strip()]

    # the narrative is checked BEFORE it is committed to a document people trust
    narrative = "\n".join([str(situation), str(complication or "")] + takeaways + next_steps)
    try:
        from agent.data_analysis import review_conclusions
        issues = review_conclusions(narrative)
    except Exception:  # noqa: BLE001
        issues = []
    if issues:
        return ("Error: the report was NOT written — its narrative contradicts what was measured. "
                "A polished document is trusted more than terminal output, so a wrong claim inside "
                "one does more damage. Fix these and call write_report again:\n"
                + "\n".join(f"  • {i}" for i in issues))

    manifest, source = _cleaning_record(dataset)
    charts = _charts_for(dataset)
    heading = title or f"{dataset.replace('_', ' ').title()} — findings"
    payload = dict(title=heading, dataset=dataset, df=df, situation=str(situation),
                   complication=str(complication or ""), takeaways=takeaways,
                   next_steps=next_steps, charts=charts, manifest=manifest,
                   source=source, dashboard_url=dashboard_url)

    stem = output_name or f"{dataset}_report"
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    md_path = os.path.join(WORKSPACE_DIR, f"{stem}.md")
    html_path = os.path.join(WORKSPACE_DIR, f"{stem}.html")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_build_markdown(**payload))
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(_build_html(**payload))

    limits = len(_limitations(df, dataset)) + len(_invented_groups())
    return "\n".join([
        f"Report written for '{dataset}':",
        f"  {stem}.md    — for git and review",
        f"  {stem}.html  — self-contained, charts embedded, opens in any browser",
        f"  Structure: Background → Key figures → Key takeaways → Methodology → Appendix",
        f"  {len(takeaways)} takeaway(s), {len(next_steps)} proposed next step(s), "
        f"{len(charts)} chart(s) embedded, {limits} limitation(s) stated.",
        "  The limitations and the numbers were generated from the recorded evidence — "
        "they are not editable from the narrative.",
    ])


_SWARN_TOOLS = {
    "write_report": (
        "Write the findings report a human actually reads and forwards. Use this at the END of an "
        "analysis, before finish_task. Structure is fixed: Background → Key figures → Key takeaways → "
        "Methodology → Appendix, telling a Situation / Complication / Resolution story.\n"
        "YOU write the narrative, in your own voice — nobody is talking to the dashboard, they "
        "are talking to you, so tell the story. 'situation' restates why this was asked in the "
        "reader's terms. 'complication' is what the data complicates about that question. "
        "'takeaways' are 3-5 plain sentences of what you found and why it matters — conclusions, "
        "not statistics. 'next_steps' is the second-degree analysis you propose.\n"
        "The numbers, charts, cleaning record and limitations are generated from recorded "
        "evidence and cannot be edited by you. If your narrative contradicts what was measured "
        "the report is REFUSED with the reasons — fix it and call again.",
        {"type": "object",
         "properties": {
             "dataset": {"type": "string", "description": "The loaded dataset the report is about (usually '<name>_clean')."},
             "situation": {"type": "string", "description": "Background: one short paragraph restating why this was asked, in the reader's terms."},
             "complication": {"type": "string", "description": "What the data complicates about the original question — the reason this is not a one-line answer."},
             "takeaways": {"type": "array", "items": {"type": "string"}, "description": "3-5 plain-language conclusions. Sentences, not statistics."},
             "next_steps": {"type": "array", "items": {"type": "string"}, "description": "The second-degree analysis you propose next."},
             "title": {"type": "string"},
             "dashboard_url": {"type": "string", "description": "Optional link to a live dashboard."},
             "output_name": {"type": "string", "description": "File stem (default '<dataset>_report')."},
         },
         "required": ["dataset", "situation", "takeaways"]},
        write_report,
    ),
}


def register_into_swarn() -> str:
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
    return f"registered {registered} reporting tool(s) into swarn"
