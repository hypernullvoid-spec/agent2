"""
Data Analysis & Visualisation — exploratory analysis over loaded datasets.

Where data_pipeline.py answers "is the data loaded and sound" and
data_cleaner.py answers "what should I fix", this module answers the
analyst's actual questions: what does this column look like, what relates
to what, how does it break down, how is it changing, and is that
difference real.

The blind-agent contract
────────────────────────
An LLM cannot look at a PNG. Every tool here therefore returns BOTH a
saved chart path AND the same finding written out in words — the chart is
for the human, the sentences are what the model reasons about. A tool that
returned only a file path would be useless to the agent. This mirrors
evaluation.py, whose plots already end with a plain-language reading.

Everything is READ-ONLY: no tool in this module ever mutates a loaded
dataset. Derived results (pivots, resamples) are registered as NEW
datasets so they can be charted, cleaned or saved in turn.

Charts land in workspace/plots/ at 120 dpi, matching evaluation.py.
Large frames are sampled for plotting — and the sampling is always stated
on the chart and in the text, never silent.

Pure pandas / numpy / matplotlib / scipy — all already project dependencies.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import pandas as pd

WORKSPACE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "workspace")
)
PLOTS_SUBDIR = "plots"

PLOT_SAMPLE_LIMIT = 20_000      # points beyond this are sampled for legibility
TOP_CATEGORIES = 15             # bars shown for a categorical column
CORR_MIN_ROWS = 5
HIGH_CARDINALITY = 50

# correlation strength wording, so the agent describes it consistently
_STRENGTH_BANDS = ((0.7, "strong"), (0.4, "moderate"), (0.2, "weak"))


# ─── evidence ledger ───────────────────────────────────────────────────────────
#
# Every guard in this module protects a single tool's output. Nothing protected
# the FINAL SUMMARY — so true tool results could be assembled into a false
# narrative ("ratings are declining" three steps after the tool said the
# correlation was -0.02, "none"). The ledger records what was actually measured
# so a conclusion can be checked against it before it reaches the user.

_TREND_WORDS = (
    "declin", "decreas", "fall", "fell", "drop", "shrink", "downward", "worsen",
    "rising", "rise", "rose", "increas", "grow", "growth", "upward", "improv",
    "trend", "trending", "over time", "year on year", "year-on-year",
)
FLAT_CORRELATION = 0.1          # below this the tools already say "none"

# Correctly REPORTING the absence of a trend uses the same vocabulary as
# claiming one. Flagging "no trend was found" would be a false alarm, and false
# alarms are how a checker gets ignored.
_NEGATIONS = (
    "no trend", "not a trend", "no clear trend", "no relationship", "no correlation",
    "no significant", "no pattern", "no evidence", "no meaningful", "not related",
    "unrelated", "essentially flat", "is flat", "remains flat", "stayed flat",
    "(none)", "none)", "no direction", "does not", "doesn't", "did not", "didn't",
    "no apparent", "no discernible", "little or no", "weak or no",
)

_EVIDENCE: dict = {"correlations": {}, "imputed_groups": {}, "rankings": {},
                   "durations": {}, "joins": [], "truncation": {}, "grain": {},
                   "reconciliations": [], "effects": []}


def reset_evidence() -> None:
    _EVIDENCE["correlations"] = {}
    _EVIDENCE["imputed_groups"] = {}
    _EVIDENCE["rankings"] = {}
    _EVIDENCE["durations"] = {}
    _EVIDENCE["joins"] = []
    _EVIDENCE["truncation"] = {}
    _EVIDENCE["grain"] = {}
    _EVIDENCE["reconciliations"] = []
    _EVIDENCE["effects"] = []


def note_correlation(x, y, r) -> None:
    """Record a measured correlation so a later claim can be checked against it."""
    try:
        value = float(r)
    except (TypeError, ValueError):
        return
    if value == value:                     # not NaN
        _EVIDENCE["correlations"][tuple(sorted((str(x), str(y))))] = value


def note_join(record) -> None:
    """Record a join so the report must disclose it.

    A join is the only analysis step that can change every total in the
    dataset without changing anything a reader can see. data_report.py used to
    detect one after the fact by spotting pandas' _x/_y name-clash suffixes —
    which fires only when the two tables happened to share a column name, so a
    clean join stayed invisible and its dropped rows were never mentioned.
    Recording it at the moment it happens is the only way the Methodology
    section can state the keys, the direction and the row-count effect.
    """
    if isinstance(record, dict) and record.get("output"):
        _EVIDENCE.setdefault("joins", []).append(dict(record))


def note_effect(dataset, record) -> None:
    """Record how BIG a difference was, not just whether it was real.

    Held separately from the p-value because the report needs to police a
    specific failure: a narrative calling a negligible-but-significant gap an
    outperformance. Without the effect size on record there is no way to tell
    that claim from a correct one.
    """
    if isinstance(record, dict) and record.get("group_col"):
        _EVIDENCE.setdefault("effects", []).append({"dataset": str(dataset), **record})


def effects_for(dataset) -> list:
    return [e for e in _EVIDENCE.get("effects", []) if e.get("dataset") == str(dataset)]


def note_grain(dataset, record) -> None:
    """Record that a table is not one row per what it claims to be.

    Kept with the other evidence because it changes the meaning of every total
    computed from the table, and because nothing about the frame afterwards
    shows it — a doubled order looks exactly like two orders.
    """
    if isinstance(record, dict) and record.get("keys"):
        _EVIDENCE.setdefault("grain", {})[str(dataset)] = dict(record)


def grain_for(dataset) -> dict:
    """What check_grain found for this table — {} if it was never checked."""
    return _EVIDENCE.get("grain", {}).get(str(dataset), {})


def note_reconciliation(record) -> None:
    """Record a comparison against a figure from outside this analysis."""
    if isinstance(record, dict) and record.get("label"):
        _EVIDENCE.setdefault("reconciliations", []).append(dict(record))


def reconciliations_for(dataset) -> list:
    return [r for r in _EVIDENCE.get("reconciliations", [])
            if r.get("dataset") == str(dataset)]


def note_truncation(dataset, record) -> None:
    """Record that a dataset is only part of its file.

    A partial read is invisible in the frame itself — 2 million rows look
    exactly like 2 million rows whether or not another 3 million were left on
    disk. Nothing downstream can rediscover the shortfall, so the only place it
    can be caught is here, at the moment of reading.
    """
    if isinstance(record, dict) and record.get("rows_total"):
        _EVIDENCE.setdefault("truncation", {})[str(dataset)] = dict(record)


def truncation_for(dataset) -> dict:
    """What was left unread when `dataset` was loaded — {} if it was read whole."""
    return _EVIDENCE.get("truncation", {}).get(str(dataset), {})


def joins_for(dataset) -> list:
    """Every recorded join that produced `dataset`, newest last."""
    return [j for j in _EVIDENCE.get("joins", []) if j.get("output") == str(dataset)]


def note_imputed_groups(column, group_labels) -> None:
    """Record that some groups of `column` are entirely invented values."""
    labels = [str(g) for g in group_labels]
    if labels:
        _EVIDENCE["imputed_groups"].setdefault(str(column), []).extend(labels)


def review_conclusions(summary: str) -> list:
    """Check a written summary against what was actually measured.

    Deliberately narrow: it only reports contradictions it can prove from the
    ledger. A checker that guessed would bury the real warnings.
    """
    issues: list = []
    if not summary:
        return issues
    lines = [ln for ln in summary.split("\n") if ln.strip()]

    flat = {}
    for (a, b), r in _EVIDENCE["correlations"].items():
        if abs(r) < FLAT_CORRELATION:
            flat.setdefault(a, []).append((b, r))
            flat.setdefault(b, []).append((a, r))

    for column, partners in flat.items():
        low = column.lower()
        for line in lines:
            text = line.lower()
            if low not in text or not any(w in text for w in _TREND_WORDS):
                continue
            if any(neg in text for neg in _NEGATIONS):
                continue                       # correctly reporting no trend
            detail = ", ".join(f"vs {p}: {r:+.2f}" for p, r in partners[:3])
            issues.append(
                f"CONTRADICTION: you describe a trend in '{column}' — \"{line.strip()[:110]}\" — "
                f"but the correlation actually measured was flat ({detail}), which the tool "
                f"reported as 'none'. Either drop the trend claim or say plainly that no "
                f"relationship was found.")
            break

    for column, labels in _EVIDENCE["imputed_groups"].items():
        low = column.lower()
        unique = sorted(set(labels))
        for line in lines:
            text = line.lower()
            if any(neg in text for neg in _NEGATIONS):
                continue
            if low in text and any(w in text for w in _TREND_WORDS):
                issues.append(
                    f"INVENTED VALUES: you describe a trend in '{column}', but "
                    f"{len(unique)} group(s) ({', '.join(unique[:5])}) are 100% imputed — their "
                    f"figures are the fill value, not measurements. Any trend that leans on "
                    f"them is an artefact of cleaning. Remove those groups or say so.")
                break

    issues += _check_ranking_claims(lines)
    issues += _check_share_claims(lines)
    issues += _check_duration_claims(lines)
    issues += _check_pairing_claims(lines)
    issues += _check_noise_claims(lines)
    return issues


# A ranking whose groups are indistinguishable from chance still prints in a
# perfect 1-2-3 order, and reads exactly like a real finding. "Peak ordering
# hours are 19:00-21:00" and "test evening flash sales" both came out of a table
# whose 24 groups were flat at p = 0.45. The ledger now records that verdict, so
# a superlative claim about such a dimension can be refused.

_SUPERLATIVES = (
    "peak", "highest", "lowest", "top ", "leads", "lead ", "leading", "best",
    "worst", "strongest", "weakest", "outperform", "dominat", "drives", "driver",
    "concentrat", "disproportionate", "spike", "trough", "prefer", "favour",
    "favor", "most valuable", "biggest", "largest", "smallest",
)

_NOISE_SAFE = _NEGATIONS + (
    "noise", "not significant", "indistinguishable", "chance", "random",
    "sampling", "flat", "no difference", "nearly equal", "roughly equal",
    "broadly equal", "similar", "equally",
)


def _mentions_dimension(column: str, text: str) -> bool:
    """'Order_Hour' is written as 'hours' in prose, so the trailing word counts too.

    Kept local to the noise check rather than widened into _column_aliases, which
    the ranking and share checkers rely on being strict.
    """
    if _mentions_column(column, text):
        return True
    words = [w for w in re.split(r"[^0-9a-zA-Z]+", str(column)) if len(w) >= 4]
    if not words:
        return False
    tail = words[-1].lower()
    return _mentions(tail, text) or _mentions(tail + "s", text) or _mentions(tail + "ly", text)


def _mentions_label(label: str, text: str) -> bool:
    """Hour buckets are labelled '19' and '20' — below the length floor that stops
    short words matching inside longer ones, but unambiguous when they are digits."""
    if label.isdigit() and len(label) >= 2:
        return re.search(rf"(?<![\w]){re.escape(label)}(?![\w])", text) is not None
    return _mentions(label, text)


def _check_noise_claims(lines: list) -> list:
    issues: list = []
    for (group_col, value_col, how), record in _EVIDENCE.get("rankings", {}).items():
        p_counts, p_values = record.get("p_counts"), record.get("p_values")
        if p_counts is None or p_counts <= 0.05:
            continue
        if value_col != "row count" and how != "count":
            if p_values is None or p_values <= 0.05:
                continue          # the per-row value is real even if sizes are flat
        labels = [str(x) for x in record.get("order", [])][:3]
        for line in lines:
            text = line.lower()
            if not any(word in text for word in _SUPERLATIVES):
                continue
            if any(safe in text for safe in _NOISE_SAFE):
                continue          # correctly reporting the null result
            if not (_mentions_dimension(str(group_col), line)
                    or any(_mentions_label(l, line) for l in labels)):
                continue
            detail = f"chi-square p = {p_counts:.3g} on group sizes"
            if p_values is not None:
                detail += f", Kruskal-Wallis p = {p_values:.3g} on {value_col} per row"
            issues.append(
                f"RANKS NOISE: you describe a winner in '{group_col}' — \"{line.strip()[:110]}\" — "
                f"but that ranking is not distinguishable from chance ({detail}). The order exists "
                f"because something has to be printed first, not because the groups differ. Drop "
                f"the claim or state plainly that '{group_col}' shows no real difference."
            )
            break
    return issues


# Two separate league tables do not tell you what happens where they cross. The
# top category and the top occasion are each read off their own ranking, then
# joined in a sentence — "stock Colors ahead of Anniversary" — and the pairing is
# never computed. In a real report that recommendation was exactly backwards:
# Colors is the WORST-selling category at Anniversary. The marginals cannot show
# this; the cross-tab always can, and it is one groupby away.

_PAIRING_WORDS = (
    " for ", " before ", " ahead of ", " during ", " at ", " in ", " on ", " where ",
    " drives", " driven by", " leads", " lead ", " promote", " push", " stock",
    " front-load", " front load", " bundle", " target", " prioriti",
)


def _pairing_candidates() -> list:
    """Ranked dimensions that can be crossed: same dataset, same additive measure."""
    out = []
    for (group_col, value_col, how), record in _EVIDENCE.get("rankings", {}).items():
        if how not in ("sum", "count", "mean"):
            continue
        out.append((str(group_col), str(value_col), str(how), str(record.get("dataset", "")),
                    [str(x) for x in record.get("order", [])]))
    return out


def _check_pairing_claims(lines: list) -> list:
    """Flag a sentence that pairs a value of one dimension with a value of another
    when the cross-tab says that pairing is weak."""
    issues: list = []
    candidates = _pairing_candidates()
    seen: set = set()
    cache: dict = {}

    for i, (dim_a, value_col, how, dataset, labels_a) in enumerate(candidates):
        for dim_b, _v, _h, dataset_b, labels_b in candidates[i + 1:]:
            if dim_a == dim_b or dataset != dataset_b or not dataset:
                continue
            df, err = _get_df(dataset)
            if err or dim_a not in df.columns or dim_b not in df.columns:
                continue
            cross = _cross_tab(cache, df, dim_a, dim_b, value_col, how, dataset)
            if cross is None:
                continue
            for line in lines:
                for unit in _pairing_units(line, labels_a, labels_b):
                    text = " " + unit.lower() + " "
                    if not any(w in text for w in _PAIRING_WORDS):
                        continue
                    issues += _judge_unit(cross, dim_a, dim_b, value_col,
                                          labels_a, labels_b, unit, dataset, seen)
    return issues


def _cross_tab(cache, df, dim_a, dim_b, value_col, how, dataset):
    key = (dataset, dim_a, dim_b, value_col, how)
    if key not in cache:
        try:
            if value_col == "row count" or how == "count":
                table = df.pivot_table(index=dim_a, columns=dim_b, aggfunc="size", fill_value=0)
            elif value_col not in df.columns:
                table = None
            else:
                table = df.pivot_table(index=dim_a, columns=dim_b, values=value_col,
                                       aggfunc=how, fill_value=0)
        except Exception:  # noqa: BLE001
            table = None
        # A 2-level dimension cannot be ranked within, but it can still be the
        # profile that a 5-level dimension is ranked inside — so the table is kept
        # and each DIRECTION is judged on its own depth by _weak_within.
        if table is not None and (table.empty or max(len(table.index), len(table.columns)) < 3):
            table = None
        cache[key] = table
    return cache[key]


def _pairing_units(line: str, labels_a: list, labels_b: list) -> list:
    """Split a bullet into the separate pairings it actually claims.

    A comma usually separates two recommendations ("stock Colors ahead of Holi,
    and Sweets ahead of Anniversary") — pairing across it invents a Holi/Sweets
    link nobody made. But a comma also just fronts a clause ("For Anniversary,
    stock Sweets"), where splitting would lose the only pairing in the sentence.
    So clauses are re-joined until each unit names both dimensions, and a clause
    that already names both stands alone.
    """
    parts = [p.strip() for p in re.split(r"[,;•]|(?<=[A-Za-z])\.\s+", line) if p.strip()]
    units, buffer = [], []
    for part in parts:
        buffer.append(part)
        joined = ", ".join(buffer)
        has_a = any(_mentions(l, joined) for l in labels_a)
        has_b = any(_mentions(l, joined) for l in labels_b)
        if has_a and has_b:
            units.append(joined)
            buffer = []
    return units


def _label_position(label: str, text: str) -> Optional[int]:
    match = re.search(rf"(?<![\w]){re.escape(label)}(?![\w])", text, re.IGNORECASE)
    return match.start() if match else None


def _nearest(label: str, others: list, text: str) -> Optional[str]:
    """The label of the other dimension this one is actually written next to."""
    here = _label_position(label, text)
    if here is None:
        return None
    ranked = []
    for other in others:
        there = _label_position(other, text)
        if there is not None:
            ranked.append((abs(there - here), other))
    return min(ranked)[1] if ranked else None


def _weak_within(column, label):
    """(position, levels, leader) when `label` is a laggard in this profile."""
    ordered = column.sort_values(ascending=False)
    levels = len(ordered)
    if levels < 3 or label not in ordered.index:
        return None
    position = int(list(ordered.index).index(label)) + 1
    if position <= max(3, (levels + 1) // 2):
        return None
    return position, levels, ordered.index[0], float(ordered.loc[label]), float(ordered.iloc[0])


def _judge_unit(cross, dim_a, dim_b, value_col, labels_a, labels_b, unit, dataset, seen) -> list:
    """Judge each claimed pairing from BOTH sides of the cross-tab.

    "Stock Colors for Anniversary" is wrong if Colors is a laggard among the
    categories bought at Anniversary, and equally wrong if Anniversary is a
    laggard among the occasions Colors sells on. Testing only one orientation
    misses half the bad recommendations.
    """
    issues: list = []
    hit_a = [l for l in labels_a if _mentions(l, unit)]
    hit_b = [l for l in labels_b if _mentions(l, unit)]
    if not hit_a or not hit_b:
        return issues

    pairs = set()
    for label_a in hit_a:
        partner = _nearest(label_a, hit_b, unit)
        if partner is not None:
            pairs.add((label_a, partner))
    for label_b in hit_b:
        partner = _nearest(label_b, hit_a, unit)
        if partner is not None:
            pairs.add((partner, label_b))

    for label_a, label_b in sorted(pairs):
        # Same word in two columns ('Raksha Bandhan' as both a Category and an
        # Occasion) is a data-quality problem, not a claimed pairing.
        if label_a == label_b:
            continue
        if label_a not in cross.index or label_b not in cross.columns:
            continue
        key = (dim_a, label_a, dim_b, label_b)
        if key in seen:
            continue
        verdict = _weak_within(cross[label_b], label_a)
        held, within, other = dim_a, dim_b, label_b
        laggard, leader_of = label_a, label_b
        if verdict is None:
            verdict = _weak_within(cross.loc[label_a], label_b)
            held, within, other = dim_b, dim_a, label_a
            laggard, leader_of = label_b, label_a
        if verdict is None:
            continue
        seen.add(key)
        position, levels, best, value, top = verdict
        issues.append(
            f"PAIRING NOT SUPPORTED: you link '{label_a}' with '{label_b}' — "
            f"\"{unit.strip()[:110]}\" — but crossing '{dim_a}' with '{dim_b}' shows {laggard} "
            f"ranks {position} of {levels} within {leader_of} ({_fmt(value)}), while {best} leads "
            f"it ({_fmt(top)}). '{label_a}' and '{label_b}' each top their OWN ranking; that is a "
            f"fact about two separate totals and says nothing about what happens where they meet. "
            f"Run pivot_dataset('{dataset}', rows=['{dim_a}'], columns=['{dim_b}'], "
            f"values='{value_col}') and pair them on what it shows."
        )
    return issues


# A ranking claim is the easiest thing in a report to get wrong and the hardest
# to notice: "top 5 products" with the 2nd-placed one quietly missing reads
# exactly like a correct list. The ledger knows the real order, so check it.

_TOP_N = re.compile(r"\btop[\s-]+(\d{1,3})\b", re.IGNORECASE)
_PERCENT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")


def _mentions(label: str, text: str) -> bool:
    """Whole-label match, so 'Cake' does not match inside 'Cakes and Bakes'."""
    if len(label) < 3:
        return False
    return re.search(rf"(?<![\w]){re.escape(label)}(?![\w])", text, re.IGNORECASE) is not None


def _column_aliases(column: str) -> list:
    """How a person would write a column name in prose.

    Nobody writes 'the top 20% of Customer_ID'; they write 'customers'. A
    checker that only matched the literal column name would silently pass every
    claim written in English, which is all of them.
    """
    raw = str(column)
    spaced = re.sub(r"[_\-]+", " ", raw).strip()
    words = spaced.split()
    if len(words) > 1 and words[-1].lower() in ("id", "ids", "name", "names", "code", "key"):
        words = words[:-1]                     # Customer_ID → customer
    stem = " ".join(words) or spaced
    aliases = {raw.lower(), spaced.lower(), stem.lower()}
    for base in list(aliases):
        aliases.add(base + "es" if base.endswith(("s", "x", "ch", "sh")) else base + "s")
    return [a for a in aliases if len(a) >= 3]


def _mentions_column(column: str, text: str) -> bool:
    return any(_mentions(alias, text) for alias in _column_aliases(column))


def _check_ranking_claims(lines: list) -> list:
    """Flag a 'top N' list naming something the recorded ranking puts outside N."""
    issues = []
    rankings = _EVIDENCE.get("rankings", {})
    if not rankings:
        return issues
    by_column: dict = {}
    for (group_col, value_col, how), record in rankings.items():
        by_column.setdefault(group_col, []).append((value_col, how, record))

    for group_col, records in by_column.items():
        for line in lines:
            match = _TOP_N.search(line)
            if not match:
                continue                       # no stated scope — nothing to prove
            claimed = int(match.group(1))
            if claimed < 1 or claimed > 100:
                continue
            # Every named item must be provably outside the top N in EVERY
            # recorded ranking for this column — otherwise the narrator may
            # simply be ranking by a measure we did not record, and a checker
            # that guesses gets ignored.
            offenders: dict = {}
            for value_col, how, record in records:
                order = record["order"]
                named = [lbl for lbl in order if _mentions(lbl, line)]
                if len(named) < 2:
                    continue                   # a passing mention, not a list
                outside = {lbl: order.index(lbl) + 1 for lbl in named
                           if order.index(lbl) + 1 > claimed}
                if not outside:
                    offenders = {}
                    break                      # consistent with this ranking — done
                offenders[(value_col, how)] = (outside, order[:claimed])
            if not offenders:
                continue
            (value_col, how), (outside, truth) = next(iter(offenders.items()))
            wrong = ", ".join(f"'{lbl}' is actually rank {rank}" for lbl, rank in outside.items())
            issues.append(
                f"WRONG RANKING: you present a top {claimed} for '{group_col}' — "
                f"\"{line.strip()[:110]}\" — but the measured ranking by {how} of {value_col} "
                f"says {wrong}. The real top {claimed} is: {', '.join(truth)}. "
                f"Use those, or say which measure you are ranking by.")
            break
    return issues


SHARE_TOLERANCE = 5.0            # percentage points; below this it is rounding

# Only a claim shaped 'the top X% ... account for Y% of <the measure>' is a
# concentration claim. '30% of customers are male and 70% reordered' contains
# two percentages and a column name but asserts nothing about concentration —
# flagging it would be the guess that gets the whole checker ignored.
_TOP_SHARE = re.compile(r"\btop\s+(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE)
_SHARE_OF_TOTAL = ("of revenue", "of total", "of sales", "of all", "of the total",
                   "of turnover", "of spend", "of the revenue")


def _check_share_claims(lines: list) -> list:
    """Flag 'the top 20% generate 40% of revenue' when the ledger says otherwise.

    Concentration claims carry real weight in a recommendation ("build a loyalty
    programme for them"), and they are almost never re-derived by the reader.
    """
    issues = []
    for (group_col, value_col, how), record in _EVIDENCE.get("rankings", {}).items():
        total, order = record.get("total"), record["order"]
        if not total or record.get("ascending") or len(order) < 5:
            continue
        values = record["values"]
        for line in lines:
            if not _mentions_column(group_col, line):
                continue
            head = _TOP_SHARE.search(line)
            if not head:
                continue                       # not a concentration claim
            text = line.lower()
            if not (_mentions_column(value_col, line)
                    or any(phrase in text for phrase in _SHARE_OF_TOTAL)):
                continue                       # the second % is about something else
            rest = _PERCENT.findall(line[head.end():])
            if not rest:
                continue
            share_of_group, claimed_share = float(head.group(1)), float(rest[0])
            if not 0 < share_of_group < 100:
                continue
            cutoff = max(1, round(len(order) * share_of_group / 100))
            actual = 100 * sum(values[lbl] for lbl in order[:cutoff]) / total
            if abs(actual - claimed_share) <= SHARE_TOLERANCE:
                continue
            issues.append(
                f"UNSUPPORTED SHARE: you claim \"{line.strip()[:110]}\" — but the measured "
                f"{how} of {value_col} says the top {share_of_group:g}% of '{group_col}' "
                f"({cutoff:,} of {len(order):,}) account for {actual:.1f}%, not "
                f"{claimed_share:g}%. Use the measured figure.")
            break
    return issues


DURATION_TOLERANCE = 0.1         # 10% off the measured average is a different number


def _check_duration_claims(lines: list) -> list:
    """Flag a stated average elapsed time that contradicts the measured one."""
    issues = []
    for (start_col, end_col), record in _EVIDENCE.get("durations", {}).items():
        unit, mean = record["unit"], record["mean"]
        tokens = [t for t in re.split(r"[\s_]+", f"{start_col} {end_col}") if len(t) > 3]
        for line in lines:
            text = line.lower()
            if unit not in text or not any(t.lower() in text for t in tokens):
                continue
            numbers = [float(n) for n in _NUMBER.findall(line.replace(",", ""))]
            candidates = [n for n in numbers if n > 0]
            if not candidates:
                continue
            if any(abs(n - mean) <= abs(mean) * DURATION_TOLERANCE for n in candidates):
                continue
            issues.append(
                f"WRONG DURATION: you state a figure in {unit} for the gap between "
                f"'{start_col}' and '{end_col}' — \"{line.strip()[:110]}\" — but the measured "
                f"average is {mean:,.2f} {unit} (median {record['median']:,.2f}, "
                f"{record['count']:,} rows). Quote the measured figure.")
            break
    return issues


# ─── small helpers ─────────────────────────────────────────────────────────────

def _plots_dir() -> str:
    d = os.path.join(WORKSPACE_DIR, PLOTS_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def _save_fig(fig, filename: str) -> str:
    path = os.path.join(_plots_dir(), filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return os.path.join(PLOTS_SUBDIR, filename)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "col"


def _new_figure(width: float = 7.0, height: float = 4.4):
    """Headless figure — no window is ever opened (server/terminal safe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width, height))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def _get_df(name: str):
    """Return (df, None) or (None, error_string) — never raises."""
    from agent.ml.data_pipeline import get_data_pipeline
    pipe = get_data_pipeline()
    df = pipe.datasets.get(name)
    if df is None:
        known = ", ".join(pipe.datasets) or "(none loaded)"
        return None, f"Error: no dataset named '{name}' is loaded. Loaded: {known}"
    return df, None


def _register(name: str, df: pd.DataFrame) -> None:
    from agent.ml.data_pipeline import get_data_pipeline
    get_data_pipeline().datasets[name] = df


def _is_numeric(s: pd.Series) -> bool:
    return s.dtype.kind in "iufc"


# ── numbers that are labels, not measures ──────────────────────────────────────
# HTTP statuses (200/404/500), postcodes, store numbers and error codes are
# stored as integers, so every naive profiler averages them, correlates them and
# offers to plot their distribution. The mean HTTP status is 300, which is not a
# fact about anything. Decided from the VALUES — a name-based rule ('status',
# 'code', 'zip') would only work for English column names, and would still miss
# a column called 'resp' or '応答'.

CODE_MAX_LEVELS = 25            # beyond this many distinct values it is a measure
CODE_MAX_SHARE = 0.05           # distinct values as a share of rows
CODE_GAP_RATIO = 5.0            # span-per-distinct-value that marks a coded set
_PLAUSIBLE_YEARS = (1800, 2100)


# Column names that state outright that the numbers are labels. Deliberately
# excludes 'level', 'grade', 'score', 'rating' — those are genuinely ordinal
# and their averages mean something.
_CODE_NAME_SUFFIXES = ("_code", "code", "_status", "status", "_flag", "flag",
                       "_type", "type", "_category", "category", "_class", "class")


def _looks_like_coded_categorical(s: pd.Series) -> bool:
    """True when an integer column is a set of LABELS rather than a quantity.

    The signal is the shape of the value set, not its name: a handful of distinct
    integers scattered across a range far wider than their count. Counts, ratings
    and Likert scales (1-5) sit densely in their range and are therefore left
    alone — averaging those IS meaningful.
    """
    if not _is_numeric(s):
        return False
    valid = pd.to_numeric(s, errors="coerce").dropna()
    n = len(valid)
    if n < 10:
        return False                      # too little to judge; leave it numeric
    if not bool((valid == valid.round()).all()):
        return False                      # fractions are measurements, not codes
    levels = int(valid.nunique())
    if levels < 2 or levels > CODE_MAX_LEVELS:
        return False
    if levels > max(2, CODE_MAX_SHARE * n):
        return False                      # too varied to be a small code set

    # The spread is measured over values that RECUR, not over every value seen.
    # Measuring min-to-max lets a single mistyped entry decide the answer: a
    # quantity column of 1-15 with one 9,999 in it spans 9,998, clears the gap
    # test on that alone, and gets reclassified as a code — after which its
    # average is declared meaningless and it is dropped from the correlation
    # map. One fat-finger entry silently removes a real measure from the
    # analysis, which is the opposite of what this guard is for. A genuine code
    # recurs; a typo does not, so values seen once or twice are excluded.
    counts = valid.value_counts()
    support = max(2, int(0.001 * n))
    recurring = counts[counts >= support]
    if len(recurring) < 2:
        return False
    lo, hi = float(recurring.index.min()), float(recurring.index.max())
    levels = int(len(recurring))

    if lo >= _PLAUSIBLE_YEARS[0] and hi <= _PLAUSIBLE_YEARS[1] and hi - lo <= 200:
        return False                      # a span of years is ordinal, not a code

    # Densely-packed small integers (1-5 ratings, 1-3 statuses) fail the gap
    # test by design, because averaging a Likert scale IS meaningful and the
    # shape alone cannot tell a rating from a status. The NAME can, and it is
    # the only evidence available, so a column explicitly called a code or a
    # status is taken at its word.
    if str(s.name or "").lower().endswith(_CODE_NAME_SUFFIXES):
        return True
    return (hi - lo) >= CODE_GAP_RATIO * levels


def _coded_columns(df: pd.DataFrame, columns: Optional[list] = None) -> list:
    """The subset of `columns` that hold codes rather than measurements."""
    cols = df.columns if columns is None else columns
    return [c for c in cols if c in df.columns and _looks_like_coded_categorical(df[c])]


def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


# columns the cleaner adds to record what IT did — bookkeeping, not subject data
_MARKER_SUFFIXES = ("_was_missing", "_was_capped", "_valid")


def _is_marker(col) -> bool:
    """True for a cleaning-bookkeeping column.

    These must stay out of correlation analysis: 'RATING_was_capped' correlates
    with 'RATING' only because capping changed those very rows. Reporting that
    as a finding is a tautology dressed up as insight.
    """
    return str(col).endswith(_MARKER_SUFFIXES)


MIN_GROUP_ROWS = 5              # groups smaller than this cannot be ranked meaningfully
MULTIVALUE_SEPARATORS = (",", ";", "|")
MULTIVALUE_MAX_VOCAB = 100      # a tag vocabulary is small; free text is not
MULTIVALUE_HARD_LIMIT = 2000     # beyond this the column is names or prose, not tags
PIVOT_MAX_LEVELS = 60            # a grid wider than this is unreadable text, not a table


def _split_values(value, sep: str = ",") -> list:
    return [part.strip() for part in str(value).split(sep) if part.strip()]


def _detect_multivalue(series: pd.Series) -> tuple[Optional[str], int, int]:
    """Spot a column whose cells hold LISTS: 'action, adventure, comedy'.

    Treated as atomic categories such a column looks like it has 500 values;
    split, it has 27. Every per-category comparison is meaningless until it is
    split. Returns (separator, distinct-combinations, distinct-single-values).
    """
    strings = [v for v in series.dropna() if isinstance(v, str)]
    if len(strings) < 10:
        return None, 0, 0
    for sep in MULTIVALUE_SEPARATORS:
        share = sum(1 for s in strings if sep in s) / len(strings)
        if share < 0.5:
            continue
        combos = len(set(strings))
        singles = {p for s in strings for p in _split_values(s, sep)}
        # a large vocabulary means free prose or names, not tags
        if not singles or len(singles) > MULTIVALUE_MAX_VOCAB:
            continue
        # a genuine list has a VARIABLE number of parts; a compound value like
        # 'Mumbai, MH' always has exactly two
        part_counts = {len(_split_values(s, sep)) for s in strings}
        if len(part_counts) >= 2 or combos >= 2 * len(singles):
            return sep, combos, len(singles)
    return None, 0, 0


def _clean_label(value, width: int = 42) -> str:
    """Flatten a category label for a chart axis. Real-world text columns carry
    embedded newlines (a STARS field holding 'Director:\\n...\\nStars:\\n...'),
    which overlap into an unreadable mess on a bar or box plot."""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return "(blank)"
    return text[:width - 1] + "…" if len(text) > width else text


def _cleaning_warnings(df: pd.DataFrame, column: str) -> list:
    """Read the cleaner's own marker columns and say how much of this column is
    invented. Without this, an imputed column shows a huge artificial spike at
    the median and gets reported as a real feature of the data."""
    notes = []
    miss = f"{column}_was_missing"
    if miss in df.columns and _is_numeric(df[miss]):
        frac = float(df[miss].mean())
        if frac >= 0.02:
            filled = df.loc[df[miss] == 1, column]
            value = f" (all set to {_fmt(filled.mode().iloc[0])})" if len(filled) and not filled.mode().empty else ""
            notes.append(f"WARNING: {frac:.0%} of these values were BLANK and were filled in by "
                         f"cleaning{value}. That creates a spike which is not real — do not read "
                         f"it as a pattern, and treat the average with suspicion.")
    capped = f"{column}_was_capped"
    if capped in df.columns and _is_numeric(df[capped]):
        frac = float(df[capped].mean())
        if frac >= 0.01:
            notes.append(f"WARNING: {frac:.0%} of these values were extreme and were rewritten to "
                         f"the edge of the normal range, so the pile-up at each end is an artefact.")
    return notes


def _is_texty(s: pd.Series) -> bool:
    """Text column. pandas 3 gives these dtype 'str', not 'object', so a bare
    `dtype == object` check silently misses every string column."""
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _as_datetime(s: pd.Series, note: Optional[list] = None) -> pd.Series:
    """Parse to datetimes with ONE convention for the whole column.

    Two silent corruptions this guards against:

    1. pandas reads a numeric column as nanoseconds since 1970, so a YEAR column
       of 2020.0 silently becomes 1970-01-01 00:00:00.000002020 — every row lands
       in 1970 and any trend over it is fiction.

    2. `format="mixed"` re-decides the layout for EVERY ROW independently. Handed
       a DD-MM-YYYY column it reads '24-02-2023' as 24 February (no month is 24)
       but '07-11-2023' as 11 July — scattering a chunk of the rows into the
       wrong month while parsing 100% of them without one complaint. A date
       column has ONE convention: detect it once, apply it to every row.

    Pass a list as `note` to collect a plain-language description of which
    convention was used and how it was chosen. Anything decided by heuristic
    rather than proven by the data says so, because a guessed date convention
    that is never reported is indistinguishable from a correct one.
    """
    if _is_datetime(s):
        return s
    if _is_numeric(s):
        return pd.Series(pd.NaT, index=s.index)

    import warnings

    def parse(**kwargs) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return pd.to_datetime(s, errors="coerce", **kwargs)
            except (TypeError, ValueError):
                return pd.Series(pd.NaT, index=s.index)

    filled = int(s.notna().sum())
    dayfirst = _detect_dayfirst(s)
    ambiguous_dmy = dayfirst is None and _dmy_shaped(s)

    if dayfirst is None:
        parsed, how = parse(), None
    else:
        parsed, how = parse(dayfirst=dayfirst), dayfirst

    # A consistent convention that cannot read the column is worse than a mixed
    # one — fall back, but never quietly.
    readable = int(parsed.notna().sum())
    if filled and readable < filled * 0.99:
        salvage = parse(format="mixed")
        if int(salvage.notna().sum()) > readable:
            if note is not None:
                note.append(
                    f"WARNING: '{s.name}' does not use one consistent date layout — "
                    f"{filled - readable:,} of {filled:,} value(s) could not be read with a single "
                    f"format, so each row was parsed on its own. Rows written DD-MM-YYYY and "
                    f"MM-DD-YYYY in the same column CANNOT be told apart this way and some are "
                    f"almost certainly in the wrong month. Run uniform_dates on it before "
                    f"trusting any trend."
                )
            return salvage

    if note is not None:
        if how is True:
            note.append(f"Dates in '{s.name}' read day-first (DD-MM-YYYY) — proven by the column "
                        f"itself, which contains day values above 12.")
        elif how is False:
            note.append(f"Dates in '{s.name}' read month-first (MM-DD-YYYY) — proven by the column "
                        f"itself, which contains month-position values above 12.")
        elif ambiguous_dmy:
            note.append(f"CAUTION: every date in '{s.name}' is ambiguous (no value has a day above "
                        f"12), so nothing in the column proves the layout. It was read MONTH-FIRST, "
                        f"the pandas default. If this data is Indian/European DD-MM-YYYY, every "
                        f"month below is wrong — confirm the source convention before reporting.")
    return parsed


def _dmy_shaped(s: pd.Series) -> int:
    """How many values look like d-m-y / m-d-y — the layouts that can be read two
    ways. ISO '2023-02-24' is not among them, so it never triggers a warning."""
    if _is_datetime(s) or _is_numeric(s):
        return 0
    return int(sum(1 for v in s.dropna().astype(str).head(2000) if _DMY.match(v)))


def _sample_for_plot(df: pd.DataFrame, limit: int = PLOT_SAMPLE_LIMIT):
    """Down-sample big frames so charts stay legible. Returns (frame, note);
    the note is non-empty whenever sampling happened and is ALWAYS surfaced —
    a silently sampled chart reads as if it showed everything."""
    if len(df) <= limit:
        return df, ""
    sampled = df.sample(limit, random_state=0).sort_index()
    return sampled, f"sampled {limit:,} of {len(df):,} rows for the chart"


def _strength(r: float) -> str:
    a = abs(r)
    for threshold, word in _STRENGTH_BANDS:
        if a >= threshold:
            return word
    return "none"


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4g}"
    return str(value)


def _outlier_count(s: pd.Series) -> tuple[int, float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0, float(q1), float(q3)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum()), float(lo), float(hi)


# ─── 1. one column ─────────────────────────────────────────────────────────────

def plot_column(name: str, column: str, bins: int = 30, top: int = TOP_CATEGORIES,
                log_scale: bool = False) -> str:
    """Chart + written reading of a single column's shape."""
    df, err = _get_df(name)
    if err:
        return err
    if column not in df.columns:
        return f"Error: column '{column}' not in '{name}'. Columns: {list(df.columns)}"

    series = df[column]
    nn = series.dropna()
    nulls = int(series.isnull().sum())
    if not len(nn):
        return f"'{column}' in '{name}' is entirely empty ({nulls} nulls) — nothing to plot."

    lines: list[str] = []
    plot_df, sample_note = _sample_for_plot(df[[column]].dropna())
    values = plot_df[column]

    if _is_numeric(series):
        fig, ax = _new_figure()
        bin_count = min(bins, max(5, int(values.nunique())))
        spread = float(nn.max()) / float(nn.median()) if float(nn.median()) > 0 else 0
        use_log = bool(log_scale) and bool((values > 0).all())
        if use_log:
            # a linear axis on a column spanning orders of magnitude squashes
            # everything into the first bar and shows nothing
            edges = np.logspace(np.log10(values.min()), np.log10(values.max()), bin_count)
            ax.hist(values, bins=edges, color="#3E7CB1", edgecolor="white", linewidth=0.6)
            ax.set_xscale("log")
        else:
            ax.hist(values, bins=bin_count, color="#3E7CB1", edgecolor="white", linewidth=0.6)
        ax.set_xlabel(column + (" (log scale)" if use_log else ""))
        ax.set_ylabel("rows")
        ax.set_title(f"Distribution of '{column}'" + (f"  ({sample_note})" if sample_note else ""))
        rel = _save_fig(fig, f"{_safe_name(name)}__{_safe_name(column)}"
                             f"{'_log' if use_log else ''}_distribution.png")

        q = nn.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
        mean, median = float(nn.mean()), float(nn.median())
        n_out, lo, hi = _outlier_count(nn)
        skew = "roughly symmetric"
        if mean > median * 1.1:
            skew = "leans high — a few large values pull the average up"
        elif mean < median * 0.9:
            skew = "leans low — a few small values pull the average down"
        lines = [
            f"Histogram of '{column}' saved to {rel}",
            f"  {len(nn):,} values, {nulls:,} blank.",
            f"  Smallest {_fmt(q[0])}, typical (middle) {_fmt(q[2])}, largest {_fmt(q[4])}.",
            f"  Half the rows sit between {_fmt(q[1])} and {_fmt(q[3])}. Average {_fmt(mean)}.",
            f"  Shape: {skew}.",
        ]
        if n_out:
            lines.append(f"  {n_out:,} value(s) fall outside the usual range "
                         f"[{_fmt(lo)}, {_fmt(hi)}] — worth a look before averaging.")
        if log_scale and not use_log:
            n_bad = int((values <= 0).sum())
            lines.append(f"  (log_scale was requested but {n_bad:,} value(s) are zero or "
                         f"negative, which a log axis cannot show — drawn on a linear axis. "
                         f"Filter those rows out first if you need the log view.)")
        if spread >= 50 and not use_log:
            lines.append(f"  This column spans {spread:,.0f}x from its middle to its largest "
                         f"value — a naturally long tail, so those high values are probably "
                         f"real rather than errors. Re-run with log_scale=true to actually "
                         f"see the shape; do NOT cap it without checking.")
    elif _is_datetime(series) or (_as_datetime(nn).notna().mean() >= 0.8 and series.dtype != bool):
        parsed = _as_datetime(values).dropna()
        fig, ax = _new_figure()
        ax.hist(parsed, bins=min(bins, max(5, parsed.nunique())), color="#3E7CB1",
                edgecolor="white", linewidth=0.6)
        ax.set_xlabel(column)
        ax.set_ylabel("rows")
        ax.set_title(f"'{column}' over time" + (f"  ({sample_note})" if sample_note else ""))
        fig.autofmt_xdate()
        rel = _save_fig(fig, f"{_safe_name(name)}__{_safe_name(column)}_overtime.png")
        lines = [
            f"Timeline of '{column}' saved to {rel}",
            f"  {len(parsed):,} dated rows, {nulls:,} blank.",
            f"  Runs from {parsed.min():%Y-%m-%d} to {parsed.max():%Y-%m-%d}.",
        ]
    else:
        counts = nn.astype(str).value_counts()
        shown = counts.head(top)
        labels = [_clean_label(v) for v in shown.index]
        fig, ax = _new_figure(height=max(3.0, 0.34 * len(shown) + 1.2))
        ax.barh(labels[::-1], list(shown.values)[::-1], color="#3E7CB1")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.25, linewidth=0.7)
        ax.set_xlabel("rows")
        ax.set_title(f"Most common values in '{column}'"
                     + (f" (top {len(shown)} of {counts.size:,})" if counts.size > len(shown) else ""))
        rel = _save_fig(fig, f"{_safe_name(name)}__{_safe_name(column)}_counts.png")

        lines = [
            f"Bar chart of '{column}' saved to {rel}",
            f"  {counts.size:,} distinct value(s) across {len(nn):,} rows, {nulls:,} blank.",
        ]
        for label, count in shown.head(5).items():
            lines.append(f"  {_clean_label(label, 60)}: {count:,} rows ({100 * count / len(nn):.1f}%)")
        if counts.size > len(shown):
            lines.append(f"  ...and {counts.size - len(shown):,} more value(s) not charted.")
        if counts.size == len(nn):
            lines.append("  Every row is different — this looks like an ID, not a category.")
        sep, combos, singles = _detect_multivalue(series)
        if sep:
            lines.append(f"  These are LISTS, not single categories: {combos} combinations built "
                         f"from only {singles} distinct values (split on '{sep}'). Comparing the "
                         f"combinations is meaningless — call "
                         f"analyze_multivalue('{name}', '{column}') instead.")
    if sample_note:
        lines.append(f"  NOTE: {sample_note}; the numbers above use all rows.")
    lines.extend(f"  {w}" for w in _cleaning_warnings(df, column))
    return "\n".join(lines)


# ─── 1b. multi-valued columns ──────────────────────────────────────────────────

def analyze_multivalue(name: str, column: str, value_col: Optional[str] = None,
                       how: str = "mean", sep: Optional[str] = None,
                       min_rows: int = MIN_GROUP_ROWS, top: int = TOP_CATEGORIES,
                       output_name: Optional[str] = None) -> str:
    """Split a list-valued column into its individual values and analyse those.

    'action, adventure, comedy' counts once for action, once for adventure and
    once for comedy — so rows are counted more than once by design. Read-only.
    """
    df, err = _get_df(name)
    if err:
        return err
    if column not in df.columns:
        return f"Error: column '{column}' not in '{name}'. Columns: {list(df.columns)}"
    if value_col and value_col not in df.columns:
        return f"Error: column '{value_col}' not in '{name}'. Columns: {list(df.columns)}"
    if value_col and not _is_numeric(df[value_col]):
        return f"Error: '{value_col}' must be a number column to average."

    separator = sep or _detect_multivalue(df[column])[0] or ","
    work = df[[column] + ([value_col] if value_col else [])].dropna(subset=[column])
    exploded = work.assign(**{column: work[column].map(lambda v: _split_values(v, separator))})
    exploded = exploded.explode(column)
    exploded = exploded[exploded[column].astype(str).str.len() > 0]
    if exploded.empty:
        return f"'{column}' produced no values when split on '{separator}'."
    vocabulary = int(exploded[column].nunique())
    if vocabulary > MULTIVALUE_HARD_LIMIT:
        return (f"Error: splitting '{column}' on '{separator}' yields {vocabulary:,} distinct "
                f"values — that is a list of names or free prose, not a tag vocabulary. "
                f"Ranking thousands of values by an average over a handful of rows each "
                f"produces confident nonsense. If this column packs several fields together "
                f"(e.g. 'Director: … | Stars: …'), parse it into separate columns first.")

    if value_col:
        stats = exploded.groupby(column)[value_col].agg(["count", "mean", "median"])
        stats = stats.rename(columns={"count": "rows", "mean": f"{value_col}_mean",
                                      "median": f"{value_col}_median"})
        eligible = stats[stats["rows"] >= min_rows].sort_values(f"{value_col}_mean", ascending=False)
        dropped = len(stats) - len(eligible)
        if eligible.empty:
            return (f"Split '{column}' into {len(stats)} value(s), but none has {min_rows}+ rows — "
                    f"no average is trustworthy. Lower min_rows only if you accept that.")
        shown = eligible.head(top)
        fig, ax = _new_figure(height=max(3.2, 0.34 * len(shown) + 1.3))
        ax.barh([_clean_label(i, 28) for i in shown.index][::-1],
                list(shown[f"{value_col}_mean"])[::-1], color="#3E7CB1")
        ax.grid(axis="y", visible=False); ax.grid(axis="x", alpha=.25, linewidth=.7)
        ax.set_xlabel(f"average {value_col}")
        ax.set_title(f"Average '{value_col}' per individual '{column}'")
        rel = _save_fig(fig, f"{_safe_name(name)}__{_safe_name(column)}_by_{_safe_name(value_col)}.png")
        result = eligible.reset_index()
        lines = [f"Chart saved to {rel}",
                 f"  '{column}' split on '{separator}' → {len(stats)} distinct value(s) "
                 f"(from {df[column].nunique()} combinations).",
                 f"  Highest average '{value_col}' (min {min_rows} rows):"]
        for label, row in shown.head(6).iterrows():
            lines.append(f"    {_clean_label(label, 30)}: {_fmt(row[f'{value_col}_mean'])} "
                         f"({int(row['rows']):,} rows)")
        tail = eligible.tail(3).iloc[::-1]
        lines.append(f"  Lowest:")
        for label, row in tail.iterrows():
            lines.append(f"    {_clean_label(label, 30)}: {_fmt(row[f'{value_col}_mean'])} "
                         f"({int(row['rows']):,} rows)")
        if dropped:
            lines.append(f"  Left out {dropped} value(s) with fewer than {min_rows} rows.")
    else:
        counts = exploded[column].value_counts()
        shown = counts.head(top)
        fig, ax = _new_figure(height=max(3.2, 0.34 * len(shown) + 1.3))
        ax.barh([_clean_label(i, 28) for i in shown.index][::-1], list(shown.values)[::-1],
                color="#3E7CB1")
        ax.grid(axis="y", visible=False); ax.grid(axis="x", alpha=.25, linewidth=.7)
        ax.set_xlabel("rows containing it")
        ax.set_title(f"How often each individual '{column}' appears")
        rel = _save_fig(fig, f"{_safe_name(name)}__{_safe_name(column)}_split_counts.png")
        result = pd.DataFrame({column: list(counts.index), "rows": list(counts.values)})
        total = len(work)
        lines = [f"Chart saved to {rel}",
                 f"  '{column}' split on '{separator}' → {counts.size} distinct value(s) "
                 f"(from {df[column].nunique()} combinations).",
                 f"  Most common:"]
        for label, count in shown.head(8).items():
            lines.append(f"    {_clean_label(label, 30)}: {count:,} rows ({100 * count / total:.1f}%)")

    out = output_name or f"{name}_{_safe_name(column)}_split"
    _register(out, result)
    lines.append(f"  Registered the full breakdown as '{out}'.")
    lines.append("  NOTE: a row with several values is counted once per value, so the "
                 "percentages add to more than 100%.")
    return "\n".join(lines)


# ─── 1c. missing-value patterns ────────────────────────────────────────────────

MISSING_TOGETHER = 0.9          # missingness correlation that counts as "always together"
SUBGROUP_MIN_ROWS = 30          # a correlation over fewer rows than this is noise


def _missing_blocks(miss: pd.DataFrame, columns: list) -> list:
    """Group columns whose blanks appear in the SAME rows. Three columns each
    20% empty is a very different problem depending on whether that is 20% of
    rows missing one field, or the same 20% missing all three."""
    if len(columns) < 2:
        return []
    corr = miss[columns].corr()
    blocks, seen = [], set()
    for col in columns:
        if col in seen:
            continue
        partners = [o for o in columns
                    if o != col and pd.notna(corr.loc[col, o])
                    and corr.loc[col, o] >= MISSING_TOGETHER]
        if partners:
            block = [col] + partners
            seen.update(block)
            blocks.append(block)
    return blocks


# ── is the missingness random, or driven by something we can see? ─────────────
# Two columns going blank in the SAME rows is one pattern (handled by
# _missing_blocks). The other, more dangerous one is missingness that depends on
# an OBSERVED value — income blank exactly when age > 55. Those blanks look
# perfectly scattered, so a co-missingness check calls them random and blesses
# mean-imputation, which then biases the very relationship being studied.
#
# Testing every column against every gap is a multiple-comparison machine, so the
# threshold is Bonferroni-corrected: with 40 candidate columns, one "significant"
# result at p<0.05 is expected by chance alone.

MAR_ALPHA = 0.05
MAR_MIN_GROUP = 10              # rows on each side before a test means anything
MAR_MAX_TESTS = 40


def _missingness_drivers(df: pd.DataFrame, col: str, miss: pd.DataFrame) -> list:
    """Observed columns that predict whether `col` is blank. [] when none do.

    Returns (predictor, plain-language direction, p) sorted by strength.
    """
    flag = miss[col]
    if not flag.any() or flag.all():
        return []
    try:
        from scipy import stats
    except Exception:  # noqa: BLE001 — optional dependency; silence beats a crash
        return []
    candidates = [c for c in df.columns
                  if c != col and not _is_marker(c) and not miss[c].all()][:MAR_MAX_TESTS]
    if not candidates:
        return []
    alpha = MAR_ALPHA / len(candidates)          # Bonferroni
    hits = []
    for other in candidates:
        s = df[other]
        try:
            if _is_numeric(s):
                a = pd.to_numeric(s[flag], errors="coerce").dropna()
                b = pd.to_numeric(s[~flag], errors="coerce").dropna()
                if len(a) < MAR_MIN_GROUP or len(b) < MAR_MIN_GROUP:
                    continue
                if a.nunique() < 2 and b.nunique() < 2:
                    continue
                p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                if p < alpha:
                    hi, lo = float(a.median()), float(b.median())
                    word = "higher" if hi > lo else "lower"
                    hits.append((other, f"blank rows have {word} '{other}' "
                                        f"(median {_fmt(hi)} vs {_fmt(lo)})", p, False))
            else:
                table = pd.crosstab(s.fillna("(blank)").astype(str), flag)
                if table.shape[0] < 2 or table.shape[1] < 2 or table.to_numpy().sum() < 20:
                    continue
                if table.shape[0] > 50:
                    continue                     # free text, not a grouping
                p = float(stats.chi2_contingency(table)[1])
                if p < alpha:
                    rate = flag.groupby(s.fillna("(blank)").astype(str)).mean().sort_values()
                    worst, best = rate.index[-1], rate.index[0]
                    hits.append((other, f"'{other}'={_clean_label(worst, 20)} is blank "
                                        f"{rate.iloc[-1]:.0%} of the time vs "
                                        f"{rate.iloc[0]:.0%} for {_clean_label(best, 20)}", p, True))
        except Exception:  # noqa: BLE001 — a degenerate column must not stop the scan
            continue
    hits.sort(key=lambda h: h[2])
    return hits


def analyze_missing(name: str, min_frac: float = 0.01, top: int = 12) -> str:
    """Are the blanks random accidents, or one distinct group of records?

    This is the question that decides whether imputation is legitimate at all.
    Columns that go blank together mark a structurally different set of rows —
    filling each one separately fabricates whole records.
    """
    df, err = _get_df(name)
    if err:
        return err
    if df.empty:
        return f"'{name}' has no rows."
    miss = df.isnull()
    frac = miss.mean().sort_values(ascending=False)
    affected = frac[frac >= min_frac]
    if affected.empty:
        return (f"'{name}': no column is more than {min_frac:.0%} empty — "
                f"nothing to worry about. {len(df):,} complete rows.")

    shown = affected.head(top)
    fig, ax = _new_figure(height=max(3.0, 0.36 * len(shown) + 1.2))
    ax.barh([_clean_label(c, 26) for c in shown.index][::-1],
            [100 * v for v in shown.values][::-1], color="#C25E4B")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", alpha=.25, linewidth=.7)
    ax.set_xlabel("% of rows blank")
    ax.set_title(f"What is missing in '{name}'")
    rel = _save_fig(fig, f"{_safe_name(name)}__missing.png")

    complete = int((~miss.any(axis=1)).sum())
    lines = [
        f"Missing-value chart saved to {rel}",
        f"  {complete:,} of {len(df):,} rows ({complete / len(df):.0%}) are completely filled in.",
        "  Blank per column:",
    ]
    for col, value in shown.items():
        lines.append(f"    {col}: {int(miss[col].sum()):,} ({value:.1%})")

    # Does anything we can SEE predict the gaps? This decides whether imputing is
    # legitimate, so it is checked before any verdict is offered.
    driver_lines: list = []
    for col in list(affected.index)[:top]:
        drivers = _missingness_drivers(df, col, miss)
        if not drivers:
            continue
        why = "; ".join(d[1] for d in drivers[:3])
        names = ", ".join(f"'{d[0]}'" for d in drivers[:3])
        driver_lines.append(
            f"    '{col}' is NOT missing at random — which rows are blank depends on {names}: "
            f"{why}" + (f" (+{len(drivers) - 3} more)" if len(drivers) > 3 else "") + ".")
        # Grouped imputation only makes sense on a driver you can group BY; doing
        # it on a continuous one yields a group per row, which imputes nothing.
        groupable = next((d[0] for d in drivers if d[3]), None)
        remedy = (f"Impute within groups of '{groupable}' (impute strategy=median, "
                  f"by='{groupable}')" if groupable else
                  f"Predict '{col}' from '{drivers[0][0]}' rather than filling a flat value "
                  f"(or bucket '{drivers[0][0]}' first and impute within buckets)")
        driver_lines.append(
            f"      ⚠ Filling '{col}' with a mean/median computed over the rows that DO have "
            f"it will bias the result, because those rows are systematically different. "
            f"{remedy}, or report '{col}' only on the rows that have it — and keep a "
            f"_was_missing marker either way.")

    blocks = _missing_blocks(miss, list(affected.index))
    if not blocks:
        if driver_lines:
            lines.append("  PATTERN: no column group goes missing together — but the blanks "
                         "are NOT random either:")
            lines.extend(driver_lines)
        else:
            lines.append("  PATTERN: the blanks are spread across different rows — no column "
                         "group goes missing together, and nothing else in the data predicts "
                         "which rows are blank, so filling each one separately is reasonable.")
        return "\n".join(lines)

    lines.append("  PATTERN — these columns go blank in the SAME rows:")
    for block in blocks:
        together = miss[block].all(axis=1)
        n = int(together.sum())
        lines.append(f"    {block} → {n:,} row(s) missing ALL of them "
                     f"({n / len(df):.0%} of the data)")
        # what do those rows look like? context is what makes the pattern readable
        context = [c for c in df.columns if c not in block and not _is_marker(c)][:3]
        if context and n:
            preview = df.loc[together, context].head(2)
            for idx, row in preview.iterrows():
                cells = ", ".join(f"{c}={_clean_label(row[c], 24)}" for c in context)
                lines.append(f"        e.g. [{idx}] {cells}")
        lines.append(f"      ⚠ These {n:,} rows are a distinct GROUP, not scattered gaps. "
                     f"Imputing {len(block)} column(s) here invents a whole record for each "
                     f"one. Consider dropping them, or analysing them separately — and if "
                     f"you do impute, keep the _was_missing markers and exclude these rows "
                     f"from any average you report.")
    if driver_lines:
        lines.append("  PATTERN — the blanks also depend on values you can see:")
        lines.extend(driver_lines)
    return "\n".join(lines)


# ─── 2. two columns ────────────────────────────────────────────────────────────

def plot_relationship(name: str, x: str, y: str) -> str:
    """Chart + written reading of how two columns relate. The chart type is
    chosen from the column types so the caller cannot pick a wrong one."""
    df, err = _get_df(name)
    if err:
        return err
    missing = [c for c in (x, y) if c not in df.columns]
    if missing:
        return f"Error: column(s) {missing} not in '{name}'. Columns: {list(df.columns)}"

    pair = df[[x, y]].dropna()
    if len(pair) < 3:
        return f"Error: only {len(pair)} row(s) have both '{x}' and '{y}' — too few to compare."
    plot_df, sample_note = _sample_for_plot(pair)
    title_note = f"  ({sample_note})" if sample_note else ""

    x_num, y_num = _is_numeric(df[x]), _is_numeric(df[y])
    x_date = _is_datetime(df[x]) or (not x_num and _as_datetime(pair[x]).notna().mean() >= 0.8)

    # date vs number → trend over time
    if x_date and y_num:
        ordered = pair.assign(_d=_as_datetime(pair[x])).dropna(subset=["_d"]).sort_values("_d")
        plot_o, _ = _sample_for_plot(ordered)
        fig, ax = _new_figure()
        ax.plot(plot_o["_d"], plot_o[y], color="#3E7CB1", linewidth=1.4)
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(f"'{y}' over '{x}'{title_note}")
        fig.autofmt_xdate()
        rel = _save_fig(fig, f"{_safe_name(name)}__{_safe_name(y)}_over_{_safe_name(x)}.png")
        first, last = float(ordered[y].iloc[0]), float(ordered[y].iloc[-1])
        change = (last - first) / abs(first) * 100 if first else float("nan")
        direction = "rising" if last > first else "falling" if last < first else "flat"
        out = [
            f"Trend chart saved to {rel}",
            f"  '{y}' from {ordered['_d'].min():%Y-%m-%d} to {ordered['_d'].max():%Y-%m-%d}.",
            f"  Starts at {_fmt(first)}, ends at {_fmt(last)} — {direction}"
            + (f", a change of {change:+.1f}%." if np.isfinite(change) else "."),
            "  This is raw row order, not a total per period — use analyze_over_time for that.",
        ]
        if sample_note:
            out.append(f"  NOTE: {sample_note}.")
        return "\n".join(out)

    # number vs number → scatter with trend line
    if x_num and y_num:
        fig, ax = _new_figure()
        ax.scatter(plot_df[x], plot_df[y], s=14, alpha=0.55, color="#3E7CB1",
                   edgecolors="none")
        r = float(pair[x].corr(pair[y])) if len(pair) >= CORR_MIN_ROWS else float("nan")
        note_correlation(x, y, r)
        if len(pair) >= 2 and pair[x].nunique() > 1:
            slope, intercept = np.polyfit(pair[x], pair[y], 1)
            xs = np.linspace(pair[x].min(), pair[x].max(), 50)
            ax.plot(xs, slope * xs + intercept, color="#C25E4B", linewidth=1.6)
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(f"'{y}' against '{x}'{title_note}")
        rel = _save_fig(fig, f"{_safe_name(name)}__{_safe_name(x)}_vs_{_safe_name(y)}.png")
        if np.isfinite(r):
            move = "up together" if r > 0 else "in opposite directions"
            reading = (f"  As '{x}' changes, '{y}' tends to move {move}. "
                       f"Correlation {r:+.2f} ({_strength(r)}).")
        else:
            reading = "  Too few rows to measure a correlation."
        out = [
            f"Scatter plot saved to {rel}",
            f"  {len(pair):,} rows have both values.",
            reading,
            "  A relationship is not proof that one causes the other.",
        ]
        # a pooled correlation can be wrong for every group inside it
        if np.isfinite(r) and abs(r) >= 0.15:
            groupers = [c for c in df.columns
                        if c not in (x, y) and not _is_marker(c)
                        and not _is_numeric(df[c])
                        and 2 <= df[c].nunique(dropna=True) <= HIGH_CARDINALITY]
            if not groupers:
                groupers = [c for c in df.columns
                            if c not in (x, y) and not _is_marker(c) and _detect_multivalue(df[c])[0]]
            if groupers:
                out.append(f"  This is the POOLED figure. Confirm it holds within groups: "
                           f"check_subgroups('{name}', '{x}', '{y}', by='{groupers[0]}').")
        if sample_note:
            out.append(f"  NOTE: {sample_note}; the correlation uses all rows.")
        return "\n".join(out)

    # number vs category → box plot per group
    if x_num != y_num:
        cat_col, num_col = (x, y) if y_num else (y, x)
        groups = pair.groupby(pair[cat_col].astype(str))[num_col]
        summary = groups.agg(["count", "mean", "median"]).sort_values("mean", ascending=False)
        # a group of 1 row has an "average" that is just that row — ranking
        # categories by it produces confident-looking nonsense
        big_enough = summary[summary["count"] >= MIN_GROUP_ROWS]
        dropped = len(summary) - len(big_enough)
        if big_enough.empty:
            return (f"'{cat_col}' has {len(summary)} group(s) but none with at least "
                    f"{MIN_GROUP_ROWS} rows, so no average is trustworthy. Group the column "
                    f"into broader categories first (pivot_dataset), then compare.")
        summary = big_enough
        shown = summary.head(TOP_CATEGORIES)
        data = [pair.loc[pair[cat_col].astype(str) == label, num_col].values for label in shown.index]
        fig, ax = _new_figure(height=max(3.2, 0.34 * len(shown) + 1.4))
        ax.boxplot(data, vert=False, tick_labels=[_clean_label(i) for i in shown.index],
                   patch_artist=True,
                   boxprops=dict(facecolor="#D6E4F0", edgecolor="#3E7CB1"),
                   medianprops=dict(color="#C25E4B", linewidth=1.6),
                   flierprops=dict(marker=".", markersize=4, alpha=0.5))
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.25, linewidth=0.7)
        ax.set_xlabel(num_col)
        ax.set_title(f"'{num_col}' by '{cat_col}'"
                     + (f" (top {len(shown)} of {len(summary)})" if len(summary) > len(shown) else ""))
        rel = _save_fig(fig, f"{_safe_name(name)}__{_safe_name(num_col)}_by_{_safe_name(cat_col)}.png")
        out = [f"Box plot saved to {rel}",
               f"  Average '{num_col}' per '{cat_col}' (highest first):"]
        for label, row in shown.head(6).iterrows():
            out.append(f"    {_clean_label(label, 55)}: average {_fmt(row['mean'])}, "
                       f"middle {_fmt(row['median'])}, {int(row['count']):,} rows")
        if dropped:
            out.append(f"  Left out {dropped:,} group(s) with fewer than {MIN_GROUP_ROWS} rows — "
                       f"an average over 1-2 rows is not a finding.")
        spread = shown['mean'].max() - shown['mean'].min()
        out.append(f"  Gap between highest and lowest average: {_fmt(spread)}. "
                   f"Use compare_groups to test whether that gap is real.")
        n_out, _, _ = _outlier_count(pair[num_col])
        if n_out:
            out.append(f"  WARNING: {n_out:,} extreme value(s) stretch the scale, so the boxes "
                       f"look flat and the groups are hard to tell apart. Clean '{num_col}' "
                       f"first, then re-plot — extremes also hide real differences from "
                       f"compare_groups.")
        return "\n".join(out)

    # category vs category → counts grid
    table = pd.crosstab(pair[x].astype(str), pair[y].astype(str))
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index[:TOP_CATEGORIES],
                      table.sum(axis=0).sort_values(ascending=False).index[:TOP_CATEGORIES]]
    fig, ax = _new_figure(width=max(5.5, 0.6 * table.shape[1] + 3),
                          height=max(3.2, 0.42 * table.shape[0] + 1.6))
    im = ax.imshow(table.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(table.shape[1]), labels=[_clean_label(c, 22) for c in table.columns],
                  rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]), labels=[_clean_label(i, 22) for i in table.index])
    ax.grid(visible=False)
    ax.set_title(f"How often '{x}' and '{y}' occur together")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    rel = _save_fig(fig, f"{_safe_name(name)}__{_safe_name(x)}_by_{_safe_name(y)}_grid.png")
    biggest = table.stack().idxmax()
    return "\n".join([
        f"Counts grid saved to {rel}",
        f"  {table.shape[0]} × {table.shape[1]} combinations shown.",
        f"  Most common pairing: '{biggest[0]}' with '{biggest[1]}' "
        f"({int(table.stack().max()):,} rows).",
    ])


# ─── 2b. does the relationship hold within groups? ─────────────────────────────

def _subgroup_rates(df, name, x, y, by, min_rows, top) -> str:
    """Does the group that wins overall also win inside every subgroup?

    The pooled winner can lose everywhere at once, and it happens whenever the
    groups are mixed differently across the subgroups: a treatment given mostly
    to easy cases posts a better overall rate while being worse on easy cases
    AND worse on hard ones. The pooled number is not wrong arithmetic — it is
    the wrong question, and no amount of significance testing on it helps.
    """
    work = df[[x, y, by]].dropna()
    if len(work) < min_rows:
        return f"Error: only {len(work)} complete row(s) — too few to split into groups."
    if work[x].nunique() < 2:
        return f"Error: '{x}' has only one value, so there is nothing to compare."
    if work[x].nunique() > HIGH_CARDINALITY:
        return (f"Error: '{x}' has {work[x].nunique()} values — too many to compare. "
                f"Use a column with a handful of levels.")

    labels_x = work[x].astype(str)
    overall = work.groupby(labels_x)[y].agg(["mean", "count"]).sort_values("mean",
                                                                          ascending=False)
    winner, loser = str(overall.index[0]), str(overall.index[-1])

    rows = []
    for label, part in work.groupby(work[by].astype(str)):
        if len(part) < min_rows:
            continue
        inner = part.groupby(part[x].astype(str))[y].agg(["mean", "count"])
        if winner not in inner.index or loser not in inner.index:
            continue
        rows.append((str(label), int(len(part)),
                     float(inner.loc[winner, "mean"]), int(inner.loc[winner, "count"]),
                     float(inner.loc[loser, "mean"]), int(inner.loc[loser, "count"])))
    if len(rows) < 2:
        return (f"Only {len(rows)} group(s) in '{by}' contain both '{winner}' and '{loser}' "
                f"with {min_rows}+ rows, so the comparison cannot be checked across groups. "
                f"Overall: '{winner}' {overall.loc[winner, 'mean']:.3g} vs '{loser}' "
                f"{overall.loc[loser, 'mean']:.3g}.")

    lines = [f"Comparing '{y}' between values of '{x}', checked inside each '{by}':",
             f"  OVERALL ({len(work):,} rows):"]
    for label, row in overall.iterrows():
        lines.append(f"    {_clean_label(str(label), 26)}: {row['mean']:.4g} "
                     f"({int(row['count']):,} rows)")
    lines.append(f"  → '{winner}' looks best overall.")
    lines.append(f"  WITHIN each '{by}' ({len(rows)} group(s) with {min_rows}+ rows):")

    reversed_in = []
    for label, n, w_mean, w_n, l_mean, l_n in rows[:top]:
        flipped = l_mean > w_mean
        if flipped:
            reversed_in.append(label)
        mark = "   ← reversed" if flipped else ""
        lines.append(f"    {_clean_label(label, 20)} (n={n:,}): '{winner}' {w_mean:.4g} "
                     f"(n={w_n:,})  vs  '{loser}' {l_mean:.4g} (n={l_n:,}){mark}")
    # count reversals across ALL groups, not only the ones shown
    reversed_all = [r[0] for r in rows if r[4] > r[2]]

    if len(reversed_all) == len(rows):
        lines.append(
            f"  ⚠ SIMPSON'S PARADOX. '{winner}' wins overall but LOSES to '{loser}' in EVERY "
            f"single '{by}'. The overall figure is not a summary of these groups — it is an "
            f"artefact of how '{x}' is distributed across them ('{winner}' is concentrated in "
            f"the easier groups). Reporting the overall number would recommend the worse "
            f"option. Quote the per-'{by}' figures instead, and if you need one headline "
            f"number, weight the groups equally rather than by their size.")
    elif reversed_all:
        lines.append(
            f"  ⚠ THE WINNER REVERSES in {len(reversed_all)} of {len(rows)} group(s) "
            f"({', '.join(repr(g) for g in reversed_all[:4])}). The overall comparison hides "
            f"opposite results, so it cannot be reported on its own — state it per '{by}'.")
    else:
        lines.append(f"  '{winner}' wins in every '{by}' as well as overall, so the headline "
                     f"comparison is safe to report.")
    lines.append("  A difference in rates still does not tell you why it exists.")
    return "\n".join(lines)


def check_subgroups(name: str, x: str, y: str, by: str,
                    min_rows: int = SUBGROUP_MIN_ROWS, top: int = 12) -> str:
    """Recompute the x~y relationship inside each group of `by`.

    One pooled correlation can be true on average and wrong for every actual
    group in it — the average national temperature is nobody's weather. This is
    the check that turns a provisional finding into a defensible one.
    """
    df, err = _get_df(name)
    if err:
        return err
    missing = [c for c in (x, y, by) if c not in df.columns]
    if missing:
        return f"Error: column(s) {missing} not in '{name}'. Columns: {list(df.columns)}"
    if not _is_numeric(df[y]):
        return (f"Error: '{y}' must be a number column — it is the outcome being compared. "
                f"For a yes/no outcome, encode it as 1/0 first.")
    if not _is_numeric(df[x]):
        # The RATE form of the same paradox. Simpson's has two shapes and this
        # tool used to cover only one: a correlation that reverses inside its
        # groups. The other — one treatment beating another overall while
        # losing in every subgroup — is the textbook case (Berkeley admissions,
        # the kidney-stone trial) and by far the more common one in business:
        # "the new checkout converts better overall, but worse on both mobile
        # AND desktop". Refusing it sent exactly the question this tool exists
        # for to a tool that cannot answer it.
        return _subgroup_rates(df, name, x, y, by, min_rows, top)

    work = df[[x, y, by]].dropna()
    if len(work) < min_rows:
        return f"Error: only {len(work)} complete row(s) — too few to split into groups."

    note = ""
    sep, _, _ = _detect_multivalue(work[by])
    if sep:
        # 'action, adventure, comedy' → group by the first listed value, and say so
        work = work.assign(**{by: work[by].map(lambda v: _split_values(v, sep)[0]
                                               if _split_values(v, sep) else v)})
        note = (f"  NOTE: '{by}' holds lists; grouped by the FIRST value in each. "
                f"Use analyze_multivalue for a full per-value breakdown.")

    overall = float(work[x].corr(work[y]))
    note_correlation(x, y, overall)
    rows = []
    for label, part in work.groupby(work[by].astype(str)):
        if len(part) >= min_rows and part[x].nunique() > 1 and part[y].nunique() > 1:
            r = part[x].corr(part[y])
            if pd.notna(r):
                rows.append((str(label), len(part), float(r)))
    if len(rows) < 2:
        return (f"Only {len(rows)} group(s) in '{by}' have {min_rows}+ rows, so the "
                f"relationship cannot be compared across groups. Overall correlation "
                f"{overall:+.2f}. Try a broader grouping column.")

    rows.sort(key=lambda t: t[2])
    shown = rows[:top] if len(rows) <= top else rows[:top // 2] + rows[-(top - top // 2):]

    fig, ax = _new_figure(height=max(3.2, 0.34 * len(shown) + 1.4))
    labels = [_clean_label(f"{lbl} (n={n:,})", 30) for lbl, n, _ in shown]
    values = [r for _, _, r in shown]
    colors = ["#C25E4B" if (r < 0) != (overall < 0) else "#3E7CB1" for r in values]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(overall, color="#22303C", linestyle="--", linewidth=1.2)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", alpha=.25, linewidth=.7)
    ax.set_xlabel(f"correlation of '{x}' with '{y}'   (dashed = overall {overall:+.2f})")
    ax.set_title(f"Does '{x}' ↔ '{y}' hold within each '{by}'?")
    rel = _save_fig(fig, f"{_safe_name(name)}__{_safe_name(x)}_{_safe_name(y)}_by_{_safe_name(by)}.png")

    lo, hi = rows[0][2], rows[-1][2]
    flipped = [r for r in rows if (r[2] < 0) != (overall < 0) and abs(r[2]) >= 0.05]
    lines = [f"Subgroup chart saved to {rel}",
             f"  Overall '{x}' ↔ '{y}': {overall:+.2f} ({_strength(overall)}), "
             f"across {len(work):,} rows.",
             f"  Within each '{by}' ({len(rows)} group(s) with {min_rows}+ rows):"]
    for label, n, r in shown:
        mark = "  ← opposite direction" if (r < 0) != (overall < 0) and abs(r) >= 0.05 else ""
        lines.append(f"    {_clean_label(label, 26)}: {r:+.2f} ({n:,} rows){mark}")
    if note:
        lines.append(note)

    spread = hi - lo
    if flipped:
        names = ", ".join(f"'{f[0]}'" for f in flipped[:4])
        lines.append(f"  ⚠ THE RELATIONSHIP REVERSES for {names}. The overall {overall:+.2f} "
                     f"is an average of opposite effects — reporting it alone would be wrong. "
                     f"State it per '{by}'.")
    elif spread >= 0.3:
        lines.append(f"  ⚠ Direction holds everywhere, but the strength ranges from {lo:+.2f} to "
                     f"{hi:+.2f} — a spread of {spread:.2f}. The overall figure is much stronger "
                     f"for some groups than others, so quote it per '{by}'.")
    else:
        lines.append(f"  The relationship is consistent across groups (range {lo:+.2f} to "
                     f"{hi:+.2f}). The overall {overall:+.2f} is safe to report on its own.")
    return "\n".join(lines)


# ─── 3. correlations ───────────────────────────────────────────────────────────

# A correlation without its sample size is not a finding. r = 0.84 on six points
# has a 95% interval of roughly [0.03, 0.98] — consistent with "almost nothing"
# and with "almost perfect". These attach n and a significance test to every
# number the tool reports, so a thin estimate cannot be quoted as a strong one.

CORR_THIN_N = 30                # below this an estimate is too unstable to lean on
CORR_ALPHA = 0.05


def _pair_n(df: pd.DataFrame, a: str, b: str) -> int:
    """Rows where BOTH columns are present — the n the estimate actually rests on."""
    return int(df[[a, b]].dropna().shape[0])


def _corr_pvalue(df: pd.DataFrame, a: str, b: str, method: str) -> Optional[float]:
    """Two-sided p for 'the true correlation is zero'. None if scipy is absent."""
    pair = df[[a, b]].dropna()
    if len(pair) < 3:
        return None
    try:
        from scipy import stats
    except Exception:  # noqa: BLE001 — scipy is optional everywhere else too
        return None
    fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr,
          "kendall": stats.kendalltau}.get(method, stats.pearsonr)
    try:
        return float(fn(pair[a].to_numpy(), pair[b].to_numpy())[1])
    except Exception:  # noqa: BLE001 — a constant column makes the test undefined
        return None


def _corr_caveat(n: int, p: Optional[float]) -> str:
    """The short clause that stops a thin or non-significant r being quoted flat."""
    if n < CORR_THIN_N:
        return (f", n={n} — TOO FEW ROWS to rely on; this could easily be chance. "
                f"Do not report it as a finding without more data")
    if p is not None and p > CORR_ALPHA:
        return f", n={n}, p={p:.3f} — NOT significant; consistent with no relationship at all"
    if p is not None:
        return f", n={n}, p={p:.3g}"
    return f", n={n}"


def analyze_correlations(name: str, target: Optional[str] = None, method: str = "pearson") -> str:
    """Heatmap + a ranked, plain-language list of which columns move together."""
    df, err = _get_df(name)
    if err:
        return err
    numeric = df.select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    # Numbers that are labels (status codes, postcodes) have no meaningful
    # correlation — the arithmetic runs, the result means nothing.
    coded = [c for c in _coded_columns(numeric) if c != target]
    if coded and numeric.shape[1] - len(coded) >= 2:
        numeric = numeric.drop(columns=coded)
    else:
        coded = []
    # keep markers only if the caller explicitly asked about one
    markers = [c for c in numeric.columns if _is_marker(c) and c != target]
    if markers and numeric.shape[1] - len(markers) >= 2:
        numeric = numeric.drop(columns=markers)
    else:
        markers = []
    if numeric.shape[1] < 2:
        return (f"'{name}' has {numeric.shape[1]} usable number column(s) — "
                "at least 2 are needed to compare. Try cast_types on text columns first.")
    if len(numeric.dropna(how="all")) < CORR_MIN_ROWS:
        return f"'{name}' has too few rows ({len(numeric)}) to measure correlations."
    if target and target not in numeric.columns:
        return (f"Error: '{target}' is not a usable number column in '{name}'. "
                f"Available: {list(numeric.columns)}")

    corr = numeric.corr(method=method)
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            note_correlation(a, b, corr.loc[a, b])
    fig, ax = _new_figure(width=max(5.0, 0.55 * len(corr) + 2.6),
                          height=max(4.0, 0.55 * len(corr) + 2.0))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), labels=list(corr.columns), rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), labels=list(corr.index))
    ax.grid(visible=False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else "#22303C")
    ax.set_title(f"How the numbers in '{name}' relate ({method})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    rel = _save_fig(fig, f"{_safe_name(name)}__correlations.png")

    lines = [f"Correlation heatmap saved to {rel}",
             "  +1 means they rise together, -1 means one rises as the other falls, 0 means unrelated."]
    if coded:
        lines.append(f"  Left out {len(coded)} code/label column(s) ({coded}) — they are numbers "
                     f"used as labels, so a correlation on them is arithmetic without meaning. "
                     f"Compare them with rank_by / plot_relationship instead.")
    if markers:
        lines.append(f"  Left out {len(markers)} cleaning-marker column(s) ({markers[:3]}"
                     f"{'...' if len(markers) > 3 else ''}) — they record what cleaning did, "
                     "so correlating them with their own source column proves nothing.")
    if target:
        ranked = corr[target].drop(index=target).sort_values(key=abs, ascending=False)
        lines.append(f"  What moves most with '{target}':")
        for col, r in ranked.items():
            n = _pair_n(numeric, target, col)
            caveat = _corr_caveat(n, _corr_pvalue(numeric, target, col, method))
            lines.append(f"    {col}: {r:+.2f} ({_strength(r)}{caveat})")
        best = ranked.iloc[0] if len(ranked) else 0
        if abs(best) < 0.2:
            lines.append(f"  Nothing here explains '{target}' well — the strongest link is only "
                         f"{best:+.2f}. The driver may be a column that is not a number yet.")
    else:
        pairs = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = corr.loc[a, b]
                if pd.notna(r):
                    pairs.append((abs(r), a, b, r))
        pairs.sort(reverse=True)
        lines.append("  Strongest relationships:")
        for _, a, b, r in pairs[:8]:
            n = _pair_n(numeric, a, b)
            caveat = _corr_caveat(n, _corr_pvalue(numeric, a, b, method))
            lines.append(f"    {a} ↔ {b}: {r:+.2f} ({_strength(r)}{caveat})")
        if pairs and pairs[0][0] < 0.2:
            lines.append("  No strong relationships — these columns look largely independent.")
    lines.append("  Moving together is not proof that one causes the other.")
    return "\n".join(lines)


# ─── 4. pivot / cross-tab ──────────────────────────────────────────────────────

def pivot_dataset(name: str, rows: list, columns: Optional[list] = None,
                  values: Optional[str] = None, how: str = "sum",
                  percent: bool = False, output_name: Optional[str] = None) -> str:
    """Cross-tab grid: rows down the side, columns across the top."""
    df, err = _get_df(name)
    if err:
        return err
    rows = [rows] if isinstance(rows, str) else list(rows or [])
    columns = [columns] if isinstance(columns, str) else list(columns or [])
    unknown = [c for c in rows + columns + ([values] if values else []) if c not in df.columns]
    if unknown:
        return f"Error: column(s) {unknown} not in '{name}'. Columns: {list(df.columns)}"
    if not rows:
        return "Error: 'rows' needs at least one column to group down the side."
    # a grid of 500 columns is a quarter of a megabyte of text nobody can read,
    # and it crowds out the rest of the conversation
    for axis, cols in (("columns", columns), ("rows", rows)):
        for col in cols:
            n = int(df[col].nunique(dropna=False))
            if n > PIVOT_MAX_LEVELS:
                sep = _detect_multivalue(df[col])[0]
                extra = (f" '{col}' holds lists — analyze_multivalue('{name}', '{col}') gives the "
                         f"real breakdown." if sep else
                         f" Use group_dataset for a flat summary, or pick a coarser column.")
                return (f"Error: '{col}' has {n:,} distinct values — too many to put on the "
                        f"{axis} of a grid (limit {PIVOT_MAX_LEVELS}).{extra}")

    try:
        if values is None:
            table = pd.pivot_table(df, index=rows, columns=columns or None,
                                   aggfunc="size", fill_value=0)
            measure = "row count"
        else:
            table = pd.pivot_table(df, index=rows, columns=columns or None,
                                   values=values, aggfunc=how, fill_value=0)
            measure = f"{how} of {values}"
    except Exception as e:  # noqa: BLE001 — errors as strings, per project contract
        return f"Error building pivot: {type(e).__name__}: {e}"

    if percent:
        total = float(np.nansum(np.asarray(table, dtype="float64")))
        if total:
            table = (table / total * 100).round(2)
            measure += " (as % of grand total)"

    flat = table.reset_index()
    flat.columns = [" ".join(str(p) for p in c if str(p) != "") if isinstance(c, tuple) else str(c)
                    for c in flat.columns]
    out = output_name or f"{name}_pivot"
    _register(out, flat)

    preview = table.head(20).to_string()
    lines = [
        f"Pivot of '{name}': {measure}",
        f"  rows = {rows}" + (f", columns = {columns}" if columns else ""),
        f"  {table.shape[0]} row(s) × {table.shape[1]} column(s) — registered as '{out}'",
        "",
        preview,
    ]
    if table.shape[0] > 20:
        lines.append(f"\n  ...{table.shape[0] - 20} more row(s) in '{out}'.")
    return "\n".join(lines)


# ─── 5. over time ──────────────────────────────────────────────────────────────

_FREQ_WORDS = {"D": "daily", "W": "weekly", "ME": "monthly", "M": "monthly",
               "QE": "quarterly", "Q": "quarterly", "YE": "yearly", "Y": "yearly"}

# How many periods back the same-period-last-year comparison sits, by frequency.
_SEASONAL_LAG = {"ME": 12, "M": 12, "QE": 4, "Q": 4, "W": 52, "D": 365}

# A period-on-period % change computed off a base this much smaller than the
# typical period is arithmetic, not news: a month that recorded 3 orders
# followed by one that recorded 110 is "+3,567%", and reporting that as the
# biggest rise buries whatever actually happened.
LOW_BASE_RATIO = float(os.environ.get("SWARN_LOW_BASE_RATIO", "0.2"))


def _period_end(stamp, freq: str):
    """The last instant the bucket labelled `stamp` covers."""
    try:
        return stamp.to_period(_PERIOD_ALIAS.get(freq, freq)).end_time
    except Exception:  # noqa: BLE001
        return None


_PERIOD_ALIAS = {"ME": "M", "QE": "Q", "YE": "Y", "W": "W", "D": "D",
                 "M": "M", "Q": "Q", "Y": "Y"}


def _partial_last_period(dates, series, freq: str) -> dict:
    """Is the final bucket still being filled?

    This is the single most common false alarm in reporting. An extract pulled
    on the 8th puts 8 days of sales in the newest month and compares them
    against a full 31 — the tool announces a 74% collapse, somebody takes it to
    a meeting, and the month ends up perfectly normal. Nothing about the shape
    of the data reveals it; the only evidence is that the last date in the file
    falls short of the end of the period it belongs to.

    Returns {} when the last period is complete.
    """
    if series.empty:
        return {}
    last_label = series.index[-1]
    end = _period_end(last_label, freq)
    latest = dates.max()
    if end is None or pd.isna(latest):
        return {}
    # Compared at DAY resolution, not to the last instant of the period. Daily
    # data stamps every row at midnight, so a month that ran to its final day
    # still ends 23h59m short of its own end_time — comparing exactly would
    # flag every complete month as unfinished, which is the same false alarm
    # this function exists to prevent, only inverted and firing constantly.
    try:
        if latest.normalize() >= end.normalize():
            return {}
    except AttributeError:
        if latest >= end:
            return {}
    start = getattr(last_label, "to_period", lambda _f: None)(_PERIOD_ALIAS.get(freq, freq))
    try:
        span = (end - start.start_time).total_seconds()
        done = (latest - start.start_time).total_seconds()
    except Exception:  # noqa: BLE001
        return {}
    if span <= 0:
        return {}
    share = max(0.0, min(1.0, done / span))
    if share >= 0.995:                       # complete to within rounding
        return {}
    return {"label": last_label, "share": share, "latest": latest, "end": end,
            "value": float(series.iloc[-1])}


def _year_on_year(series, freq: str):
    """Same period last year, which is the comparison most trends actually need.

    December always beats November and January always falls; against last
    December, neither is news. Returns None when the history is too short for
    the comparison to exist.
    """
    lag = _SEASONAL_LAG.get(freq)
    if not lag or len(series) <= lag:
        return None
    return (series / series.shift(lag) - 1.0) * 100


def analyze_over_time(name: str, date_col: str, freq: str = "ME",
                      value_col: Optional[str] = None, how: str = "sum",
                      output_name: Optional[str] = None) -> str:
    """Group by time period, chart the trend, and report period-on-period change."""
    df, err = _get_df(name)
    if err:
        return err
    if date_col not in df.columns:
        return f"Error: column '{date_col}' not in '{name}'. Columns: {list(df.columns)}"
    if value_col and value_col not in df.columns:
        return f"Error: column '{value_col}' not in '{name}'. Columns: {list(df.columns)}"

    if _is_numeric(df[date_col]) and not _is_datetime(df[date_col]):
        lo, hi = df[date_col].min(), df[date_col].max()
        looks_like_years = 1800 <= lo <= 2200 and 1800 <= hi <= 2200
        hint = (f" '{date_col}' looks like plain YEAR numbers — group by it directly with "
                f"group_dataset(name, keys=['{date_col}'], ...) instead."
                if looks_like_years else "")
        return (f"Error: '{date_col}' is a number column, not dates. Treating numbers as dates "
                f"would read them as nanoseconds since 1970 and put every row in 1970.{hint}")
    date_notes: list = []
    dates = _as_datetime(df[date_col], note=date_notes)
    if dates.notna().sum() < 2:
        return (f"Error: '{date_col}' does not look like dates "
                f"({int(dates.notna().sum())} parsed). Run uniform_dates on it first.")
    dropped = int(df[date_col].notna().sum() - dates.notna().sum())
    if dropped:
        date_notes.append(f"NOTE: {dropped:,} row(s) had an unreadable '{date_col}' and are "
                          f"excluded from every period below.")

    work = df.assign(_period=dates).dropna(subset=["_period"]).set_index("_period")
    try:
        if value_col is None:
            series = work.resample(freq).size()
            measure = "rows per period"
        else:
            series = getattr(work[value_col].resample(freq), how)()
            measure = f"{how} of '{value_col}' per period"
    except Exception as e:  # noqa: BLE001
        return (f"Error grouping by time: {type(e).__name__}: {e}. "
                "Try freq='D', 'W', 'ME', 'QE' or 'YE'.")

    series = series.dropna()
    if series.empty:
        return f"No periods produced for '{name}' — check '{date_col}'."

    change = series.pct_change() * 100
    partial = _partial_last_period(dates, series, freq)
    yoy = _year_on_year(series, freq)
    columns = {
        str(date_col): series.index.astype(str),
        "value": series.values,
        "change_pct": change.values.round(2),
    }
    if yoy is not None:
        columns["vs_same_period_last_year_pct"] = yoy.values.round(2)
    if partial:
        # Carried on the frame as well as in the text, so code that reads the
        # registered result rather than the message still knows.
        columns["period_complete"] = [True] * (len(series) - 1) + [False]
    result = pd.DataFrame(columns)
    out = output_name or f"{name}_over_time"
    _register(out, result)

    fig, ax = _new_figure()
    ax.plot(series.index, series.values, marker="o", markersize=3.5,
            color="#3E7CB1", linewidth=1.5)
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col or "rows")
    ax.set_title(f"{_FREQ_WORDS.get(freq, freq)} {measure} — '{name}'")
    fig.autofmt_xdate()
    rel = _save_fig(fig, f"{_safe_name(name)}__{_safe_name(date_col)}_{_safe_name(freq)}_trend.png")

    # Every headline below is computed on the SETTLED series. A period that is
    # still filling is not a smaller period, it is an unfinished one, and
    # letting it into a first-to-last comparison or a "biggest fall" is how a
    # report announces a collapse that is really just the current month.
    settled = series.iloc[:-1] if partial and len(series) > 1 else series
    settled_change = change.iloc[:-1] if partial and len(change) > 1 else change

    first, last = float(settled.iloc[0]), float(settled.iloc[-1])
    overall = (last - first) / abs(first) * 100 if first else float("nan")
    direction = "up" if last > first else "down" if last < first else "flat"
    lines = [
        f"Trend chart saved to {rel}",
        f"  {_FREQ_WORDS.get(freq, freq)} {measure}, {len(series)} period(s), "
        f"registered as '{out}'.",
        f"  {settled.index[0]:%Y-%m-%d} → {settled.index[-1]:%Y-%m-%d}: "
        f"{_fmt(first)} → {_fmt(last)}, {direction}"
        + (f" {overall:+.1f}%." if np.isfinite(overall) else "."),
    ]

    if partial:
        lines.append(
            f"  ⚠ THE LAST PERIOD IS NOT FINISHED. '{partial['label']:%Y-%m-%d}' covers up to "
            f"{partial['latest']:%Y-%m-%d} — about {partial['share']:.0%} of the period — but "
            f"it is charted against complete ones, so it will look like a fall whatever is "
            f"happening. Its value ({_fmt(partial['value'])}) is EXCLUDED from every figure "
            f"above and from the biggest rise/fall below. Do not report it as a decline; "
            f"compare it with the same {partial['share']:.0%} of an earlier period, or wait "
            f"for the period to close.")

    if len(settled_change.dropna()) > 1:
        # A change measured against a near-empty period is arithmetic, not news.
        typical = float(settled.median()) if len(settled) else 0.0
        base = settled.shift(1)
        trustworthy = settled_change[base.abs() >= abs(typical) * LOW_BASE_RATIO]
        noisy = int(len(settled_change.dropna()) - len(trustworthy.dropna()))
        source = trustworthy if len(trustworthy.dropna()) else settled_change
        biggest_up, biggest_down = source.idxmax(), source.idxmin()
        if pd.notna(biggest_up):
            lines.append(f"  Biggest rise: {biggest_up:%Y-%m-%d} ({source.max():+.1f}%).")
        if pd.notna(biggest_down):
            lines.append(f"  Biggest fall: {biggest_down:%Y-%m-%d} ({source.min():+.1f}%).")
        if noisy and len(trustworthy.dropna()):
            lines.append(
                f"  ({noisy} period(s) excluded from those two: each followed a period smaller "
                f"than {LOW_BASE_RATIO:.0%} of the typical {_fmt(typical)}, so their percentage "
                f"swings are an artefact of a near-empty base, not a real move.)")

    if yoy is not None:
        settled_yoy = yoy.iloc[:-1] if partial and len(yoy) > 1 else yoy
        recent = settled_yoy.dropna()
        if len(recent):
            lines.append(
                f"  vs SAME PERIOD LAST YEAR: latest is {recent.iloc[-1]:+.1f}%; "
                f"typical across the year {recent.median():+.1f}%. This is the comparison to "
                f"quote — period-on-period change mostly measures the calendar.")
    elif freq in _SEASONAL_LAG:
        lines.append(
            f"  No year-on-year comparison — the data covers {len(series)} period(s) and "
            f"{_SEASONAL_LAG[freq] + 1} are needed. Every change above is period-on-period, "
            f"which cannot separate a real move from an ordinary seasonal one.")
    # How the dates were read decides every number above it, so it is stated with
    # them rather than left for anyone to assume.
    lines += ["  " + n for n in date_notes]
    lines += ["", result.head(15).to_string(index=False)]
    if len(result) > 15:
        lines.append(f"\n  ...{len(result) - 15} more period(s) in '{out}'.")
    return "\n".join(lines)


# ─── effect size ───────────────────────────────────────────────────────────────
#
# A significance test answers "is this difference real?". It does not answer
# "is it big?", and the two come apart precisely where it matters most. With
# 500,000 rows a 0.3% gap between two groups returns p < 0.001 — genuinely
# real, and worth nobody's time. Reporting only the p-value licenses "Region A
# significantly outperforms Region B" for a difference no one could act on,
# which is a more credible way of being useless than saying nothing.
#
# So both are always reported: whether it is real, and how big it is. Cohen's d
# for two groups, eta-squared for more, each with the plain-language reading.

def _cohens_d(a, b) -> float:
    """Difference in means measured in standard deviations."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def _eta_squared(samples) -> float:
    """Share of the variation in the values explained by which group they are in."""
    values = np.concatenate([np.asarray(s, dtype=float) for s in samples])
    if len(values) < 2:
        return float("nan")
    grand = values.mean()
    between = sum(len(s) * (np.mean(s) - grand) ** 2 for s in samples)
    total = float(((values - grand) ** 2).sum())
    return float(between / total) if total > 0 else float("nan")


def _effect_words(kind: str, value: float) -> str:
    """Plain reading of an effect size — the sentence a stakeholder can use."""
    if not np.isfinite(value):
        return ""
    size = abs(value)
    if kind == "d":
        band = ("negligible" if size < 0.2 else "small" if size < 0.5
                else "moderate" if size < 0.8 else "large")
        return (f"Cohen's d = {value:+.2f} ({band}) — the groups overlap "
                f"{'almost completely' if size < 0.2 else 'heavily' if size < 0.5 else 'substantially' if size < 0.8 else 'only partly'}")
    band = ("negligible" if size < 0.01 else "small" if size < 0.06
            else "moderate" if size < 0.14 else "large")
    return (f"eta-squared = {value:.3f} ({band}) — which group a row is in explains "
            f"{value:.1%} of the variation in the values")


def _size_verdict(kind: str, value: float, significant: bool) -> str:
    """The one line that stops a real-but-tiny difference being reported as news."""
    if not np.isfinite(value):
        return ""
    negligible = abs(value) < (0.2 if kind == "d" else 0.01)
    if significant and negligible:
        return ("  ⚠ REAL BUT NOT MEANINGFUL. The test says this difference is unlikely to be "
                "chance, which with enough rows is true of almost any difference. The effect "
                "is negligible: the groups are practically the same. Do NOT describe this as "
                "one group outperforming another, and do not build a recommendation on it.")
    if significant:
        return "  This difference is both real and large enough to act on."
    if not negligible:
        return ("  The gap is sizeable but the test cannot rule out chance — usually too few "
                "rows. Worth measuring again with more data rather than dismissing.")
    return "  Neither real nor large. Treat the groups as the same."


# ─── 6. compare groups ─────────────────────────────────────────────────────────

def compare_groups(name: str, value_col: str, group_col: str) -> str:
    """Are the group averages really different, or is it noise?"""
    df, err = _get_df(name)
    if err:
        return err
    missing = [c for c in (value_col, group_col) if c not in df.columns]
    if missing:
        return f"Error: column(s) {missing} not in '{name}'. Columns: {list(df.columns)}"
    if not _is_numeric(df[value_col]):
        return f"Error: '{value_col}' must be a number column to compare averages."

    pair = df[[value_col, group_col]].dropna()
    grouped = {str(k): v[value_col].values for k, v in pair.groupby(pair[group_col].astype(str))}
    grouped = {k: v for k, v in grouped.items() if len(v) >= 2}
    if len(grouped) < 2:
        return (f"Error: need at least 2 groups with 2+ rows each in '{group_col}'; "
                f"found {len(grouped)}.")
    if len(grouped) > HIGH_CARDINALITY:
        return (f"Error: '{group_col}' has {len(grouped)} groups — too many to compare "
                "meaningfully. Group the data first, or pick a broader column.")

    summary = (pair.groupby(pair[group_col].astype(str))[value_col]
               .agg(["count", "mean", "std"]).sort_values("mean", ascending=False))
    lines = [f"Comparing '{value_col}' across '{group_col}' in '{name}':"]
    for label, row in summary.iterrows():
        lines.append(f"  {label}: average {_fmt(row['mean'])} "
                     f"({int(row['count']):,} rows)")

    try:
        from scipy import stats
    except ImportError:
        lines.append("  (install scipy for a significance test)")
        return "\n".join(lines)

    samples = list(grouped.values())
    if len(samples) == 2:
        stat, p = stats.ttest_ind(samples[0], samples[1], equal_var=False)
        test = "Welch t-test"
        effect_kind, effect = "d", _cohens_d(samples[0], samples[1])
    else:
        stat, p = stats.f_oneway(*samples)
        test = "one-way ANOVA"
        effect_kind, effect = "eta2", _eta_squared(samples)

    top, bottom = summary.index[0], summary.index[-1]
    gap = summary["mean"].iloc[0] - summary["mean"].iloc[-1]
    rel_gap = 100 * gap / abs(summary["mean"].iloc[-1]) if summary["mean"].iloc[-1] else float("nan")
    lines.append(f"  Widest gap: '{top}' is {_fmt(gap)} higher than '{bottom}'"
                 + (f" ({rel_gap:+.1f}%)." if np.isfinite(rel_gap) else "."))
    if pd.isna(p):
        lines.append(f"  {test} could not be computed (check for zero variance).")
        return "\n".join(lines)

    significant = bool(p < 0.05)
    if significant:
        lines.append(f"  IS IT REAL?  Unlikely to be chance ({test}, p = {p:.4g}).")
    else:
        lines.append(f"  IS IT REAL?  Could easily be chance ({test}, p = {p:.4g}) — "
                     "treat the groups as similar until you have more data.")
    words = _effect_words(effect_kind, effect)
    if words:
        lines.append(f"  IS IT BIG?   {words}.")
        verdict = _size_verdict(effect_kind, effect, significant)
        if verdict:
            lines.append(verdict)
    lines.append("  A real difference still does not tell you why it exists.")
    try:
        note_effect(name, {"value_col": value_col, "group_col": group_col,
                           "test": test, "p": float(p),
                           "effect_kind": effect_kind,
                           "effect": float(effect) if np.isfinite(effect) else None,
                           "significant": significant,
                           "negligible": bool(np.isfinite(effect) and
                                              abs(effect) < (0.2 if effect_kind == "d" else 0.01))})
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


# ─── 7. rankings ───────────────────────────────────────────────────────────────

def _ranking_noise(df: pd.DataFrame, group_col: str, value_col: Optional[str], how: str) -> tuple:
    """Is this league table telling you anything, and if so, what?

    Returns (p_counts, p_values, note). Two separate questions get conflated into
    one ranking:

      1. Do the groups differ in SIZE more than chance? A revenue table where
         every group is the same size but one is twice as big is a different
         finding from one where a group is simply twice as common.
      2. Given those sizes, does the value PER ROW differ by group?

    Reporting a total without separating them is how "peak ordering hours,
    19:00-21:00" survives when order counts across the 24 hours are flat, and how
    a revenue league table gets read as a per-order preference when it is purely
    a volume ranking.
    """
    try:
        from scipy import stats
    except ImportError:
        return None, None, ""

    labels = df[group_col].astype(str)
    counts = labels.value_counts()
    p_counts = None
    if len(counts) >= 2 and counts.sum() >= 5 * len(counts):
        try:
            p_counts = float(stats.chisquare(counts.values).pvalue)
        except Exception:  # noqa: BLE001
            p_counts = None

    p_values = None
    if value_col is not None and _is_numeric(df[value_col]) and how in ("sum", "mean"):
        groups = [g.dropna().values for _, g in df.groupby(labels)[value_col] if len(g.dropna()) >= 2]
        if len(groups) >= 2:
            try:
                p_values = float(stats.kruskal(*groups).pvalue)
            except Exception:  # noqa: BLE001
                p_values = None

    flat_counts = p_counts is not None and p_counts > 0.05
    flat_values = p_values is not None and p_values > 0.05

    if how == "count" or value_col is None:
        if flat_counts:
            return p_counts, p_values, (
                f"  NOISE CHECK: the {len(counts):,} '{group_col}' groups are NOT distinguishable "
                f"from an even split (chi-square p = {p_counts:.3g}). The order above is sampling "
                f"noise — do not call the top group a peak or the bottom one a weak spot.")
        if p_counts is not None:
            return p_counts, p_values, (
                f"  Noise check: group sizes really do differ (chi-square p = {p_counts:.3g}), so "
                f"the shape of this ranking is real.")
        return p_counts, p_values, ""

    if flat_counts and flat_values:
        return p_counts, p_values, (
            f"  NOISE CHECK: neither the group sizes (p = {p_counts:.3g}) nor the value per row "
            f"(p = {p_values:.3g}) differ from chance across these {len(counts):,} groups. This "
            f"table ranks noise — do not build a recommendation on its order.")
    if flat_values and p_counts is not None:
        return p_counts, p_values, (
            f"  NOISE CHECK: this ranking is driven by GROUP SIZE, not by value per row. Group "
            f"sizes differ far more than chance (p = {p_counts:.3g}), while {value_col} per row "
            f"does NOT differ between groups (Kruskal-Wallis p = {p_values:.3g}). The leader is "
            f"the biggest group, not the most valuable one — rank by how='mean' before calling it "
            f"an opportunity.")
    if flat_values:
        return p_counts, p_values, (
            f"  NOISE CHECK: {value_col} per row does NOT differ between these groups "
            f"(Kruskal-Wallis p = {p_values:.3g}), so the order above reflects how many rows each "
            f"group has, not how valuable it is. Rank by how='mean' before calling any group a "
            f"stronger performer.")
    if p_values is not None and not flat_values:
        return p_counts, p_values, (
            f"  Noise check: {value_col} per row genuinely differs between groups "
            f"(Kruskal-Wallis p = {p_values:.3g}), so this is more than a volume ranking — "
            f"rank by how='mean' as well to separate the two effects.")
    return p_counts, p_values, ""


_GENERIC_TOKENS = {"name", "id", "code", "key", "uuid", "sku", "no", "num", "number",
                   "title", "label", "desc", "description", "type", "value"}
_ID_TOKENS = {"id", "code", "key", "uuid", "sku", "no", "num", "number"}


def _tokens(column: str) -> set:
    return {t for t in re.split(r"[^0-9a-zA-Z]+", str(column).lower()) if t}


def _label_collisions(df: pd.DataFrame, group_col: str) -> list:
    """Warn when ranking by a NAME silently merges two different entities.

    Two products can share a name. Grouping by 'Product_Name' then sums them into
    one row that looks like a single best-seller and is not — the ranking names a
    product that does not exist, and the real one it displaced never appears. The
    giveaway is always sitting in the same frame: 70 Product_IDs, 68 Product_Names.

    Only sibling identifier columns are checked ('Product_Name' against
    'Product_ID', by shared stem), so grouping City by Order_ID — a deliberate
    many-to-one — never trips this.
    """
    stem = _tokens(group_col) - _GENERIC_TOKENS
    if not stem:
        return []
    notes = []
    for candidate in df.columns:
        if candidate == group_col:
            continue
        cand_tokens = _tokens(candidate)
        if not (stem & cand_tokens) or not (cand_tokens & _ID_TOKENS):
            continue
        pair = df[[group_col, candidate]].dropna()
        if pair.empty:
            continue
        per_label = pair.groupby(pair[group_col].astype(str))[candidate].nunique()
        clashing = per_label[per_label > 1]
        if clashing.empty:
            continue
        examples = []
        for label in list(clashing.index)[:3]:
            ids = sorted(pair.loc[pair[group_col].astype(str) == label, candidate].unique().tolist())
            examples.append(f"'{label}' = {candidate} " + "/".join(str(i) for i in ids[:4]))
        notes.append(
            f"  WARNING — RANKING MERGES DISTINCT ENTITIES: {len(clashing):,} of {len(per_label):,} "
            f"'{group_col}' value(s) cover more than one '{candidate}': "
            + "; ".join(examples) + ". "
            f"Each of those rows is two or more different things added together, so the ranking "
            f"above overstates them and pushes a genuine entry out of the table. Re-rank with "
            f"group_col='{candidate}' and use '{group_col}' only as the display label."
        )
    return notes


def rank_by(name: str, group_col: str, value_col: Optional[str] = None,
            how: str = "sum", top: int = 10, ascending: bool = False) -> str:
    """The ranked league table for a dimension — 'top 5 products by revenue'.

    Exists because reading a ranking off a bar chart is exactly where a blind
    agent goes wrong: adjacent bars are indistinguishable, so a near-tie gets
    reordered or a row is skipped entirely. The ranking is returned as numbered
    text AND recorded in the ledger, so a later claim about who is in the top N
    can be checked rather than trusted.
    """
    df, err = _get_df(name)
    if err:
        return err
    if group_col not in df.columns:
        return f"Error: column '{group_col}' not in '{name}'. Columns: {list(df.columns)}"
    if value_col is not None and value_col not in df.columns:
        return f"Error: column '{value_col}' not in '{name}'. Columns: {list(df.columns)}"
    how = (how or "sum").lower()
    if how not in ("sum", "mean", "count", "min", "max", "median"):
        return f"Error: how must be sum, mean, median, count, min or max — got '{how}'."
    if value_col is None:
        how = "count"
    elif not _is_numeric(df[value_col]):
        return (f"Error: '{value_col}' is not a number column, so it cannot be ranked by "
                f"{how}. Omit value_col to rank by row count instead.")
    top = max(1, int(top or 10))

    labels = df[group_col].astype(str)
    if value_col is None:
        series = labels.value_counts()
        measure = "row count"
    else:
        pair = pd.DataFrame({"_g": labels, "_v": df[value_col]}).dropna(subset=["_v"])
        if pair.empty:
            return f"Error: '{value_col}' has no values to rank in '{name}'."
        series = getattr(pair.groupby("_g")["_v"], how)()
        measure = f"{how} of {value_col}"
    series = series.sort_values(ascending=bool(ascending))
    if series.empty:
        return f"Error: '{group_col}' produced no groups to rank."

    # A share of a total is only meaningful for additive measures; the mean of
    # a group is not a slice of the mean of everything.
    additive = how in ("sum", "count")
    total = float(series.sum()) if additive else float("nan")

    p_counts, p_values, noise_note = _ranking_noise(df, group_col, value_col, how)

    order = [str(i) for i in series.index]
    ordered_values = {str(i): float(v) for i, v in series.items()}
    _EVIDENCE.setdefault("rankings", {})[(str(group_col), str(value_col or "row count"), how)] = {
        "order": order,
        "values": ordered_values,
        "total": total if additive else None,
        "ascending": bool(ascending),
        "dataset": str(name),
        "p_counts": p_counts,
        "p_values": p_values,
    }

    end = "lowest" if ascending else "highest"
    head = series.head(top)
    lines = [f"'{group_col}' ranked by {measure} in '{name}' — {len(series):,} group(s), "
             f"{end} first:"]
    running = 0.0
    for rank, (label, value) in enumerate(head.items(), start=1):
        running += float(value)
        share = ""
        if additive and total:
            share = (f"  ({100 * float(value) / total:.1f}% of total, "
                     f"{100 * running / total:.1f}% cumulative)")
        lines.append(f"  {rank}. {label}: {_fmt(value)}{share}")
    if len(series) > top:
        lines.append(f"  …{len(series) - top:,} more group(s) not shown.")
        tail = series.tail(min(3, len(series) - top))
        lines.append("  Bottom of the table: "
                     + ", ".join(f"{lbl} {_fmt(v)}" for lbl, v in tail.items()))
    if additive and total:
        lines.append(f"  Total across all groups: {_fmt(total)}. "
                     f"The top {len(head)} account for {100 * running / total:.1f}% of it.")

    # A near-tie at the cut-off is the thing a chart hides, so say it in words —
    # with the actual gap, because "within 5%" gets quoted back as the finding
    # when the true distance is a hundredth of a percent.
    if len(series) > top:
        boundary, following = float(series.iloc[top - 1]), float(series.iloc[top])
        if boundary and abs(boundary - following) / abs(boundary) < 0.05:
            gap = 100 * abs(boundary - following) / abs(boundary)
            gap_text = f"{gap:.2f}%" if gap >= 0.01 else "under 0.01%"
            lines.append(f"  CAUTION: rank {top} ({series.index[top - 1]}, {_fmt(boundary)}) and "
                         f"rank {top + 1} ({series.index[top]}, {_fmt(following)}) differ by "
                         f"{gap_text} — effectively a tie. Do not present that cut-off as a real "
                         f"gap, and do not report the 5% test itself as the size of the gap.")

    if noise_note:
        lines.append(noise_note)
    lines += _label_collisions(df, group_col)
    lines.append(f"  This ranking is now recorded; a report naming a different top {top} "
                 f"for '{group_col}' will be refused.")
    return "\n".join(lines)


# ─── 8. durations ──────────────────────────────────────────────────────────────

_DMY = re.compile(r"^\s*(\d{1,2})\D(\d{1,2})\D(\d{2,4})\s*$")


def _detect_dayfirst(series: pd.Series) -> Optional[bool]:
    """True/False when the column PROVES its own convention, None when it cannot.

    '24-03-2023' can only be day-first — no month is 24. That is evidence, not a
    guess, so it is used ahead of any heuristic. A column whose every value is
    ambiguous ('05-03-2023') returns None and is decided further down.
    """
    if _is_datetime(series) or _is_numeric(series):
        return None
    first_over_12 = second_over_12 = 0
    for value in series.dropna().astype(str).head(2000):
        match = _DMY.match(value)
        if not match:
            continue
        a, b = int(match.group(1)), int(match.group(2))
        if a > 12 and b <= 12:
            first_over_12 += 1
        elif b > 12 and a <= 12:
            second_over_12 += 1
    if first_over_12 and not second_over_12:
        return True
    if second_over_12 and not first_over_12:
        return False
    return None


def _parse_date_pair(start: pd.Series, end: pd.Series) -> tuple:
    """Parse two date columns so the GAP between them makes sense.

    Two columns in one file are often written differently ('2023-02-24' and
    '05-03-2023'). Parsing each on its own default is how '05-03-2023' becomes
    May 3rd instead of March 5th — every duration is then silently wrong and
    some go negative. So each column is read with its OWN convention, decided in
    this order:

      1. the column proves it ('24-03-2023' can only be day-first);
      2. otherwise, whichever reading yields fewer impossible (negative) gaps
         and fewer unreadable rows;
      3. if still tied, the shorter typical gap — between two equally supported
         readings of an elapsed time, three days beats seven months.

    Step 3 is a heuristic, which is exactly why the chosen convention is always
    reported back rather than applied quietly.
    """
    def parse(series: pd.Series, dayfirst: bool) -> pd.Series:
        if _is_datetime(series):
            return series
        if _is_numeric(series):
            return pd.Series(pd.NaT, index=series.index)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)

    proven_start, proven_end = _detect_dayfirst(start), _detect_dayfirst(end)
    starts = [proven_start] if proven_start is not None else [False, True]
    ends = [proven_end] if proven_end is not None else [False, True]

    best = None
    for sf in starts:
        for ef in ends:
            s, e = parse(start, sf), parse(end, ef)
            delta = (e - s).dt.total_seconds()
            usable = delta.dropna()
            typical = float(usable.abs().median()) if len(usable) else float("inf")
            score = (int((delta < 0).sum()), int(delta.isna().sum()), typical)
            if best is None or score < best[0]:
                best = (score, s, e, sf, ef)
    (negative, unparsed, _typical), parsed_start, parsed_end, sf, ef = best
    return parsed_start, parsed_end, (sf, ef), negative, unparsed


_DURATION_UNITS = {"days": 86400.0, "hours": 3600.0, "minutes": 60.0, "seconds": 1.0,
                   "weeks": 604800.0}


def measure_duration(name: str, start_col: str, end_col: str, unit: str = "days",
                     by: Optional[str] = None) -> str:
    """How long the gap between two dates is — 'average order to delivery time'.

    Nothing else in this module measures elapsed time, so this figure previously
    had to be eyeballed or done by hand. It is a headline KPI in most operational
    reports, so it is computed and recorded here instead.
    """
    df, err = _get_df(name)
    if err:
        return err
    missing = [c for c in (start_col, end_col) if c not in df.columns]
    if by and by not in df.columns:
        missing.append(by)
    if missing:
        return f"Error: column(s) {missing} not in '{name}'. Columns: {list(df.columns)}"
    unit = (unit or "days").lower()
    if unit not in _DURATION_UNITS:
        return f"Error: unit must be one of {sorted(_DURATION_UNITS)} — got '{unit}'."

    start, end, conventions, negative, unparsed = _parse_date_pair(df[start_col], df[end_col])
    if start.isna().all() or end.isna().all():
        return (f"Error: '{start_col}' and/or '{end_col}' could not be read as dates. "
                f"A number column is refused here — pandas would read it as nanoseconds "
                f"since 1970 and put every row in 1970.")

    gap = (end - start).dt.total_seconds() / _DURATION_UNITS[unit]
    usable = gap.dropna()
    if usable.empty:
        return f"Error: no row in '{name}' has both '{start_col}' and '{end_col}' filled."

    lines = [f"Time from '{start_col}' to '{end_col}' in '{name}', in {unit}:",
             f"  average {usable.mean():,.2f}  ·  median {usable.median():,.2f}  ·  "
             f"range {usable.min():,.2f} to {usable.max():,.2f}",
             f"  measured on {len(usable):,} of {len(df):,} rows."]
    worded = ["day-first (DD-MM-YYYY)" if flag else "month-first (MM-DD-YYYY)"
              for flag in conventions]
    # A day/month convention only exists for values that HAVE an ambiguous
    # layout. ISO '2023-02-24' can be read exactly one way, so announcing that
    # it was "read month-first" is noise that invites a pointless double-check.
    ambiguous = [_dmy_shaped(df[c]) > 0 for c in (start_col, end_col)]
    if not any(ambiguous):
        lines.append("  Both date columns are written unambiguously (ISO YYYY-MM-DD), "
                     "so no day/month convention had to be chosen.")
    else:
        if worded[0] == worded[1]:
            lines.append(f"  Both date columns were read {worded[0]}.")
        else:
            lines.append(f"  '{start_col}' was read {worded[0]} and '{end_col}' was read "
                         f"{worded[1]} — the two columns are NOT written the same way.")
        lines.append("  That reading was the one this file supports; if it is the wrong "
                     "convention here, every figure above is wrong.")
    if unparsed:
        lines.append(f"  {unparsed:,} row(s) had a date that could not be read and are excluded.")
    if negative:
        lines.append(f"  WARNING: {negative:,} row(s) ({negative / len(df):.1%}) have "
                     f"'{end_col}' BEFORE '{start_col}'. That is impossible for an elapsed "
                     f"time, so either the columns are the wrong way round or those rows are "
                     f"bad. Do not quote the average until this is resolved.")

    _EVIDENCE.setdefault("durations", {})[(str(start_col), str(end_col))] = {
        "unit": unit, "mean": float(usable.mean()), "median": float(usable.median()),
        "count": int(len(usable)), "negative": int(negative), "dataset": str(name),
        "dayfirst": [bool(f) for f in conventions],
    }

    if by:
        grouped = pd.DataFrame({"_g": df[by].astype(str), "_d": gap}).dropna(subset=["_d"])
        summary = grouped.groupby("_g")["_d"].agg(["count", "mean"]).sort_values(
            "mean", ascending=False)
        if len(summary) > HIGH_CARDINALITY:
            lines.append(f"  '{by}' has {len(summary):,} groups — too many to break down; "
                         f"showing the 10 slowest.")
            summary = summary.head(10)
        lines.append(f"  By '{by}' (slowest first):")
        for label, row in summary.iterrows():
            lines.append(f"    {label}: {row['mean']:,.2f} {unit} ({int(row['count']):,} rows)")
    lines.append("  An average gap describes the rows that have both dates — it says nothing "
                 "about orders still outstanding.")
    return "\n".join(lines)


# ─── 9. the sweep ──────────────────────────────────────────────────────────────

def analyze_dataset(name: str, target: Optional[str] = None) -> str:
    """One-call overview: shape, what each column is, what stands out, what to
    do next. Built for the agent — it replaces a dozen exploratory calls."""
    df, err = _get_df(name)
    if err:
        return err
    n_rows, n_cols = df.shape
    if not n_rows:
        return f"'{name}' has no rows."

    lines = [f"ANALYSIS OF '{name}' — {n_rows:,} rows × {n_cols} columns", ""]

    numeric, categorical, dates, ids, constants, marker_cols = [], [], [], [], [], []
    coded: list = []
    for col in df.columns:
        s = df[col]
        nun = s.nunique(dropna=True)
        if _is_marker(col):
            marker_cols.append(col)
        elif nun <= 1:
            constants.append(col)
        elif _is_numeric(s):
            (coded if _looks_like_coded_categorical(s) else numeric).append(col)
        elif _is_datetime(s) or (_is_texty(s)
                                 and _as_datetime(s.dropna().head(200)).notna().mean() >= 0.8):
            dates.append(col)
        elif nun == n_rows:
            ids.append(col)
        else:
            categorical.append(col)

    lines.append("WHAT'S IN IT")
    lines.append(f"  {len(numeric)} number column(s): {numeric or '—'}")
    lines.append(f"  {len(categorical)} category column(s): {categorical or '—'}")
    lines.append(f"  {len(dates)} date column(s): {dates or '—'}")
    if coded:
        lines.append(f"  {len(coded)} code/label column(s) — stored as numbers but used as "
                     f"labels: {coded}")
        lines.append(f"     Their average and their correlations are meaningless (the mean of "
                     f"an error code is not an error code). Treat them as categories: "
                     f"rank_by('{name}', '{coded[0]}') or "
                     f"plot_relationship('{name}', '{coded[0]}', '<a real measure>').")
    if ids:
        lines.append(f"  {len(ids)} all-different column(s), probably IDs: {ids}")
    if constants:
        lines.append(f"  {len(constants)} column(s) with only one value: {constants}")
    if marker_cols:
        lines.append(f"  {len(marker_cols)} cleaning-marker column(s) recording what cleaning "
                     f"did — left out of the analysis below: {marker_cols}")

    # ── things worth knowing ──
    findings: list[str] = []
    nulls = df.isnull().mean().sort_values(ascending=False)
    bad_nulls = nulls[nulls > 0.2]
    for col, frac in bad_nulls.head(5).items():
        findings.append(f"'{col}' is {frac:.0%} empty — decide whether to fill or drop it.")
    # blanks that co-occur mark a distinct group of records, not scattered gaps
    miss = df.isnull()
    gappy = [c for c in df.columns if not _is_marker(c) and 0.01 <= miss[c].mean() < 1.0]
    for block in _missing_blocks(miss, gappy):
        n = int(miss[block].all(axis=1).sum())
        findings.append(f"{block} go blank in the SAME {n:,} row(s) — one group of incomplete "
                        f"records, not random gaps. Imputing each separately invents whole "
                        f"records. Run analyze_missing('{name}').")
    dups = int(df.duplicated().sum())
    if dups:
        findings.append(f"{dups:,} duplicate row(s) — run clean_dataset to remove them.")
    # surface what cleaning invented BEFORE any distribution is described
    for col in numeric + categorical:
        for warning in _cleaning_warnings(df, col):
            findings.append(f"'{col}' — {warning}")
    for col in numeric[:12]:
        s = df[col].dropna()
        if len(s) < 8:
            continue
        n_out, lo, hi = _outlier_count(s)
        if n_out and n_out / len(s) >= 0.01:
            findings.append(f"'{col}' has {n_out:,} extreme value(s) outside "
                            f"[{_fmt(lo)}, {_fmt(hi)}] — they will distort the average.")
        elif len(s) > 20 and s.median() and s.mean() > s.median() * 1.5:
            findings.append(f"'{col}' is lopsided (average {_fmt(s.mean())} vs middle "
                            f"{_fmt(s.median())}) — use the middle value, not the average.")
    multivalue_cols = []
    for col in categorical[:12]:
        sep, combos, singles = _detect_multivalue(df[col])
        if sep:
            multivalue_cols.append(col)
            findings.append(f"'{col}' holds LISTS, not single categories — {combos} combinations "
                            f"built from only {singles} real values. Use "
                            f"analyze_multivalue('{name}', '{col}') or every per-category "
                            f"comparison will be nonsense.")
            continue
        counts = df[col].value_counts(normalize=True)
        if len(counts) and counts.iloc[0] >= 0.9:
            # ':.0%' turns 99.5% into "100%", which reads as "there is only one
            # value here" and hides the minority entirely. In fraud, churn,
            # defect and adverse-event data that minority IS the analysis, so the
            # share is floored below 100 and the rare rows are always counted.
            raw = df[col].value_counts()
            share = counts.iloc[0]
            shown = f"{share:.1%}" if share < 0.9995 else f"{share:.3%}"
            rest = int(len(df[col].dropna()) - raw.iloc[0])
            if rest:
                others = ", ".join(f"{_clean_label(v, 18)}={c:,}" for v, c in raw.iloc[1:4].items())
                findings.append(
                    f"'{col}' is {shown} '{counts.index[0]}' — heavily imbalanced, but the "
                    f"remaining {rest:,} row(s) are NOT nothing ({others}"
                    f"{', …' if len(raw) > 4 else ''}). If those rare rows are what you are "
                    f"studying, report counts rather than percentages; if not, the column "
                    f"carries little signal.")
            else:
                findings.append(f"'{col}' is entirely '{counts.index[0]}' — a constant, "
                                f"so it cannot explain anything.")
        elif len(counts) > HIGH_CARDINALITY:
            findings.append(f"'{col}' has {len(counts):,} different values — too many to "
                            "chart directly; group it first.")
    lines += ["", "WORTH KNOWING"]
    lines += [f"  • {f}" for f in (findings[:10] or ["Nothing unusual stood out."])]

    # ── relationships ──
    num_df = df[numeric].loc[:, df[numeric].nunique(dropna=True) > 1] if numeric else pd.DataFrame()
    if num_df.shape[1] >= 2 and len(num_df.dropna(how="all")) >= CORR_MIN_ROWS:
        corr = num_df.corr()
        if target and target in corr.columns:
            ranked = corr[target].drop(index=target).sort_values(key=abs, ascending=False)
            lines += ["", f"WHAT MOVES WITH '{target}'"]
            for col, r in ranked.head(6).items():
                lines.append(f"  {col}: {r:+.2f} ({_strength(r)})")
        else:
            pairs = []
            cols = list(corr.columns)
            for i, a in enumerate(cols):
                for b in cols[i + 1:]:
                    r = corr.loc[a, b]
                    if pd.notna(r):
                        pairs.append((abs(r), a, b, r))
            pairs.sort(reverse=True)
            strong = [p for p in pairs if p[0] >= 0.4][:6]
            lines += ["", "WHAT RELATES TO WHAT"]
            lines += ([f"  {a} ↔ {b}: {r:+.2f} ({_strength(r)})" for _, a, b, r in strong]
                      or ["  No strong relationships between the number columns."])

    if dates:
        parsed = _as_datetime(df[dates[0]]).dropna()
        if len(parsed):
            lines += ["", "TIME RANGE",
                      f"  '{dates[0]}' runs {parsed.min():%Y-%m-%d} → {parsed.max():%Y-%m-%d}."]

    lines += ["", "SUGGESTED NEXT STEPS"]
    steps = []
    if bad_nulls.any() or dups:
        steps.append("clean_dataset — deal with the blanks and duplicates above")
    if numeric:
        steps.append(f"plot_column('{name}', '{numeric[0]}') — see the shape of a key number")
    # only suggest a category that can actually be charted — suggesting a
    # 6,000-value column two lines after warning about it reads as broken
    if multivalue_cols:
        target_num = numeric[0] if numeric else None
        steps.append(f"analyze_multivalue('{name}', '{multivalue_cols[0]}'"
                     + (f", value_col='{target_num}'" if target_num else "")
                     + ") — split the lists before comparing them")
    chartable = [c for c in categorical
                 if c not in multivalue_cols and df[c].nunique(dropna=True) <= HIGH_CARDINALITY]
    if numeric and chartable:
        steps.append(f"plot_relationship('{name}', '{chartable[0]}', '{numeric[0]}') — compare groups")
    elif numeric and categorical:
        steps.append(f"pivot_dataset('{name}', rows=['{categorical[0]}'], values='{numeric[0]}') "
                     "— too many categories to chart; summarise them first")
    if dates and numeric:
        steps.append(f"analyze_over_time('{name}', '{dates[0]}', value_col='{numeric[0]}') — see the trend")
    if len(numeric) >= 2:
        steps.append(f"analyze_correlations('{name}') — full relationship map")
    if bad_nulls.any() or _missing_blocks(miss, gappy):
        steps.append(f"analyze_missing('{name}') — are the blanks random, or one distinct group?")
    lines += [f"  {i}. {s}" for i, s in enumerate(steps or ["The data looks ready to use."], 1)]
    return "\n".join(lines)


# ─── swarn tool registration ───────────────────────────────────────────────────

# ─── reconciliation ────────────────────────────────────────────────────────────
#
# Every experienced analyst does this before any meeting, and no tool in this
# package did it: check the number against the one the business already has.
#
#     "My total revenue is 4.2 crore. Finance says 4.35. Why the gap?"
#
# You find that out BEFORE the meeting, not during it — because if your figure
# does not match the number everyone already trusts, nothing else you say gets
# heard, however carefully it was measured. Note that this is the one check in
# the whole package that reaches outside the data: everything else verifies the
# analysis against itself, which cannot catch a file that was the wrong extract
# in the first place.

# Rounding, timing and currency conversion routinely move a total by a fraction
# of a percent. Below this the two figures are the same number.
RECONCILE_TOLERANCE = float(os.environ.get("SWARN_RECONCILE_TOLERANCE", "0.005"))


def reconcile(name: str, column: str, expected: float, label: str = "the expected figure",
              how: str = "sum") -> str:
    """Compare a figure computed here against one from outside this analysis."""
    df, err = _get_df(name)
    if err:
        return err
    if column not in df.columns:
        return f"Error: column '{column}' not in '{name}'. Columns: {list(df.columns)}"
    if not _is_numeric(df[column]):
        return f"Error: '{column}' is not a number column, so it has no total to reconcile."
    try:
        expected = float(expected)
    except (TypeError, ValueError):
        return f"Error: expected must be a number — got {expected!r}."

    series = pd.to_numeric(df[column], errors="coerce")
    if how not in ("sum", "mean", "count", "min", "max", "nunique"):
        return "Error: how must be one of sum, mean, count, min, max, nunique."
    actual = float(getattr(series.dropna(), how)())
    gap = actual - expected
    rel = gap / abs(expected) if expected else float("nan")

    lines = [f"Reconciling {how} of '{column}' in '{name}' against {label}:",
             f"  this analysis : {actual:,.2f}",
             f"  {label:<14}: {expected:,.2f}",
             f"  difference    : {gap:+,.2f}"
             + (f" ({rel:+.2%})" if np.isfinite(rel) else "")]

    matches = np.isfinite(rel) and abs(rel) <= RECONCILE_TOLERANCE
    if matches:
        lines.append(f"  ✓ MATCHES within {RECONCILE_TOLERANCE:.1%}. This figure can be quoted "
                     f"as consistent with {label}.")
    else:
        lines.append(f"  ✗ DOES NOT MATCH. Do not present this number until the gap is "
                     f"explained — a figure that disagrees with {label} will be disbelieved, "
                     f"and it is usually right to disbelieve it.")
        # The causes, in the order they actually turn out to be true.
        causes = []
        cut = _EVIDENCE.get("truncation", {}).get(name)
        if cut:
            causes.append(f"only {cut['rows_read']:,} of {cut['rows_total']:,} rows were read "
                          f"— this alone would make the figure too low")
        for join in _EVIDENCE.get("joins", []):
            if join.get("output") == name and join.get("dropped_rows"):
                causes.append(f"a join dropped {join['dropped_rows']:,} row(s), which removes "
                              f"their contribution")
            if join.get("output") == name and join.get("right_duplicate_keys"):
                causes.append("a join duplicated rows, which inflates any total over them")
        grain = _EVIDENCE.get("grain", {}).get(name)
        if grain and grain.get("extra_rows"):
            causes.append(f"{grain['extra_rows']:,} row(s) duplicate a "
                          f"{'/'.join(grain['keys'])}, so they are counted twice")
        blank = int(df[column].isnull().sum())
        if blank:
            causes.append(f"{blank:,} row(s) have no '{column}' and contribute nothing")
        if gap < 0:
            causes.append("rows may be filtered out that the other figure includes "
                          "(cancelled orders, a different date range, a region left out)")
        else:
            causes.append("rows may be included that the other figure excludes "
                          "(test orders, internal transfers, refunds counted as sales)")
        lines.append("  Where the difference usually comes from, most likely first:")
        lines += [f"    {i}. {c}" for i, c in enumerate(causes[:5], 1)]

    try:
        note_reconciliation({"dataset": name, "column": column, "how": how,
                             "actual": actual, "expected": expected, "label": str(label),
                             "gap": gap, "relative": float(rel) if np.isfinite(rel) else None,
                             "matches": bool(matches)})
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


_SWARN_TOOLS = {
    "reconcile": (
        "Check a figure computed here against a number the business already has — the total "
        "revenue finance reports, a row count from the source system, last month's published "
        "figure. Do this BEFORE presenting any headline number. A figure that disagrees with "
        "the one everybody already trusts will be disbelieved whatever the analysis behind it, "
        "and it is usually right to disbelieve it. This is the only check that reaches outside "
        "the data: every other tool here verifies the analysis against itself, which cannot "
        "catch having been given the wrong extract. "
        "When the numbers disagree it lists the likely causes in order, using what it already "
        "knows about this dataset — a capped read, a join that dropped or duplicated rows, "
        "duplicate keys, blank values.",
        {"type": "object",
         "properties": {
             "name": {"type": "string", "description": "Name of a loaded dataset."},
             "column": {"type": "string", "description": "The number column to total, e.g. 'revenue'."},
             "expected": {"type": "number", "description": "The figure from outside this analysis to check against."},
             "label": {"type": "string", "description": "Where that figure came from, e.g. 'the finance report' — quoted back in the output."},
             "how": {"type": "string", "description": "sum (default), mean, count, min, max, or nunique."},
         },
         "required": ["name", "column", "expected"]},
        reconcile,
    ),
    "analyze_dataset": (
        "START HERE for any 'analyse/explore/what's in this data' request. One-call overview "
        "of a loaded dataset: column roles, what stands out (blanks, duplicates, extreme values, "
        "lopsided distributions), which columns relate to each other, the time range, and a "
        "suggested next step. Changes nothing. Prefer this over calling several plot tools blind.",
        {"type": "object",
         "properties": {
             "name": {"type": "string", "description": "Name of a loaded dataset."},
             "target": {"type": "string", "description": "Optional column of interest — everything is ranked against it."},
         },
         "required": ["name"]},
        analyze_dataset,
    ),
    "plot_column": (
        "Chart one column and describe its shape in words. Numbers get a histogram plus "
        "min/middle/max, spread and extreme-value count; categories get a bar chart plus the "
        "most common values and their share; dates get a timeline. The chart type is chosen "
        "automatically from the column. Saves a PNG under workspace/plots/ and returns both "
        "the path and the reading — you cannot see the image, so use the words.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "column": {"type": "string"},
             "bins": {"type": "integer", "description": "Histogram bars for number columns (default 30)."},
             "top": {"type": "integer", "description": "How many bars for a category column (default 15)."},
             "log_scale": {"type": "boolean", "description": "Use a log x-axis. Essential for columns spanning orders of magnitude (vote counts, revenue) — a linear axis squashes them into one bar."},
         },
         "required": ["name", "column"]},
        plot_column,
    ),
    "analyze_multivalue": (
        "For a column whose cells hold LISTS — 'Action, Adventure, Comedy', tags, skills. Splits "
        "them and analyses the individual values, so 500 meaningless combinations become ~25 real "
        "categories. Pass value_col to rank them by an average (e.g. which genre rates highest). "
        "Rows with several values are counted once per value, so shares exceed 100%. Registers the "
        "breakdown as a dataset. Use this INSTEAD of plot_column/compare_groups on such a column — "
        "comparing the raw combinations is meaningless.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "column": {"type": "string", "description": "The list-valued column."},
             "value_col": {"type": "string", "description": "Optional number column to average per value."},
             "how": {"type": "string", "description": "mean (default) or median."},
             "sep": {"type": "string", "description": "Separator; auto-detected (',' ';' '|') if omitted."},
             "min_rows": {"type": "integer", "description": "Ignore values with fewer rows than this (default 5)."},
             "top": {"type": "integer"},
             "output_name": {"type": "string"},
         },
         "required": ["name", "column"]},
        analyze_multivalue,
    ),
    "plot_relationship": (
        "Chart two columns together and describe how they relate. Picks the right view from the "
        "column types: two numbers → scatter with trend line and correlation; number and category "
        "→ box plot with per-group averages; date and number → trend line with the change; two "
        "categories → counts grid. Saves a PNG and returns the path plus the reading in words.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "x": {"type": "string", "description": "First column (the date, if there is one)."},
             "y": {"type": "string", "description": "Second column (the measure, if there is one)."},
         },
         "required": ["name", "x", "y"]},
        plot_relationship,
    ),
    "analyze_missing": (
        "Are the blanks random accidents, or one distinct group of records? Reports the "
        "percentage empty per column AND — the important part — which columns go blank in the "
        "SAME rows. Columns that always go blank together mark a structurally different set of "
        "rows (e.g. unreleased titles with no rating, no votes and no runtime); imputing each "
        "one separately fabricates a whole record. Call this BEFORE approving impute operations "
        "on a dataset with several gappy columns.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "min_frac": {"type": "number", "description": "Ignore columns emptier than this fraction (default 0.01)."},
             "top": {"type": "integer"},
         },
         "required": ["name"]},
        analyze_missing,
    ),
    "check_subgroups": (
        "Does a correlation actually hold, or is it an average of opposite effects? Recomputes "
        "the x~y relationship inside each group of `by` and flags any group where it reverses or "
        "is far stronger/weaker. A pooled correlation can be true overall and wrong for every "
        "group in it. Run this before reporting any correlation as a finding.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "x": {"type": "string", "description": "First number column."},
             "y": {"type": "string", "description": "Second number column."},
             "by": {"type": "string", "description": "Category column to split on. List-valued columns are grouped by their first value."},
             "min_rows": {"type": "integer", "description": "Skip groups smaller than this (default 30 — fewer is noise)."},
             "top": {"type": "integer"},
         },
         "required": ["name", "x", "y", "by"]},
        check_subgroups,
    ),
    "analyze_correlations": (
        "Which number columns move together. Saves a heatmap and returns a ranked list in words "
        "('revenue ↔ qty: +0.72 strong'). Pass target to rank every column against that one — "
        "this is how you answer 'what drives X'. Reports association, never causation.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "target": {"type": "string", "description": "Optional column to rank everything against."},
             "method": {"type": "string", "description": "pearson (default), spearman or kendall."},
         },
         "required": ["name"]},
        analyze_correlations,
    ),
    "pivot_dataset": (
        "Cross-tab grid — categories down the side, categories across the top, a measure in the "
        "cells (e.g. rows=['region'], columns=['month'], values='revenue', how='sum'). Set "
        "percent=true for share of the grand total. Registers the grid as a new dataset "
        "('<name>_pivot') so it can be charted or saved. Use group_dataset for a flat summary.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "rows": {"type": "array", "items": {"type": "string"}, "description": "Column(s) down the side."},
             "columns": {"type": "array", "items": {"type": "string"}, "description": "Optional column(s) across the top."},
             "values": {"type": "string", "description": "Column to aggregate. Omit to count rows."},
             "how": {"type": "string", "description": "sum (default), mean, count, min, max."},
             "percent": {"type": "boolean", "description": "Show each cell as % of the grand total."},
             "output_name": {"type": "string"},
         },
         "required": ["name", "rows"]},
        pivot_dataset,
    ),
    "analyze_over_time": (
        "Group a dataset into time periods and show the trend: daily/weekly/monthly/quarterly "
        "totals with the percentage change from the period before, the biggest rise and fall, and "
        "a saved line chart. Registers the result as '<name>_over_time'. freq: 'D', 'W', 'ME' "
        "(month, default), 'QE', 'YE'.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "date_col": {"type": "string", "description": "Column holding the dates."},
             "freq": {"type": "string", "description": "D, W, ME (default), QE or YE."},
             "value_col": {"type": "string", "description": "Column to total. Omit to count rows."},
             "how": {"type": "string", "description": "sum (default), mean, min, max."},
             "output_name": {"type": "string"},
         },
         "required": ["name", "date_col"]},
        analyze_over_time,
    ),
    "compare_groups": (
        "Is the gap between groups real, or just noise? Reports each group's average and row "
        "count, the widest gap, and a significance test (Welch t-test for two groups, one-way "
        "ANOVA for more) translated into plain language. Use after plot_relationship shows "
        "groups that look different.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "value_col": {"type": "string", "description": "The number being compared."},
             "group_col": {"type": "string", "description": "The column that splits the rows into groups."},
         },
         "required": ["name", "value_col", "group_col"]},
        compare_groups,
    ),
    "rank_by": (
        "The league table for a dimension — 'top 5 products by revenue', 'cities by order "
        "count'. Returns a NUMBERED list with each group's share and the running cumulative "
        "share, warns when the cut-off is a near-tie, and names the bottom of the table. Use "
        "this for EVERY 'top N' or 'best/worst' statement you intend to make: reading a "
        "ranking off a bar chart reorders near-ties and drops rows. The ranking is recorded, "
        "and a report naming a different top N will be refused.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "group_col": {"type": "string", "description": "The dimension to rank (product, city, category…)."},
             "value_col": {"type": "string", "description": "The number to rank by. Omit to rank by row count."},
             "how": {"type": "string", "description": "sum (default), mean, median, count, min, max."},
             "top": {"type": "integer", "description": "How many to list (default 10)."},
             "ascending": {"type": "boolean", "description": "true for the worst performers first."},
         },
         "required": ["name", "group_col"]},
        rank_by,
    ),
    "measure_duration": (
        "Elapsed time between two date columns — 'average order to delivery time'. Returns "
        "average, median and range, and breaks it down by a category if you pass `by`. Handles "
        "the case where the two columns are written in different date formats (parsing each "
        "alone turns '05-03-2023' into May 3rd and silently corrupts every gap), states which "
        "convention it used, and warns loudly if any end date falls before its start date. Use "
        "this for any 'how long does X take' KPI.",
        {"type": "object",
         "properties": {
             "name": {"type": "string"},
             "start_col": {"type": "string", "description": "The earlier date column (e.g. order date)."},
             "end_col": {"type": "string", "description": "The later date column (e.g. delivery date)."},
             "unit": {"type": "string", "description": "days (default), hours, minutes, seconds or weeks."},
             "by": {"type": "string", "description": "Optional category column to break the average down by."},
         },
         "required": ["name", "start_col", "end_col"]},
        measure_duration,
    ),
}


def register_into_swarn() -> str:
    """Register the analysis/visualisation tools into agent.tools.TOOL_REGISTRY."""
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
    return f"registered {registered} data-analysis tool(s) into swarn"
