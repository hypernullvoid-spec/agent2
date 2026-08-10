"""
The analyst - works with Claude, GPT or Llama without any code change.

Every provider difference lives in llm.py. This file just asks questions.

    export ANTHROPIC_API_KEY=...   (or GROQ_API_KEY, or OPENAI_API_KEY)
    python3 analyst.py "Which genre has the highest average rating?"
"""

import os
import re
import sys

import llm
import loader

# Columns that must always be sent. n_rows above all: without it the model
# cannot weight averages and will average the averages, which is the wrong sum.
ALWAYS_SEND = ("n_rows",)


# ----------------------------------------------------------------------
# DESCRIBE THE COLUMNS
# ----------------------------------------------------------------------

def describe_columns(rows):
    """A one-line note per column, built automatically from the data."""
    weights = {col: (pct, tok) for col, pct, tok in loader.column_weights(rows)}
    notes = {}

    for col in rows[0]:
        percent, tokens = weights[col]
        # str() because rolled-up rows contain numbers, not just text
        values = [str(r[col]) for r in rows if r.get(col) not in (None, "")]
        unique = len(set(values))
        avg_length = sum(len(v) for v in values) / max(1, len(values))

        # HEAVY needs BOTH a big share AND real size - 38% of a 268-token
        # summary table is not heavy.
        if (percent > 20 and tokens > 5_000) or avg_length > 60:
            kind = "HEAVY - only if the question really needs it"
        elif unique <= 25:
            kind = f"{unique} values: {sorted(set(values))[:6]}"
        else:
            kind = f"text, {unique:,} different values"

        notes[col] = f"{kind} (~{tokens:,} tokens)"

    return notes


# ----------------------------------------------------------------------
# LET THE QUESTION DECIDE THE SHAPE
# ----------------------------------------------------------------------

def pick_grouping(question, columns):
    """
    How should the data be grouped for this question?

    This matters more than picking columns. Grouped correctly, the answer is a
    single row the model can read. Grouped wrongly it has to average averages,
    which it does badly and which is the wrong sum anyway.
    """
    reply = llm.chat(f"""Available columns: {', '.join(columns)}

The user asks: "{question}"

Which columns should the data be GROUPED BY to answer this?
Use the fewest possible - add a second only if the question compares across it
(over time, by region, and so on).

Reply with just the column names, comma separated. Nothing else.""",
                     tier="small", max_tokens=100)

    chosen = [c for c in columns if c.lower() in reply.lower()]

    # The small model is not reliable at this. If the question names a column
    # outright - "which GENRE rates highest" - that column must be in the
    # grouping whatever the model said. Cheap check, catches the worst errors.
    named = [c for c in columns if _mentions(question, c)]

    # A question that names its column and asks for no comparison needs ONLY
    # that column. Adding a second forces the model to aggregate rows itself,
    # which is exactly what we group the data to avoid.
    if named and not _wants_comparison(question):
        return named[:2]

    for col in reversed(named):
        if col not in chosen:
            chosen.insert(0, col)

    return chosen[:2] or [columns[0]]


COMPARISON_WORDS = (
    "over time", "trend", "changed", "change", "improve", "decline", "grow",
    "each year", "by year", "yearly", "history", "evolv", "progress",
    "compare", "versus", " vs ", "between",
)


def _wants_comparison(question):
    """Does the question ask how something varies, rather than just what it is?"""
    text = question.lower()
    return any(word in text for word in COMPARISON_WORDS)


def _mentions(question, column):
    """Does the question refer to this column by name? Handles simple plurals."""
    word = column.lower().replace("_", " ").strip()
    if not word:
        return False
    return bool(re.search(rf"\b{re.escape(word)}s?\b", question.lower()))


def pick_columns(question, notes):
    """Which columns does this question need?"""
    catalogue = "\n".join(f"  {col:20s} {note}" for col, note in notes.items())

    reply = llm.chat(f"""A dataset has these columns:

{catalogue}

The user asks: "{question}"

Reply with ONLY the column names needed, comma separated. Nothing else.
Choose as few as possible. Leave out HEAVY columns unless truly needed.""",
                     tier="small", max_tokens=200)

    chosen = [col for col in notes if col.lower() in reply.lower()]
    return chosen or list(notes)[:3]


# ----------------------------------------------------------------------
# MAKE IT FIT
# ----------------------------------------------------------------------

def fit_to_budget(rows, columns, budget=None, protect=ALWAYS_SEND):
    """
    Build the CSV and make sure it fits. Returns (text, columns, tokens, warning).

    Order of retreat: drop the heaviest unprotected column, then send fewer rows.
    """
    budget = budget or llm.budget()
    columns = list(columns)
    text, tokens, warning = "", 0, None

    if not rows or not columns:
        return text, columns, tokens, "No data available to send."

    while len(columns) > 2:
        text = loader.to_csv(rows, columns)
        tokens = llm.count_tokens(text)
        if tokens <= budget:
            return text, columns, tokens, warning

        # Never drop protected columns - see ALWAYS_SEND above.
        droppable = [c for c in columns[1:] if c not in protect]
        if not droppable:
            break

        weights = {c: pct for c, pct, _ in loader.column_weights(rows) if c in columns}
        heaviest = max(droppable, key=lambda c: weights.get(c, 0))
        columns.remove(heaviest)
        warning = f"Dropped '{heaviest}' to fit the budget."

    text = loader.to_csv(rows, columns)
    tokens = llm.count_tokens(text)

    if tokens > budget:
        total = len(rows)                          # capture BEFORE truncating
        keep = max(50, int(total * budget / tokens * 0.9))
        text = loader.to_csv(rows[:keep], columns)
        tokens = llm.count_tokens(text)
        warning = (f"Only {keep:,} of {total:,} rows fit. Group the data with "
                   f"loader.rollup() to cover all of it.")

    return text, columns, tokens, warning


# ----------------------------------------------------------------------
# ASK
# ----------------------------------------------------------------------

def ask(question, rows, budget=None, protect=ALWAYS_SEND, caveats=(), quiet=False,
        group_by=()):
    """
    Answer one question about one table.

    group_by: the columns the table was grouped by. These are ALWAYS sent -
    without them the rows have no identity. Grouping by YEAR+TYPE and then
    sending only TYPE collapses 42 distinct rows onto two labels, and the
    model will happily describe the result as if it meant something.
    """
    budget = budget or llm.budget()
    protect = tuple(dict.fromkeys(tuple(protect) + tuple(group_by)))

    notes = describe_columns(rows)
    columns = pick_columns(question, notes)

    # Protected columns are always sent, whether the picker asked for them or not
    columns = list(dict.fromkeys([c for c in group_by if c in rows[0]]
                                 + columns
                                 + [c for c in protect if c in rows[0]]))

    data, columns, tokens, warning = fit_to_budget(rows, columns, budget, protect)

    if not quiet:
        print(f"  columns : {', '.join(columns)}")
        print(f"  size    : {tokens:,} tokens of {budget:,} budget")
        if warning:
            print(f"  note    : {warning}")

    answer = llm.chat(
        f"Here is a summary table. Every number in it has ALREADY been "
        f"calculated correctly from the full dataset:\n\n{data}\n\n"
        f"Question: {question}\n\n"
        "Rules:\n"
        "- Read the numbers. Do NOT recalculate them.\n"
        "- Never average a column of averages - that ignores group sizes and "
        "gives the wrong answer. n_rows says how many records each row covers.\n"
        "- Do not show your arithmetic. Answer directly.\n"
        "- Keep it under 150 words."
        + (f"\n- {warning} Say so in your answer." if warning else "")
        + ("\n\nIMPORTANT limits of this data - respect these and mention them "
           "where relevant:\n" + "\n".join(f"- {c}" for c in caveats) if caveats else ""),
        tier="large", max_tokens=1500,
    )

    return {"answer": answer, "columns": columns, "tokens": tokens,
            "warning": warning, "provider": llm.provider()}


# ----------------------------------------------------------------------
# FETCH FULL DETAIL FOR A FEW ROWS ONLY
# ----------------------------------------------------------------------

def get_details(names, rows, key_column=None, heavy_columns=None):
    """
    Pull the FULL text of the heavy columns, but only for the named rows.

    This is how the heavy columns stay available without ever sending them all.
    """
    if not rows:
        return ""

    key_column = key_column or list(rows[0])[0]
    heavy_columns = heavy_columns or [
        col for col, pct, _ in loader.column_weights(rows) if pct > 15
    ]

    wanted = {str(n).strip().lower() for n in names}
    blocks = []

    for row in rows:
        if str(row.get(key_column, "")).strip().lower() in wanted:
            detail = "\n".join(f"  {col}: {row.get(col, '')}" for col in heavy_columns)
            blocks.append(f"{row[key_column]}\n{detail}")

    return "\n\n".join(blocks)


# ----------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------

DEFAULT_FILE = "/home/spoo/Downloads/movies 2.csv"

# Above this, the file goes to store.py (Parquet + DuckDB) instead of memory.
BIG_FILE_MB = 50


def main_big(path, question):
    """
    The disk path, for files too large to hold in memory.

    Same pipeline, different engine: DuckDB does the grouping in SQL and only
    the small summary ever reaches Python.
    """
    import store

    file_id = os.path.splitext(os.path.basename(path))[0]
    parquet = store.ingest(path, file_id)["data"]
    info = store.schema(parquet)
    print(f"Indexed : {info['rows']:,} rows, {len(info['columns'])} columns")

    # Classify using a sample - reading 800 rows tells us as much as reading 900,000
    sample = store.sample_rows(parquet, 800)
    dimensions, measures = loader.classify_columns(sample)
    if not dimensions or not measures:
        print("Could not find columns to group by and measure. Try a smaller file.")
        return

    print(f"  can group by : {dimensions[:8]}")
    print(f"  can measure  : {measures[:8]}")

    groups = pick_grouping(question, dimensions)
    print(f"Grouping by: {' + '.join(groups)}")

    summary = store.rollup_rows(parquet, groups, measures[:3], min_rows=20)
    if not summary:
        summary = store.rollup_rows(parquet, groups, measures[:3])
    print(f"  {len(summary):,} rows  ~{loader.rough_tokens(loader.to_csv(summary)):,} tokens")

    caveats = loader.reliability_notes(summary, groups)
    for c in caveats:
        print(f"  caveat: {c[:90]}...")
    print()

    result = ask(question, summary, caveats=caveats, group_by=groups)
    print()
    print(result["answer"])


def main():
    """
        python3 analyst.py "your question"                  uses the default file
        python3 analyst.py path/to/file.xlsx "your question"
    """
    args = sys.argv[1:]

    # If the first argument is a file that exists, treat it as the data file
    if args and os.path.exists(args[0]):
        path, args = args[0], args[1:]
    else:
        path = DEFAULT_FILE

    question = " ".join(args) or "Analyse this data and tell me what stands out."

    print(f"Provider: {llm.info()}")
    print(f"File    : {os.path.basename(path)}")

    # A big file will not fit in memory as Python dicts - 900,000 rows x 74
    # columns is about 6 GB. Route those through store.py, which keeps the
    # data on disk and does the grouping in SQL.
    size_mb = os.path.getsize(path) / 1024 ** 2
    if size_mb > BIG_FILE_MB:
        print(f"Size    : {size_mb:.0f} MB - too big for memory, using the disk path\n")
        return main_big(path, question)

    print()
    tables = loader.read_file(path)
    table_name = next(iter(tables))
    if len(tables) > 1:
        print(f"Sheets found: {', '.join(tables)} - using '{table_name}'")
        print("(pass a different sheet by editing main, or use the API)\n")

    rows = loader.clean(tables[table_name])
    if not rows:
        print("That file has no readable data in it.")
        return
    print(f"Loaded {len(rows):,} rows")

    # Split multi-value cells, tidy messy year columns. Automatic - nothing
    # below is specific to this file, so the CLI and the API behave the same.
    rows, prep_notes = loader.auto_prepare(rows)
    for note in prep_notes:
        print(f"  {note}")

    dimensions, measures = loader.classify_columns(rows)
    groups = pick_grouping(question, dimensions)

    # A time trend must not mix kinds of record - films and series rate
    # differently and a series is filed under the year it started.
    for dim in dimensions:
        if dim.endswith("_TYPE") and dim not in groups and dim[:-5] in groups:
            groups = groups + [dim]
            print(f"  (added {dim} - these cannot share a trend line)")

    print(f"Grouping by: {' + '.join(groups)}")

    summary = loader.rollup(rows, groups, {m: "avg" for m in measures[:3]}, min_rows=20)
    print(f"  {len(summary):,} rows  ~{loader.rough_tokens(loader.to_csv(summary)):,} tokens")

    caveats = loader.reliability_notes(summary, groups)
    for c in caveats:
        print(f"  caveat: {c[:90]}...")
    print()

    result = ask(question, summary, caveats=caveats, group_by=groups)
    print()
    print(result["answer"])


if __name__ == "__main__":
    main()
