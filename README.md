# Swarn — Autonomous ML Engineering Agent

A from-scratch competitor to [HeyNeo](https://heyneo.com) (the autonomous ML
engineer), built on the Phase 1–16 foundation, the V2 upgrades, and a V3 pass
designed to *beat* HeyNeo on the things that decide benchmarks and real work:

**V2 (kept):**

1. **A solution tree search engine** (`agent/search/`) — the AIDE-style
   draft → debug → improve loop that tops the MLE-bench leaderboard, instead of
   a single-pass ReAct agent hand-driving individual tools.
2. **BYO-LLM** (`agent/llm/`) — Anthropic, OpenAI, Ollama, vLLM, Gemini, Groq,
   or any OpenAI-compatible endpoint, selected by one model-spec string.
3. **Docker-optional execution** (`agent/runtime/execution.py`) — Docker when the
   daemon is up, a real cross-platform subprocess backend (Windows-safe,
   per-call timeouts) when it isn't.

**V3 (new — where we pull ahead of HeyNeo):**

4. **Parallel tree search** (`swarn solve … --workers 4`) — HeyNeo expands one
   solution at a time; our scheduler pipelines propose→execute→review across
   a thread pool with reservation-aware policy (no duplicate debugging, no
   draft explosion). Four drafts now cost roughly one draft's wall time.
5. **Self-improvement across runs** (`agent/memory/knowledge.py`) — every finished
   run is reflected on by a cheap LLM call that distills *generalizable*
   lessons into a hard-capped playbook (hermes-agent style bounded memory),
   and archived in a SQLite FTS5 index. Future runs get the playbook plus
   the most similar past runs injected as prior art. HeyNeo starts every task
   cold; Swarn gets smarter with use. Inspect with `swarn playbook`.
6. **Checkpoint / resume** — `swarn solve --resume <run_id>` continues a killed
   run from its crash-safe `journal.json`, adding `--steps` more nodes.
7. **Static gate** (`agent/search/static_check.py`) — syntax errors, missing
   metric prints, and guaranteed hangs are scored buggy *without* burning a
   sandbox execution (opencode's diagnostics-before-run pattern).
8. **Budgets** — `--token-budget` stops a run before costs run away
   (transparent, unlike HeyNeo's opaque credits); time budgets shrink per-node
   timeouts to fit.
9. **Doom-loop detection + context compaction** (`agent/core/doom_loop.py`) — the
   ReAct loop detects same-call-same-result repetition (result-hash aware, so
   polling never false-positives) and injects a corrective note; old tool
   results are compacted once the conversation crosses a size budget.
   Iteration cap is now configurable (`SWARN_MAX_ITERATIONS`, default 30).
10. **MCP server** (`swarn mcp-serve`) — expose the whole platform to Claude
    Code, Cursor, Windsurf, Zed, or any MCP client via
    `swarn_submit_task` / `swarn_task_status` / `swarn_get_messages` /
    `swarn_list_tasks`. Register: `claude mcp add swarn -- swarn mcp-serve`.
11. **Docker timeout fix** — a timed-out container exec is now actually
    killed (container recycled), not silently left eating CPU.
12. **Dashboard upgrades** — `swarn serve` now renders search runs (tree +
    report) and the live playbook alongside the session feed.

The original 16-phase documentation lives in `README-phases-1-16.md`; all of
those capabilities (repo-RAG, self-correction, memory, data/feature/training/
eval/deploy tools, multi-agent orchestration, MCP, multimodal RAG, fine-tuning,
guardrails, CLI, dashboard) still work unchanged.

### V3 quick reference

```bash
# parallel search with budgets, learning enabled (default)
swarn solve "Predict churn. Metric: AUC." -d ./data -s 24 --workers 4 --token-budget 500000

# resume a killed run with 10 more nodes
swarn solve "Predict churn. Metric: AUC." --resume 20260714-091534-ba9b4f -s 10

# what has the agent learned so far?
swarn playbook

# serve the platform to Claude Code / Cursor over MCP
swarn mcp-serve
```

Env vars added in V3: `SWARN_SEARCH_WORKERS`, `SWARN_KNOWLEDGE_DIR`,
`SWARN_MAX_ITERATIONS`, `SWARN_CONTEXT_CHAR_BUDGET`.

---

## Why tree search is the headline feature

HeyNeo's benchmark claim is #1 on MLE-bench (34.2%). What wins that benchmark is
not a smarter chat loop — it's *systematic experimentation*: generate several
complete solution scripts, run them, measure them, fix the ones that crash,
and iteratively improve whichever is winning, until the budget runs out. That
is exactly what `agent/search/` implements:

```
            ┌─ draft ── run ── review ──┐        journal (solution tree)
task ──────►│  draft ── run ── review   ├──────► pick action for next step:
            └─ draft ── run ── review ──┘          • < num_drafts?  → draft
                     ▲                             • buggy leaf?    → debug (p=0.5)
                     └──── debug / improve ◄─────  • else           → improve best
```

- **Journal** (`journal.py`) — every attempt is a node in a tree with its code,
  execution output, reviewed metric, and buggy/good verdict. Crash-safe: saved
  to `runs/<id>/journal.json` after every step.
- **Policy** (`agent.py`) — drafts until `num_drafts` roots exist, then debugs
  buggy leaves with probability `debug_prob` (up to `max_debug_depth`
  consecutive fixes per branch) or makes one atomic improvement to a top-k
  good node (epsilon-greedy).
- **Review** (`agent.py`) — a second, low-temperature LLM call is *forced*
  (via tool call) to return `{is_bug, summary, metric, lower_is_better}`.
  The script's own printed `Final Validation Metric:` line wins on
  disagreement — regex + reviewer, not vibes.
- **Data preview** (`data_preview.py`) — file tree, CSV dtypes and heads
  injected into every prompt so generated code targets real columns.
- **Report** (`report.py`) — `runs/<id>/report.md` with the ASCII solution
  tree, metric history table, and failure analysis.

### Run it

```bash
swarn solve "Predict survival. Metric: accuracy." --data ./titanic --steps 20
swarn solve "Forecast demand, minimize RMSE." -d ./data -s 30 -t 3600 -m openai:gpt-4o
```

Or let the ReAct agent decide — it has a `solve_ml_task` tool and its
system prompt tells it to prefer search for "build the best model" tasks:

```bash
swarn run "Train the best possible model on workspace/churn.csv predicting churn"
```

Programmatic:

```python
from agent.search import SearchConfig, run_search
result = run_search("Predict y. Metric: AUC.", data_dir="data/",
                    config=SearchConfig(steps=25, time_limit_secs=3600))
print(result.best.metric, result.solution_path)
```

---

## Deployed model endpoint (BYO-LLM removed)

Every LLM call — single agent, team pipeline, search engine, dashboard, MCP
server — routes to **one deployed, OpenAI-compatible endpoint**, configured
in `agent/llm/router.py` (see the "PRODUCTION ENDPOINT — CHANGE HERE"
banner). For testing it points at a Qwen 3.5 9B deployment on Modal.

Swap in the production deployed model without code changes:

```bash
SWARN_DEPLOYED_MODEL=<served-model-name>
SWARN_DEPLOYED_BASE_URL=<https://.../v1>
SWARN_DEPLOYED_API_KEY=<key, or "dummy" if unsecured>
```

`--model` flags and `SWARN_CODE_MODEL`/`SWARN_FEEDBACK_MODEL` are accepted
but ignored for routing (display/log only) — except `mock:*`, the scripted
offline client that powers the test suite. All calls get retries with
exponential backoff and token accounting.

## Execution backends

```
SWARN_SANDBOX=docker|subprocess   # force; default = auto-detect
SWARN_EXEC_TIMEOUT=300            # default per-call timeout (seconds)
SWARN_SANDBOX_IMAGE=python:3.11-slim
```

Docker gives isolation (memory/CPU caps, bind-mounted workspace). The
subprocess backend gives universality: `sys.executable` (works on Windows),
hard timeouts, head+tail output truncation, structured `ExecResult` with exit
code and timing — which the search engine uses to detect timeouts and score
runs.

## PDF → structured data

Two different things you can do with a PDF, on purpose:

```bash
swarn index <dir>                    # PDF joins the semantic search index (embeds; needs the model)
swarn extract-pdf <file.pdf>         # whole PDF -> structured JSON (no embedding, works offline)
swarn extract-pdf <f.pdf> --md               # same structure, rendered as readable Markdown
swarn extract-pdf <f.pdf> --page 3           # one page only
swarn extract-pdf <f.pdf> -o out.json        # write to a file
swarn extract-pdf <f.pdf> --csv-dir ./tables # one CSV per detected table
swarn extract-pdf <f.pdf> --mode pages       # flatter shape: per-page text + tables
```

`extract-pdf` structures the **entire** document, not just its tables:

```
{ "title", "metadata", "n_pages",
  "fields":   { "Invoice date": "2024-09-17", ... },   # every label->value, document-wide
  "sections": [ { "heading", "level", "page", "blocks": [ ... ] } ],
  "counts":   { "sections", "paragraphs", "lists", "tables", "fields" } }
```

Each section holds typed blocks in reading order — `paragraph` (wrapped lines
rejoined, hyphenation repaired), `list` (items split out, markers stripped,
`ordered` flagged), `key_values`, and `table` (a `rows` grid **plus** `records`
of `{column: value}` when the header row is unambiguous). `fields` also mines
pairs out of prose, so `"Invoice date: 2024-09-17. Terms: Net 30."` yields both.

Structure is inferred from **layout** — font size, boldness, position — with no
LLM call and no network.

**One parse, shared.** `extract-pdf` no longer opens the PDF itself. It reads
the same stored document `swarn ingest` writes and `swarn ask` answers from
(`swarn/capabilities/doc_structure.py` converts it into the element list the
tree builder consumes). Two independent pdfplumber pipelines over one file did
not merely cost a second parse — they disagreed, and the tree was the weaker of
the two: it had no column handling, so a two-column notice block merged into
`"How to update Mobile/WhatsApp/Email How to verify Mobile/Email"`, and its
`Label | : value` rows yielded no fields at all. Deriving the tree from stored
data fixes both and gives every block a bounding box. Headings are lines larger than the char-weighted body
size (or short bold lines); distinct heading sizes rank into `level`. Table
regions are cut from the text layer before lines are built, so table contents
never appear twice.

Agent-facing tools: `index_pdf` (search), `extract_pdf_document` (full tree),
`extract_pdf_structured` (per-page text + tables).

Bilingual documents are handled where they used to be dropped: Indian utility
and tax forms label every field twice (`देय ितिथ / Due Date`), and a label that
fails validation as written is retried on its Latin half — only then, so a
genuine `Net / Gross` label is never rewritten. Cells arriving as
`Bill No | : 382-209-037-556 | Bill Month | : JUL-2026` are split into their
segments first, with a colon-prefixed cell read as the previous cell's value,
so a row carrying two independent fields yields both. On a UPPCL electricity
bill this took `fields` from **0 to 18**.

Caveats, both by design: tables are found by **visual** structure — ruled lines
or consistent column gaps — so data laid out as unruled positioned text is not
detected as a table and lands in the surrounding prose instead. And because
structuring is heuristic, a document with no font-size variation (or one that
sets body text bold) under-structures into fewer sections rather than inventing
wrong ones. A scanned/image-only PDF has no text layer at all — OCR it with
`index_image` on a rasterized page.

A third caveat belongs to the document rather than to this code: a PDF whose
text layer carries no space glyphs emits whole clauses as one word object
(`'है/Debitrepresentstheadditionalamountchargedtothe'` is a single word in the
UPPCL bill's own text layer). Word boundaries a file does not encode cannot be
recovered from it; that text needs OCR of a rasterized page.

## PDF tables → CSV

`extract-pdf --csv-dir` writes a CSV per table, but only for tables pdfplumber
can *see* — ruled ones. Plenty of PDFs that exist to carry a table have no
ruling at all, and on a 54-page credit-card dataset (1,300 records) that path
produced **zero files**. `swarn to-csv` handles both:

```bash
swarn to-csv CreditCard.pdf              # -> CreditCard/table_1_p1.csv, ... + tables.json
swarn to-csv data.pdf --dir ./out        # -> ./out/data/...
swarn to-csv data.pdf -o dataset.csv     # one flat file instead of a folder
swarn to-csv data.pdf --page 3 --page 4
swarn to-csv data.pdf --split-fused      # split cells like '124.9833yes'
```

**Each PDF gets its own folder**, named after it, holding one CSV per table plus
a `tables.json` manifest recording which pages each file came from, its shape,
its header, and how it was found. Six tables should not scatter six loose CSVs
into a directory shared with every other document converted there.

**This happens automatically at ingest.** A document is parsed once, and its
tables are already in hand at that moment, so `swarn ingest` (and the first
`swarn ask`, and the agent's `swarn_doc_ingest`) writes them to
`artifacts/documents/tables/<name>/` and records the folder on the stored
document as `tables_dir`. A caller who uploads a PDF holding tables wants files,
not a nested array inside a JSON blob they have to write code against. `swarn
to-csv` remains the explicit form, for converting somewhere specific or with
`--split-fused`.

Table boundaries are the document's to draw, not ours. Consecutive **ruled**
tables merge only when the second repeats the first's header — a participant
handbook held 18 ruled tables of which most were three columns wide (`Date /
Milestone / What Happens`, `Award / Criteria / Recognition`), and grouping on
width alone welded five unrelated ones into a single CSV. A **text**-strategy
grid asserts no boundary at all, so there width is the only signal, and the
right one: a dataset spanning 26 pages carries no header after the first.

Two sources, tried in order: **ruled** tables come from the stored parse
(exact — the grid is the one the document draws); otherwise columns are
inferred from **text alignment**, which is what recovers a borderless dump.
Consecutive pages of equal width are one table continued and land in one file;
a change of shape starts a new one — the credit-card PDF is an R console
session whose pages 2-27 are a 9-column data frame and whose pages 28-54 are a
3-column one, and it converts into exactly those pieces.

The text strategy always returns *something*, so most of the work is refusing
it when what it returned is not a table:

- **Fill tests** — a table's rows are mostly full and its columns appear in
  most of its rows. An invoice's address block fails both; without this it
  became a grid of fragments like `MAKEMYTRIP (I` | `NDIA) PRIVATE`.
- **Word integrity** — the definitive signature of a false grid is a column
  boundary falling inside a word (`asFinalNoticeunderSectio` | `n9.3` — that is
  "Section 9.3", cut in half). Nothing about that grid's *shape* is wrong, so
  the fill tests pass it. The store settles it: it grouped the page's words
  rather than re-cutting a rendered string, so a token that is not one of those
  words is text the grid invented. Measured, not forbidden — a genuine data
  page scores 0-4%, a prose page 28%.

When nothing survives, nothing is written and it says so. A garbled CSV that
looks like data is worse than an honest failure, because the CSV gets loaded
and the failure does not get noticed. Exit codes: `0` wrote something, `2`
found no table, `1` error.

Agent-facing tool: `swarn_pdf_to_csv` — convert first, then `load_csv` the
result to get a PDF's table into pandas.

## Visual document intelligence (bounding boxes)

`extract-pdf` above answers *what* a document says. `doc-inspect` answers
**where on the page each value came from** — and draws it:

```bash
python examples/demo_doc_inspector.py    # no args, no key: generates a mock
                                          # invoice and annotates it
swarn doc-inspect invoice.pdf             # fields + boxes + annotated PNG
swarn doc-inspect scan.png               # a scan: auto-routes to local OCR
swarn doc-inspect f.pdf --all-pages -q    # every page, summary only
swarn ask "<question>" f.pdf              # answer a question, with evidence
```

Every extracted field carries a normalized `BoundingBox` (0.0–1.0, so it
survives a re-raster at a different DPI), and the annotated image draws a
rounded box per field, colour-coded by confidence — **green** >0.85, **amber**
0.60–0.85, **red** <0.60 — with a label tag naming the field and value. That is
the point of the capability: on invoice/KYC/financial work the failure mode is
a confident wrong number, and a coordinate makes that falsifiable in a glance.

```
{ "document_name", "page_number", "backend",
  "fields": [ { "field_name", "field_value", "confidence",
                 "box": {"xmin","ymin","xmax","ymax"} } ],
  "annotated_image_path", "n_fields", "n_low_confidence" }
```

Four backends, selected by `--backend` or `$SWARN_DOC_BACKEND`:

| backend | needs | text | coordinates |
| --- | --- | --- | --- |
| `text` | pdfplumber | exact, from the PDF | exact, from PDF word geometry |
| `ocr` | tesseract binary | transcribed | real, from OCR word boxes |
| `vlm` | `SWARN_VLM_API_KEY` (+`_BASE_URL`, `_MODEL`) | from the model | from the model |
| `mock` | nothing | **synthetic sample data** | hard-coded |

**`auto` never selects `mock`.** For a document you supply it tries `vlm` (if
credentials are set), then the PDF's embedded text layer, then local OCR — and
if none of those can run it fails with an actionable error naming what to
install. Substituting sample data for a real document would report another
document's vendor and total as if they had been read off yours, which is the
precise failure this capability exists to catch. `mock` is reachable only by
naming it (`--backend mock`), which is what the demo does for the invoice it
generates itself; results then carry `synthetic: true` and a warning.

`text` is preferred over `ocr` whenever a PDF has a text layer: the characters
are already exact in the file, so rasterizing and transcribing them back can
only introduce errors that were never there.

The `vlm` backend speaks the OpenAI-compatible `image_url` wire format, so
OpenAI, Gemini's compat layer, and any vLLM/SGLang server fronting Qwen2.5-VL
all work without a per-provider adapter; boxes come back in the standard
normalized `[ymin, xmin, ymax, xmax]` order and are coerced by `BoundingBox`
(which also handles the 0–1000 integer convention and reversed corners).
`text` and `ocr` share one entity engine over positioned words: typed patterns
(GSTIN, IFSC, PAN, email, phone, invoice/PO number, dates, amounts), inline
`Label: value` pairs, and captions stacked above their value — with lines split
into column segments first, so a two-column header does not name a field after
its neighbour. Every value is a substring of text physically on the page and
every box is the union of the boxes of the words it came from; a field that
isn't there is simply not emitted (`raw_json.not_found` lists the common
business fields that were looked for and missing). Confidence multiplies the
transcription certainty (1.0 for a text layer, tesseract's own score for OCR)
by how much the *rule* proves — a stated `Label: value` outranks a layout
inference.

Passing a pydantic `target_schema` narrows the extraction to that model's
fields and validates the result — a mismatch is reported in
`raw_json["schema_error"]` rather than raised, because "the document did not
contain what you expected" is a finding, and the boxes are still worth seeing.

Agent-facing tool: `swarn_doc_inspect`. Annotated images land in `artifacts/`
(`$SWARN_ARTIFACTS_DIR` to relocate).

## Ask a question about a document

`doc-inspect` extracts whatever fields a page holds. `ask` answers a specific
question — including one whose answer appears **nowhere in the document** and
has to be derived from figures that do:

```bash
swarn ingest report.pdf                                     # parse once (optional)
swarn ask "what was the percentage increase in revenue" report.pdf
swarn ask "who signed this and on what date" contract.pdf --page 4
swarn ask "what is the total GST charged" invoice.pdf --json

# ...or through the universal entry point — same code path, same output:
swarn run "what is the total GST charged" invoice.pdf
swarn run "what is the total in invoice.pdf"                # path read from the task
```

`swarn run` routes a bare question about a single document straight here
(`agent/task_router.py`), and sends anything needing more — two documents, a
plot, a model — to the ReAct agent, which has `swarn_doc_ask` in its toolset.
The split is about correctness, not just latency: the three defences below run
inside `doc_qa`, so letting the agent *paraphrase* their JSON would reintroduce
an unverified claim at the one step after every check has already passed. The
fast path skips the paraphrase; the agent path prints the tool's verified
evidence before the agent's summary, and the system prompt forbids restating a
tool's figures loosely. Force either with `--ask` / `--agent`.

**A document is parsed once.** `swarn ingest` writes a structured JSON copy to
`artifacts/documents/<document_id>.json` — word text, bounding boxes, per-word
confidence, line ids, page dimensions, tables — and every later `ask` loads
that instead of re-reading the PDF. `ask` ingests automatically on first use,
so the command above works with or without the explicit step; `swarn ingest
--list` shows what is stored. `document_id` is `<stem>-<sha256[:12]>`, so an
edited file misses the cache and is re-parsed rather than answered from a
superseded revision. Measured on a 6-page deck: ~2.0 s to ingest, ~10 ms per
question afterwards (the gap is far larger for scans, where the alternative is
a full OCR pass every time).

The only thing still read from the source is page PIXELS, when an evidence
image is requested — an image cannot be reconstructed from word geometry.
`swarn ingest --render-pages` caches page rasters too, making the store fully
self-contained; `--no-annotate` skips that path entirely.

```
Revenue increased by 23.4% from FY23 to FY24.

  computed  (148200 - 120100) / 120100 = 0.23397 = 23.4%   [verified]

  evidence
    p3  FY23 revenue  120100    box(0.485, 0.168, 0.541, 0.180)
    p3  FY24 revenue  148200    box(0.636, 0.168, 0.692, 0.180)

  -> artifacts/report_p3_evidence.png

  [stored: report-8a5bc40cc6c2 · text · 3/3 pages searched]
```

The answer is derived by an LLM, so it is reported with everything needed to
check it rather than on its own authority. Three defences, enforced in code and
not merely requested in the prompt:

1. **The model never sees the document** — only a transcript of words this repo
   extracted, one stable id per line (`[p3:L3] Revenue | 120100 | 148200`).
   The transcript is column-aware. `|` joins cells of one row; independent page
   columns are kept as separate lines, and a multi-column page is emitted in
   **reading order** (each column in full, left to right). Telling a row from a
   page gutter needs two signals, because neither suffices alone: segments in
   the **same detected table** are a row however far apart they sit (a contacts
   table's columns measured 43% of page width), and where no table is detected —
   `find_tables()` sees only *ruled* tables — segments closer than 40% of page
   width are a row, which is what keeps borderless tables like `Date | Activity`
   intact. Both matter: on a two-column slide the
   old unconditional join produced `TEAM MEMBERS | Jaimin Nalin Desai`, pairing
   a left-column caption with a right-column value that actually belonged to
   `MENTOR NAME` above it — and a model asked to list team members answered
   with the mentor, correctly, given what it was shown.
2. **Every quote is verified against the document.** Three strategies are
   tried in order — the cited **line**, then the lines after it
   (**multiline**, for wrapped prose and wrapped cells), then **table** cells.
   The table strategy exists because a model citing a table row writes what it
   reads across the row (`<trigger> CC 5` — a cell, a column *header*, then a
   cell in another column), and that string is on no single extracted line; a
   line-only resolver called those correct citations `NOT FOUND`. Matching is
   token-exact throughout: case, whitespace, line breaks and punctuation are
   normalized away, but every word must still be present, in order, in the
   cells claimed — and the cells must come from one row, so a real trigger
   paired with another row's number is still rejected. A verified span reports
   which `strategy` found it; a table span also carries the table, row, columns
   and per-cell boxes. Unverified quotes are boxed in red and marked
   `<< NOT FOUND IN DOCUMENT`, and an answer with nothing verified behind it is
   reported as *unsupported*, never as an answer. Boxes are the union of the
   matched cells for table evidence, and the shortest matching run of words
   otherwise.
3. **The arithmetic is re-evaluated locally**, by an AST walker accepting
   numeric literals and operators and nothing else (the expression is LLM
   output, so `eval` would be an execution sink reachable from any document
   plus any question). `MISMATCH` prints when the sums do not check out.

Exit codes: `0` answered, `2` not answerable from this document, `1` error.
Page selection is automatic — a long document is narrowed to the pages whose
text matches the question rather than truncated mid-table. Reading uses the
same `auto` backend rules as `doc-inspect` (`text` -> `ocr`, never `mock`); the
answering model is the deployed endpoint from `agent/llm/router.py`.

Agent-facing tools: `swarn_doc_ask`, `swarn_doc_ingest`.

## Dashboard

`swarn serve` — everything from Phase 16, plus:

- `GET /api/runs` — all search runs with node counts and best metrics
- `GET /api/runs/{id}` — full journal (the tree) + report markdown
- `GET /api/playbook` — the cross-run playbook (learned lessons)

## Tests

```bash
python tests/run_tests.py    # zero-dependency runner (283 tests)
pytest tests/                # same tests, if pytest is installed
```

Coverage: deployed-endpoint routing, Anthropic↔OpenAI message/tool conversion, retry
policy, subprocess execution + timeouts, journal tree ops and persistence,
search policy transitions (incl. parallel reservations), static gate,
knowledge store + reflection, doom-loop detection, context compaction, review
parsing, PDF structuring, document intelligence + Q&A + store, table/column
evidence resolution, `swarn run` task routing, the stored-data document tree, PDF->CSV — plus full offline end-to-end searches (sequential and parallel,
resume, token budgets) on synthetic data with mock LLMs and real execution.

## Setup

```bash
cd swarn
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                # enables the `swarn` command
cp .env.example .env                            # optional: SWARN_DEPLOYED_* overrides (no API key needed for the test Qwen endpoint)
swarn solve --help
```

## What changed vs. phases 1–16 (file map)

```
agent/
  llm/                    NEW  LLM layer (hard-routed to the deployed endpoint)
    base.py                    normalized blocks/response, retries, usage tracking
    anthropic_client.py        REMOVED (tombstone stub — BYO-LLM stripped)
    openai_client.py           OpenAI-compat client for the deployed endpoint
    router.py                  deployed endpoint config ("CHANGE HERE" for production)
    mock_client.py             scripted client for offline tests
  search/                 NEW  the MLE-bench engine
    config.py journal.py agent.py runner.py report.py data_preview.py
    static_check.py            V3: pre-execution AST gate
  knowledge.py            NEW  V3: playbook + FTS5 run archive + reflection
  doom_loop.py            NEW  V3: repetition guard for the ReAct loop
  task_router.py          NEW  `swarn run` routing: document fast path vs agent
  mcp_server.py           NEW  V3: MCP server (swarn mcp-serve)
  execution.py            NEW  Docker + subprocess backends, ExecResult
  llm_client.py           now a shim over agent/llm (back-compat)
  sandbox.py              now a shim over agent/execution (back-compat)
  tools.py                + solve_ml_task tool
  prompts.py              + tells the agent when to prefer tree search
  cli.py                  + `swarn solve` (--workers/--resume/--token-budget),
                            mcp-serve, playbook, extract-pdf, doc-inspect,
                            ingest, ask — and `swarn run` as the universal
                            entry point (routes to ask, or to the agent)
  dashboard.py            + /api/runs + /api/playbook endpoints
swarn/                  NEW  capability packages (standalone, agent-registered)
  capabilities/
    doc_intelligence.py        visual document intelligence: PDF/image ->
                               fields + bounding boxes -> annotated image
    doc_structure.py           stored document -> the element list the
                               document-tree builder consumes (one parse)
    doc_csv.py                 PDF tables -> CSV files in a per-PDF folder,
                               ruled or borderless, with prose refused
    doc_qa.py                  grounded Q&A: question -> answer + verified
                               evidence boxes + re-checked arithmetic
    doc_store.py               parse once -> structured JSON on disk, reused
                               by every later question
examples/               NEW  runnable demos
  demo_doc_inspector.py        zero-config bounding-box demo
main.py                   provider-aware API-key check
tests/                    NEW  283 tests + zero-dependency runner
```
