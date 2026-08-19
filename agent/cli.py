"""
Phase 16: CLI (Typer)

A proper standalone command-line front end, alongside (not replacing)
main.py's interactive REPL. The REPL is great for an ongoing back-and-
forth session; this CLI is for the other common case — a single
one-off command run from a shell script, a CI job, or just muscle
memory ("run X and exit", not "open a prompt and type X").

Why this is a separate entry point from main.py
─────────────────────────────────────────────────────
main.py's REPL loop assumes an interactive terminal: it reads from
stdin in a `while True`, prints a banner, and only exits on 'exit' or
EOF. None of that fits a one-shot `swarn run "task"` invocation from a
script, where you want: run, print the result, exit with a meaningful
status code. Typer (built on Click) is the natural fit for that shape —
each subcommand below is a normal Python function with type-annotated
parameters, and Typer derives the CLI's argument parsing, help text,
and `--option` flags directly from the signature.

Commands
──────────
  swarn run "<task>" [docs...]    — universal entry point: single agent,
                                     or the document fast path when the
                                     task is just a question about a file
  swarn team "<task>"             — Phase 11 multi-agent pipeline, one-shot
  swarn sessions [--limit N]       — Phase 5 session history
  swarn recall <session_id>         — full tool-call log of one past session
  swarn index <path>                 — Phase 3 repo indexing
  swarn extract-pdf <path>            — PDF → structured JSON (no indexing)
  swarn to-csv <path>                  — PDF's tables → CSV file(s) on disk
  swarn doc-inspect <path>             — PDF/image → fields + bounding boxes
                                          + an annotated image
  swarn ingest <path>                   — parse a document once into stored JSON
  swarn ask "<question>" <path>         — answer a question about a document,
                                          with the evidence boxed and cited
                                          (the explicit form of what
                                           `swarn run` routes to)
  swarn serve [--port N]              — Phase 16's dashboard (see dashboard.py)
  swarn guardrail-benchmark            — Phase 15's canned guardrail test suite

Exit codes
────────────
`run`/`team` exit 0 on outcome="complete", 1 otherwise — so
`swarn run "..." && echo "ok"` in a shell script behaves the way you'd
expect a build/test step to behave.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer

# Deployed model name — all calls hard-route to the endpoint configured in
# agent/llm/router.py, so --model flags below are display/log only.
from agent.llm import DEFAULT_MODEL

app = typer.Typer(
    name="swarn",
    help="Swarn — your autonomous AI engineering agent (CLI front end).",
    add_completion=False,
)


# ═══════════════════════════════════════════════════════════════════════════
# SHARED — one implementation of "answer a question about a document"
# ═══════════════════════════════════════════════════════════════════════════
# `swarn ask` is this and nothing else; `swarn run` reaches it through the
# fast path in task_router. Both go through this function rather than each
# formatting a DocumentAnswer their own way, so the guarantee `ask` makes
# about unverified quotes and bad arithmetic is the same guarantee `run`
# makes — it is enforced in one place instead of promised in two.


def _answer_document(
    path: str,
    question: str,
    page: int = None,
    backend: str = None,
    annotate: bool = True,
    show_json: bool = False,
) -> int:
    """Answer `question` about `path`, print the evidence, return an exit code."""
    from swarn.capabilities.doc_intelligence import DocumentIntelligenceError
    from swarn.capabilities.doc_qa import ask_document

    def _announce_ingest(file_path):
        typer.echo(f"[swarn] {Path(file_path).name} has not been ingested — parsing it "
                   "once now (later questions will reuse the stored copy)...")

    try:
        result = ask_document(
            path, question,
            pages=[page] if page else None,
            backend=backend, annotate=annotate,
            on_ingest=_announce_ingest)
    except DocumentIntelligenceError as exc:
        typer.echo(f"Error: {exc}", err=True)
        return 1

    typer.echo("")
    typer.echo(result.summary())
    typer.echo("")

    if result.unverified:
        typer.echo(
            f"[swarn] WARNING: {len(result.unverified)} cited quote(s) could not be located "
            "anywhere in this document (lines, wrapped text, or table cells were all "
            "searched). Treat those as unsupported.", err=True)
    if result.computation_check.startswith("MISMATCH"):
        typer.echo(f"[swarn] WARNING: the model's arithmetic does not check out — "
                   f"{result.computation_check}", err=True)

    if show_json:
        typer.echo(result.to_json())

    # A question the document cannot answer is a legitimate outcome, but it is
    # not a success — exit non-zero so a script can branch on it.
    return 0 if result.found else 2


def _print_grounded_tool_result(tool_name: str, tool_input: dict, raw_result: str) -> None:
    """
    Show what a document tool actually found, before the agent speaks.

    The agent's closing summary is prose it wrote; these lines are what the
    repo verified. Printing both, in that order, is what keeps `swarn run`
    honest on the agent path: a reader can see the agent's answer resting on
    the evidence rather than replacing it.
    """
    if tool_name not in ("swarn_doc_ask", "swarn_doc_inspect"):
        return
    if raw_result.startswith("Error"):
        return
    try:
        data = json.loads(raw_result)
    except (ValueError, TypeError):
        return

    if tool_name == "swarn_doc_ask":
        typer.echo("\n  ── verified evidence ──")
        if data.get("computation"):
            check = data.get("computation_check") or ""
            typer.echo(f"    computed  {data['computation']}  [{check}]")
        spans = data.get("evidence") or []
        if not spans:
            typer.echo("    (none — this answer has nothing grounded behind it)")
        for span in spans:
            box  = span.get("box") or {}
            mark = "" if span.get("verified") else "   << NOT FOUND IN DOCUMENT"
            typer.echo(
                f"    p{span.get('page_number')}  {span.get('label', '')}  "
                f"{span.get('quote', '')}  "
                f"box({box.get('xmin', 0):.3f}, {box.get('ymin', 0):.3f}, "
                f"{box.get('xmax', 0):.3f}, {box.get('ymax', 0):.3f}){mark}")
        for image in data.get("annotated_image_paths") or []:
            typer.echo(f"    -> {image}")
    else:
        n_low = data.get("n_low_confidence", 0)
        typer.echo(f"\n  ── extracted {data.get('n_fields', 0)} field(s), "
                   f"{n_low} below the confidence floor ──")
        for field_data in (data.get("fields") or [])[:12]:
            box = field_data.get("box") or {}
            typer.echo(
                f"    {field_data.get('field_name', '')}: {field_data.get('field_value', '')}  "
                f"(conf {field_data.get('confidence', 0):.2f})  "
                f"box({box.get('xmin', 0):.3f}, {box.get('ymin', 0):.3f}, "
                f"{box.get('xmax', 0):.3f}, {box.get('ymax', 0):.3f})")
        if data.get("annotated_image_path"):
            typer.echo(f"    -> {data['annotated_image_path']}")


@app.command()
def run(
    task: str = typer.Argument(..., help="The task, or a question to answer about a document."),
    paths: Optional[List[str]] = typer.Argument(None, help="Optional documents (PDF/image) the task is about."),
    model: str = typer.Option(DEFAULT_MODEL, help="Ignored — all calls route to the deployed endpoint (see agent/llm/router.py)."),
    force_ask: bool = typer.Option(False, "--ask", help="Force the document fast path, skipping the agent."),
    force_agent: bool = typer.Option(False, "--agent", help="Force the ReAct agent, even for a plain document question."),
    page: int = typer.Option(None, "--page", help="Document fast path: restrict to a single 1-based page."),
    backend: str = typer.Option(None, "--backend", help="Document fast path: force 'text' or 'ocr'. Default: auto."),
    no_annotate: bool = typer.Option(False, "--no-annotate", help="Skip rendering evidence/annotated images."),
    show_json: bool = typer.Option(False, "--json", help="Document fast path: also print the full result as JSON."),
):
    """
    Run any one-off task — the universal entry point.

    Ordinary engineering tasks go through the single agent and its full
    Phase 1–15 toolset. A task that is really just a question about a
    document is routed straight to the same machinery `swarn ask` uses:

        swarn run "what is the total GST charged" invoice.pdf
        swarn run "what was the percentage increase in revenue" report.pdf

    The fast path exists for a reason beyond speed. `swarn ask` verifies
    every quote against the document and re-evaluates the arithmetic
    locally; if the agent were made to relay that through a paraphrase of
    its own, an unverified claim could re-enter after those checks had
    already run. So a bare document question skips the paraphrase entirely
    and prints the verified answer as-is.

    Anything that needs more than reading — training a model, writing a
    file, combining two documents — goes to the agent, which still has
    swarn_doc_ask in its toolset. When it calls one of the document tools,
    the tool's own verified evidence is printed before the agent's summary.

    Override the routing with --ask or --agent; `swarn run --help` and
    `swarn ask --help` document the same document options.
    """
    from agent.task_router import route

    if force_ask and force_agent:
        typer.echo("Error: --ask and --agent are mutually exclusive.", err=True)
        raise typer.Exit(code=1)

    decision = route(
        task, paths,
        force="ask" if force_ask else "agent" if force_agent else None,
    )

    # ── fast path: a question about one document ────────────────────────
    if decision.is_fast_path:
        if not decision.documents:
            typer.echo("Error: --ask needs a readable document (PDF or image) — "
                       "none was named, or the path does not exist.", err=True)
            raise typer.Exit(code=1)
        raise typer.Exit(code=_answer_document(
            decision.documents[0], task,
            page=page, backend=backend,
            annotate=not no_annotate, show_json=show_json))

    # ── agent path ──────────────────────────────────────────────────────
    # Imported lazily inside each command, not at module level — `swarn
    # --help` shouldn't need to construct an LLMClient (which reads the
    # API key from the environment) just to print usage text.
    from agent.agent_loop import AgentLoop
    from agent.self_correction import SelfCorrectionPolicy
    from agent.observability import GuardrailPolicy

    agent_task = task
    if decision.documents:
        # Name the files as absolute paths in the task itself. The agent's
        # document tools resolve relative paths against WORKSPACE_DIR, not
        # the shell's cwd, so a bare "invoice.pdf" typed at the prompt would
        # otherwise be looked up in the wrong directory.
        listed = "\n".join(f"  - {d}" for d in decision.documents)
        agent_task = (f"{task}\n\nDocuments provided for this task:\n{listed}\n"
                      "Read them with swarn_doc_ask (to answer a question) or "
                      "swarn_doc_inspect (to extract fields with their locations).")
        typer.echo(f"[swarn] routing to the agent — {decision.reason}")

    agent = AgentLoop(
        model=model,
        correction_policy=SelfCorrectionPolicy(),
        guardrail_policy=GuardrailPolicy(),
        on_tool_result=_print_grounded_tool_result,
    )
    result = agent.run(agent_task)
    typer.echo(f"\nOutcome: {result['outcome']}  (session {result['session_id'][:8]})")
    raise typer.Exit(code=0 if result["outcome"] == "complete" else 1)


@app.command()
def team(
    task: str = typer.Argument(..., help="The task for the multi-agent pipeline."),
    model: str = typer.Option(DEFAULT_MODEL, help="Ignored — all calls route to the deployed endpoint (see agent/llm/router.py)."),
    no_tester: bool = typer.Option(False, "--no-tester", help="Stop after Reviewer approval, skip the Tester stage."),
):
    """Run a one-off task through the Phase 11 Planner→Coder→Reviewer→Tester pipeline and exit."""
    from agent.orchestrator import Orchestrator
    from agent.observability import GuardrailPolicy

    orchestrator = Orchestrator(
        model=model,
        include_tester=not no_tester,
        guardrail_policy=GuardrailPolicy(),
    )
    result = orchestrator.run(task)
    typer.echo("\n" + result["report_markdown"])
    raise typer.Exit(code=0 if result["final_outcome"] == "complete" else 1)


@app.command()
def solve(
    task: str = typer.Argument(..., help="Full ML task description (target, metric, constraints)."),
    data: str = typer.Option(None, "--data", "-d", help="Path to the directory holding the task's data files (not needed with --resume)."),
    steps: int = typer.Option(20, "--steps", "-s", help="Search budget: number of solution nodes to try."),
    time_limit: int = typer.Option(None, "--time-limit", "-t", help="Wall-clock budget in seconds."),
    drafts: int = typer.Option(4, "--drafts", help="Number of independent initial solutions."),
    model: str = typer.Option(None, "--model", "-m", help="Ignored — code generation uses the deployed endpoint (see agent/llm/router.py)."),
    feedback_model: str = typer.Option(None, help="Ignored — result review uses the deployed endpoint (see agent/llm/router.py)."),
    exec_timeout: int = typer.Option(600, help="Per-node execution timeout in seconds."),
    workers: int = typer.Option(None, "--workers", "-w", help="Parallel workers: how many solution nodes run concurrently (default 1, or SWARN_SEARCH_WORKERS)."),
    token_budget: int = typer.Option(None, "--token-budget", help="Stop the run after this many total LLM tokens."),
    resume: str = typer.Option(None, "--resume", help="Resume a previous run by its run id; --steps adds that many MORE nodes."),
    no_learn: bool = typer.Option(False, "--no-learn", help="Disable cross-run knowledge: no playbook injection, no post-run reflection."),
):
    """
    V2's flagship command: solve an ML task end-to-end via AIDE-style
    solution tree search (draft -> debug -> improve until the budget is
    spent). Produces runs/<id>/best_solution.py + report.md.

    V3: add --workers N for parallel exploration, --resume <run_id> to
    continue a killed run, --token-budget for cost control. Runs learn
    from each other via the playbook unless --no-learn is given.
    """
    from pathlib import Path as _P
    from agent.search import SearchConfig, run_search

    if not resume and (not data or not _P(data).is_dir()):
        typer.echo(f"error: data directory not found: {data}", err=True)
        raise typer.Exit(code=2)

    kwargs: dict = {"steps": steps, "time_limit_secs": time_limit,
                    "num_drafts": drafts, "exec_timeout": exec_timeout,
                    "use_knowledge": not no_learn, "reflect": not no_learn}
    if workers:
        kwargs["parallel_workers"] = workers
    if token_budget:
        kwargs["token_budget"] = token_budget
    if model:
        kwargs["code_model"] = model
        kwargs["feedback_model"] = feedback_model or model
    elif feedback_model:
        kwargs["feedback_model"] = feedback_model

    result = run_search(task, data_dir=data, config=SearchConfig(**kwargs),
                        resume_run_id=resume)
    if result.best:
        typer.echo(f"\nBest metric: {result.best.metric:.6g}")
        typer.echo(f"Solution:    {result.solution_path}")
        typer.echo(f"Report:      {result.report_path}")
        raise typer.Exit(code=0)
    typer.echo(f"\nNo working solution found. Report: {result.report_path}")
    raise typer.Exit(code=1)


@app.command()
def sessions(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recent sessions to show."),
):
    """List recent sessions (Phase 5)."""
    from agent.memory import get_session_store
    typer.echo(get_session_store().list_sessions(n=limit))


@app.command()
def recall(
    session_id: str = typer.Argument(..., help="A session ID (or unique prefix) from `swarn sessions`."),
):
    """Show one past session's full tool-call log (Phase 5)."""
    from agent.memory import get_session_store
    typer.echo(get_session_store().recall_as_text(session_id))


@app.command()
def index(
    path: str = typer.Argument(..., help="Directory to index for semantic search (Phase 3)."),
):
    """Index a directory into the repo-RAG search index."""
    from agent.tools import index_project
    typer.echo(index_project(path))


def _tables_in(data: dict):
    """
    Yield (page, index, table) from either extract-pdf shape — the document
    tree nests tables inside sections, the page shape lists them per page —
    so --csv-dir works identically in both modes.
    """
    if "sections" in data:
        for section in data["sections"]:
            for i, block in enumerate((b for b in section["blocks"] if b["type"] == "table"), 1):
                yield block["page"], i, block
    else:
        for pg in data["pages"]:
            for table in pg["tables"]:
                yield pg["page"], table["index"], table


def _as_markdown(data: dict) -> str:
    """Render the document tree as Markdown — a human-readable view for
    eyeballing whether the structure came out right, not a data format."""
    lines = []
    if data.get("title"):
        lines += [f"# {data['title']}", ""]
    if data.get("fields"):
        lines += ["| Field | Value |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in data["fields"].items()]
        lines += [""]
    for section in data["sections"]:
        # The title line is usually also the first heading; printing it as
        # both an H1 and an H2 just reads as a duplicate.
        if section["heading"] == data.get("title") and not section["blocks"]:
            continue
        if section["heading"]:
            lines += ["#" * min(6, section["level"] + 1) + f" {section['heading']}", ""]
        for block in section["blocks"]:
            if block["type"] == "paragraph":
                lines += [block["text"], ""]
            elif block["type"] == "list":
                marker = "1." if block["ordered"] else "-"
                lines += [f"{marker} {item}" for item in block["items"]] + [""]
            elif block["type"] == "key_values":
                lines += [f"- **{k}:** {v}" for k, v in block["fields"].items()] + [""]
            elif block["type"] == "table":
                header = block["header"] or [f"col{i+1}" for i in range(block["n_cols"])]
                lines += ["| " + " | ".join(header) + " |",
                          "| " + " | ".join("---" for _ in header) + " |"]
                lines += ["| " + " | ".join(r) + " |" for r in block["rows"]] + [""]
    return "\n".join(lines)


@app.command(name="extract-pdf")
def extract_pdf(
    path: str = typer.Argument(..., help="Path to the PDF file to extract."),
    mode: str = typer.Option("document", "--mode", help="'document' = full structured tree (sections, headings, paragraphs, lists, fields, tables). 'pages' = flatter per-page text + tables."),
    markdown: bool = typer.Option(False, "--markdown", "--md", help="Render as readable Markdown instead of JSON (document mode only)."),
    tables_only: bool = typer.Option(False, "--tables-only", help="Return only tables (pages mode)."),
    page: int = typer.Option(None, "--page", help="Extract only this single 1-based page."),
    out: str = typer.Option(None, "--out", "-o", help="Write the output to this file instead of printing it."),
    csv_dir: str = typer.Option(None, "--csv-dir", help="Also write each detected table to a CSV file in this directory."),
):
    """
    Convert a PDF into structured data.

    Default ('document') mode structures the WHOLE file: title, metadata, a
    document-wide field map, and sections of typed blocks — paragraphs, lists,
    key-values, and tables — in reading order. Use --mode pages for the flatter
    per-page shape.

    Unlike `swarn index`, this does NOT embed or index anything: no model
    download, no network, works offline.
    """
    import json
    from agent.tools import extract_pdf_document, extract_pdf_structured

    if mode not in ("document", "pages"):
        typer.echo(f"Error: --mode must be 'document' or 'pages', not {mode!r}.", err=True)
        raise typer.Exit(code=2)
    if mode == "document":
        result = extract_pdf_document(path, page=page)
    else:
        result = extract_pdf_structured(path, tables_only=tables_only, page=page)

    if result.startswith("Error:"):
        typer.echo(result, err=True)
        raise typer.Exit(code=1)

    data = json.loads(result)

    if csv_dir:
        # Written from the parsed result rather than re-extracting, so the
        # CSVs are guaranteed to match the JSON the caller just got back.
        import csv as _csv
        from pathlib import Path as _Path

        target = _Path(csv_dir)
        target.mkdir(parents=True, exist_ok=True)
        stem = _Path(data["file"]).stem
        for pg, idx, tbl in _tables_in(data):
            dest = target / f"{stem}_p{pg}_table{idx}.csv"
            with dest.open("w", newline="", encoding="utf-8") as fh:
                writer = _csv.writer(fh)
                if tbl["header"]:
                    writer.writerow(tbl["header"])
                writer.writerows(tbl["rows"])
            typer.echo(f"[swarn] wrote {dest}")

    if markdown:
        if mode != "document":
            typer.echo("Error: --markdown needs --mode document.", err=True)
            raise typer.Exit(code=2)
        result = _as_markdown(data)

    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(result)
        typer.echo(f"[swarn] wrote {out}")
    else:
        typer.echo(result)


@app.command(name="to-csv")
def to_csv(
    path: str = typer.Argument(..., help="PDF to convert."),
    out: str = typer.Option(None, "--out", "-o", help="Write everything to this ONE file."),
    directory: str = typer.Option(None, "--dir", "-d", help="Base directory to create the PDF's folder in. Default: beside the PDF."),
    page: List[int] = typer.Option(None, "--page", help="Only these 1-based pages. Repeatable."),
    split_fused: bool = typer.Option(False, "--split-fused", help="Split cells holding two values with no separator (e.g. '124.9833yes')."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Print only the paths written."),
):
    """
    Convert a PDF's tables to CSV and save them.

    Ruled tables come from the stored parse. A PDF with no ruling at all — an
    R data frame printed to PDF, a statement, a report appendix — falls back
    to inferring columns from where the text lines up, which is what recovers
    a borderless dataset that `extract-pdf --csv-dir` returns nothing for.

        swarn to-csv report.pdf                  # -> report/table_1_p2.csv, ...
        swarn to-csv data.pdf --dir ./out        # -> ./out/data/table_1_...csv
        swarn to-csv data.pdf -o dataset.csv     # everything into one flat file
        swarn to-csv data.pdf --page 3 --page 4

    Each PDF gets its own folder, named after it, holding one CSV per table
    plus a `tables.json` manifest saying which pages each file came from and
    what shape it is. A document with six tables should not scatter six loose
    CSVs into a directory shared with every other document converted there.

    Consecutive pages with the same column count are treated as one table
    continued and written to one file; a change of shape starts a new one. If
    nothing table-shaped is found, nothing is written and it says so — a
    garbled CSV that looks like data is worse than an honest failure.
    """
    from swarn.capabilities.doc_intelligence import DocumentIntelligenceError
    from swarn.capabilities.doc_csv import pdf_to_csv

    if out and directory:
        typer.echo("Error: --out and --dir are mutually exclusive.", err=True)
        raise typer.Exit(code=1)

    try:
        result = pdf_to_csv(path, out_path=out, out_dir=directory,
                            pages=list(page) if page else None,
                            split_fused=split_fused)
    except DocumentIntelligenceError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    if quiet:
        for written in result.paths:
            typer.echo(written)
    else:
        typer.echo("")
        typer.echo(result.summary())
        typer.echo("")

    # Nothing found is a legitimate outcome for a prose PDF, but it is not a
    # success — a script converting a directory needs to branch on it.
    raise typer.Exit(code=0 if result.paths else 2)


@app.command(name="doc-inspect")
def doc_inspect(
    path: str = typer.Argument(None, help="PDF or image to inspect. Omit to generate and inspect a mock invoice."),
    page: int = typer.Option(1, "--page", help="1-based page number (PDFs only)."),
    backend: str = typer.Option(None, "--backend", help="Force a backend: 'vlm', 'text' (PDF text layer), 'ocr' (local tesseract), or 'mock' (SYNTHETIC sample data — never chosen automatically for a real document). Default: auto."),
    no_annotate: bool = typer.Option(False, "--no-annotate", help="Skip rendering the annotated image; return fields only."),
    out: str = typer.Option(None, "--out", "-o", help="Annotated image filename (relative paths resolve inside the artifacts directory)."),
    all_pages: bool = typer.Option(False, "--all-pages", help="Process every page of a PDF, one annotated image each."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Print the field summary only, not the full JSON."),
):
    """
    Extract a document's fields WITH their bounding boxes, and draw them.

    Unlike `swarn extract-pdf`, which returns a PDF's text and tables as data,
    this answers *where on the page* each value was read from — and writes an
    annotated PNG with a confidence-coloured box around every field, so an
    extraction can be audited by looking at it rather than by re-reading the
    source document.

    Works with no configuration (deterministic mock extraction). Set
    SWARN_VLM_API_KEY for a real vision model, or --backend ocr for local
    tesseract.
    """
    from swarn.capabilities.doc_intelligence import (
        DocumentInspector, DocumentIntelligenceError, create_mock_document,
    )

    user_supplied = bool(path)
    try:
        if not user_supplied:
            # We generate this document, so mock extraction describes it
            # accurately. For a document the USER supplies, backend selection
            # never falls back to mock — see _resolve_backend.
            path = create_mock_document()
            backend = backend or "mock"
            typer.echo(f"[swarn] no document given — generated a mock invoice at {path}")

        inspector = DocumentInspector(backend=backend)
        if all_pages:
            results = inspector.process_all_pages(path, annotate=not no_annotate)
        else:
            results = [inspector.process_document(
                path, page_number=page, annotate=not no_annotate, output_path=out)]
    except DocumentIntelligenceError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    for result in results:
        if result.backend == "mock" and user_supplied:
            typer.echo(
                "[swarn] WARNING: backend 'mock' returns SYNTHETIC SAMPLE DATA — "
                "the values below are not this document's contents.", err=True)
        typer.echo(result.summary())
        if not quiet:
            typer.echo(result.to_json())


@app.command(name="ingest")
def ingest(
    path: str = typer.Argument(None, help="PDF or image to parse and store. Omit with --list to just show what has been ingested."),
    backend: str = typer.Option(None, "--backend", help="Force how the document is read: 'text' (PDF text layer) or 'ocr' (local tesseract). Default: auto."),
    render_pages: bool = typer.Option(False, "--render-pages", help="Also cache a rendered image of each page, so evidence images can be drawn without the original file. Costs ~200 KB/page."),
    force: bool = typer.Option(False, "--force", help="Re-parse even if this document is already stored."),
    list_only: bool = typer.Option(False, "--list", help="List ingested documents and exit."),
):
    """
    Parse a document ONCE into a structured JSON representation on disk.

    Every later `swarn ask` about the same file loads that JSON instead of
    re-reading the PDF — which matters most for scanned documents, where the
    alternative is a full OCR pass per question.

    The stored form keeps word-level text, bounding boxes, per-word confidence,
    line ids, page dimensions, and tables: everything the evidence and
    bounding-box behaviour needs, so nothing is lost by not re-parsing.

        swarn ingest report.pdf
        swarn ingest --list
    """
    from swarn.capabilities.doc_intelligence import DocumentIntelligenceError
    from swarn.capabilities.doc_store import (
        ingest_document, list_documents, load_for_file, store_path,
    )

    if list_only or not path:
        stored = list_documents()
        if not stored:
            typer.echo("No documents ingested yet. Run: swarn ingest <file.pdf>")
            raise typer.Exit(code=0)
        typer.echo(f"{len(stored)} ingested document(s):")
        for item in stored:
            typer.echo(f"  {item['document_id']:<34} {item['document_name']:<28} "
                       f"{item['page_count']:>3}p  {item['backend']:<5} "
                       f"{item['size_kb']:>7} KB  {item['ingested_at']}")
        raise typer.Exit(code=0)

    try:
        if not force:
            existing = load_for_file(path)
            if existing is not None:
                typer.echo(
                    f"[swarn] already ingested: {existing.document_id} "
                    f"({existing.page_count} pages, {existing.backend}, "
                    f"{existing.ingested_at})")
                typer.echo(f"[swarn] stored at {store_path(existing.document_id)}")
                typer.echo("[swarn] use --force to re-parse.")
                raise typer.Exit(code=0)

        document = ingest_document(path, backend=backend, render_pages=render_pages)
    except DocumentIntelligenceError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    destination = store_path(document.document_id)
    typer.echo(f"[swarn] ingested {document.document_name}")
    typer.echo(f"         document_id : {document.document_id}")
    typer.echo(f"         backend     : {document.backend}")
    typer.echo(f"         pages       : {document.page_count}")
    typer.echo(f"         lines/words : {document.n_lines():,} / {document.n_words():,}")
    typer.echo(f"         tables      : {sum(len(p.tables) for p in document.pages)}")
    typer.echo(f"         stored at   : {destination}")
    if document.tables_dir:
        from pathlib import Path as _P
        n_csv = len(list(_P(document.tables_dir).glob("*.csv")))
        typer.echo(f"         tables → csv: {n_csv} file(s) in {document.tables_dir}/")
    typer.echo(f'[swarn] ask about it:  swarn ask "<question>" {path}')
    if document.tables_dir:
        typer.echo(f"[swarn] its tables:    ls {document.tables_dir}")


@app.command(name="ask")
def ask(
    question: str = typer.Argument(..., help="The question to answer about the document."),
    path: str = typer.Argument(..., help="PDF or image to read."),
    page: int = typer.Option(None, "--page", help="Restrict to a single 1-based page. Default: search the whole document."),
    backend: str = typer.Option(None, "--backend", help="Force how the document is READ: 'text' (PDF text layer) or 'ocr' (local tesseract). Default: auto."),
    no_annotate: bool = typer.Option(False, "--no-annotate", help="Skip rendering the evidence image."),
    show_json: bool = typer.Option(False, "--json", help="Print the full result as JSON."),
):
    """
    Ask a question about a document and get an answer with its evidence.

    Unlike `swarn doc-inspect`, which extracts whatever fields a page holds,
    this answers a specific question — including one whose answer is not
    printed anywhere in the document ("what was the percentage increase") and
    has to be derived from figures that are.

    Because the derivation runs through an LLM, the answer is reported with the
    figures it rests on: each one's page, its bounding box, and an annotated
    image with those figures boxed. Quotes are verified against the document
    text and the arithmetic is re-evaluated locally, so a fabricated citation
    or a bad sum is surfaced rather than printed as fact.

        swarn ask "what was the percentage increase in revenue" report.pdf

    `swarn run` reaches this same code path when its argument is a plain
    question about a document, so the two are interchangeable for that case.
    This command stays as the explicit, unambiguous form: it never routes
    anywhere else, and it takes its arguments in a fixed order.
    """
    raise typer.Exit(code=_answer_document(
        path, question, page=page, backend=backend,
        annotate=not no_annotate, show_json=show_json))


@app.command(name="mcp-serve")
def mcp_serve():
    """
    V3: run the Swarn MCP server over stdio, exposing swarn_submit_task /
    swarn_task_status / swarn_get_messages / swarn_list_tasks to any MCP client
    (Claude Code, Cursor, Windsurf, Zed, ...).

    Register with Claude Code:  claude mcp add swarn -- swarn mcp-serve
    """
    from agent.mcp_server import main as serve_mcp
    serve_mcp()


@app.command()
def playbook(
    clear: bool = typer.Option(False, "--clear", help="Erase all learned lessons."),
):
    """V3: show (or clear) the cross-run playbook — the lessons the agent
    has distilled from past search runs."""
    from agent.knowledge import KnowledgeStore
    store = KnowledgeStore()
    if clear:
        import os as _os
        try:
            _os.remove(store.playbook_path)
        except OSError:
            pass
        typer.echo("Playbook cleared.")
        return
    pb = store.playbook()
    typer.echo(pb or "(playbook is empty — it fills up as search runs complete)")


@app.command(name="guardrail-benchmark")
def guardrail_benchmark():
    """Run Phase 15's canned prompt-injection detection benchmark."""
    from agent.observability import get_benchmark_harness
    typer.echo(get_benchmark_harness().run())


@app.command()
def serve(
    port: int = typer.Option(8420, "--port", "-p", help="Port for the dashboard web server."),
    host: str = typer.Option("127.0.0.1", help="Host to bind to."),
):
    """
    Launch Phase 16's web dashboard — a live view of agent runs streamed
    over websockets, plus a session history browser. Blocks until
    interrupted (Ctrl+C).
    """
    import uvicorn
    typer.echo(f"[swarn] Dashboard starting at http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run("agent.dashboard:app", host=host, port=port, log_level="warning")


def main():
    app()


if __name__ == "__main__":
    main()
