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
  swarn run "<task>"              — single agent, one-shot
  swarn team "<task>"             — Phase 11 multi-agent pipeline, one-shot
  swarn sessions [--limit N]       — Phase 5 session history
  swarn recall <session_id>         — full tool-call log of one past session
  swarn index <path>                 — Phase 3 repo indexing
  swarn serve [--port N]              — Phase 16's dashboard (see dashboard.py)
  swarn guardrail-benchmark            — Phase 15's canned guardrail test suite

Exit codes
────────────
`run`/`team` exit 0 on outcome="complete", 1 otherwise — so
`swarn run "..." && echo "ok"` in a shell script behaves the way you'd
expect a build/test step to behave.
"""

import sys
from pathlib import Path

import typer

# Deployed model name — all calls hard-route to the endpoint configured in
# agent/llm/router.py, so --model flags below are display/log only.
from agent.llm import DEFAULT_MODEL

app = typer.Typer(
    name="swarn",
    help="Swarn — your autonomous AI engineering agent (CLI front end).",
    add_completion=False,
)


@app.command()
def run(
    task: str = typer.Argument(..., help="The task for the single agent to perform."),
    model: str = typer.Option(DEFAULT_MODEL, help="Ignored — all calls route to the deployed endpoint (see agent/llm/router.py)."),
):
    """Run a one-off task through the single agent (full Phase 1–15 toolset) and exit."""
    # Imported lazily inside each command, not at module level — `swarn
    # --help` shouldn't need to construct an LLMClient (which reads the
    # API key from the environment) just to print usage text.
    from agent.core.agent_loop import AgentLoop
    from agent.core.self_correction import SelfCorrectionPolicy
    from agent.observability.observability import GuardrailPolicy

    agent = AgentLoop(
        model=model,
        correction_policy=SelfCorrectionPolicy(),
        guardrail_policy=GuardrailPolicy(),
    )
    try:
        result = agent.run(task)
    except Exception as e:  # noqa: BLE001 — surface a friendly message instead of a raw traceback
        from agent.utils import ui
        from agent.llm.base import LLMError
        ui.console.print()
        if isinstance(e, LLMError):
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                ui.error(
                    "LLM provider rate limit reached. If this is OpenRouter's free tier, "
                    "you're capped at 50 requests/day — it resets at 00:00 UTC, or add "
                    "$10 in credits to unlock 1000 free requests/day."
                )
            else:
                ui.error(f"LLM call failed: {msg.splitlines()[0]}")
        else:
            ui.error(f"Run failed: {type(e).__name__}: {e}")
        raise typer.Exit(code=1)

    from agent.utils import ui
    ui.console.print()
    ui.outcome(result["outcome"], result["session_id"], result.get("summary"))
    raise typer.Exit(code=0 if result["outcome"] == "complete" else 1)


@app.command()
def team(
    task: str = typer.Argument(..., help="The task for the multi-agent pipeline."),
    model: str = typer.Option(DEFAULT_MODEL, help="Ignored — all calls route to the deployed endpoint (see agent/llm/router.py)."),
    no_tester: bool = typer.Option(False, "--no-tester", help="Stop after Reviewer approval, skip the Tester stage."),
):
    """Run a one-off task through the Phase 11 Planner→Coder→Reviewer→Tester pipeline and exit."""
    from agent.core.orchestrator import Orchestrator
    from agent.observability.observability import GuardrailPolicy

    orchestrator = Orchestrator(
        model=model,
        include_tester=not no_tester,
        guardrail_policy=GuardrailPolicy(),
    )
    result = orchestrator.run(task)
    from agent.utils import ui
    ui.console.print()
    ui.markdown(result["report_markdown"])
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
    from agent.memory.memory import get_session_store
    typer.echo(get_session_store().list_sessions(n=limit))


@app.command()
def recall(
    session_id: str = typer.Argument(..., help="A session ID (or unique prefix) from `swarn sessions`."),
):
    """Show one past session's full tool-call log (Phase 5)."""
    from agent.memory.memory import get_session_store
    typer.echo(get_session_store().recall_as_text(session_id))


@app.command()
def index(
    path: str = typer.Argument(..., help="Directory to index for semantic search (Phase 3)."),
):
    """Index a directory into the repo-RAG search index."""
    from agent.runtime.tools import index_project
    typer.echo(index_project(path))


@app.command(name="mcp-serve")
def mcp_serve():
    """
    V3: run the Swarn MCP server over stdio, exposing swarn_submit_task /
    swarn_task_status / swarn_get_messages / swarn_list_tasks to any MCP client
    (Claude Code, Cursor, Windsurf, Zed, ...).

    Register with Claude Code:  claude mcp add swarn -- swarn mcp-serve
    """
    from agent.integrations.mcp_server import main as serve_mcp
    serve_mcp()


@app.command()
def playbook(
    clear: bool = typer.Option(False, "--clear", help="Erase all learned lessons."),
):
    """V3: show (or clear) the cross-run playbook — the lessons the agent
    has distilled from past search runs."""
    from agent.memory.knowledge import KnowledgeStore
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
    from agent.observability.observability import get_benchmark_harness
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
    uvicorn.run("agent.web.dashboard:app", host=host, port=port, log_level="warning")


def main():
    app()


if __name__ == "__main__":
    main()
