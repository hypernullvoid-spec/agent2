"""
The Swarn CLI: interactive REPL and headless one-shot — Rich terminal UI.

This lives inside the package rather than at the repo root because the
console-script entry point has to be importable from an installed copy: the
old root-level main.py resolved fine from a source checkout and failed with
ModuleNotFoundError once installed, since it was not part of the `agent`
package. Run it as `swarn` (installed) or `python -m agent.cli` (checkout).

Two modes (the ml-intern shape)
────────────────────────────────
    swarn                        # interactive: a conversation, tools gated
                                 # behind approval prompts
    swarn "build me a model"     # headless: one prompt, auto-approved, exits
                                 # with a meaningful status code
    swarn run "..."              # the same headless run, stated explicitly
    swarn team "..."             # headless through the multi-agent pipeline
    swarn --help                 # everything else

A bare prompt works because main() rewrites argv: the first positional
argument that isn't a known subcommand is treated as `run <prompt>`. Click
groups can't carry their own positional argument without swallowing the
subcommand name, so the rewrite happens before Typer sees argv.

Interactive vs headless, concretely
────────────────────────────────────
  • Approval — interactive prompts before anything side-effecting (see
    agent/core/approval_policy.py); headless auto-approves, which is what
    makes it usable from a script or CI job. `/yolo` turns interactive into
    auto-approve for the session.
  • Memory — interactive carries message history across turns, so "now add
    tests for that" resolves against the previous turn. Headless is one shot.
  • Output — interactive streams with the CRT typewriter effect; headless
    prints structured panels and skips streaming (also skipped automatically
    when stdout isn't a terminal, and by --no-stream).
  • Exit code — headless exits 0 on outcome="complete", 1 otherwise.

Settings persist in ~/.config/swarn/cli_agent_config.json (agent/config.py):
model, reasoning effort, yolo mode, tool runtime, max iterations, trace
visibility. `/model`, `/effort`, `/yolo` and `/share-traces` write to it.

Interactive commands
─────────────────────
    /help                — list commands           /new     — fresh conversation
    /model [id]          — show or switch model    /undo    — drop the last turn
    /effort [level]      — reasoning effort        /compact — shrink context
    /yolo                — toggle auto-approve     /clear   — clear workspace
    /status              — model, turns, settings  /plan    — current plan
    /resume [id]         — reload a past session   /quit    — exit
    /share-traces [public|private]
    history [n] · recall <id> · index <path> · team <task> · report · guardrails

Phase 15 note: the guardrail policy (prompt-injection scanning on every tool
result) is ALWAYS on. OpenTelemetry tracing is opt-in: set
SWARN_ENABLE_TRACING=1 to wrap every LLM/tool call in a span (exported to the
console unless OTEL_EXPORTER_ENDPOINT is also set).

Phase 16 note: a run started here will NOT appear live in the dashboard
(`swarn serve`) — the dashboard and this process have separate in-memory
session stores. Both show completed runs in session history. See
dashboard.py's module docstring for why.
"""

import asyncio
import atexit
import os
import shutil
import sys
import time
from contextlib import contextmanager
from typing import Optional

import typer
from dotenv import load_dotenv

from agent.core.agent_loop import AgentLoop
from agent.config import CLIConfig, DEFAULT_CONFIG_PATH, load_config, save_config
from agent.core.approval_policy import describe_operation
from agent.llm import DEPLOYED_BASE_URL, DEPLOYED_MODEL_NAME
from agent.observability import GuardrailPolicy, ObservabilityHooks
from agent.core.orchestrator import Orchestrator
from agent.runtime.sandbox import close_sandbox
from agent.core.self_correction import SelfCorrectionPolicy
from agent.runtime.tools import TOOL_REGISTRY, WORKSPACE_DIR
from agent.utils.terminal_display import (
    get_console,
    get_headless_display,
    indent,
    print_approval_header,
    print_approval_item,
    print_banner,
    print_compacted,
    print_help,
    print_init_done,
    print_markdown,
    print_plan,
    print_yolo_approve,
    reset_headless_display,
    set_stream_enabled,
)

console = get_console()

VALID_EFFORT_LEVELS = ("low", "medium", "high")

app = typer.Typer(
    name="swarn",
    help='Swarn - autonomous AI engineering agent. Run bare for interactive mode, or `swarn "task"` for headless.',
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
)


@contextmanager
def _cancellable():
    """Yield a cancel event that Ctrl+C sets, so streaming output can stop
    cleanly instead of unwinding through a half-written ANSI style run."""
    cancel_event = asyncio.Event()
    try:
        yield cancel_event
    except KeyboardInterrupt:
        cancel_event.set()
        raise


def _apply_runtime_options(
    config: CLIConfig,
    model: Optional[str],
    max_iterations: Optional[int],
    no_stream: bool,
    sandbox_tools: bool,
) -> CLIConfig:
    """Overlay this invocation's flags onto the persisted config.

    Flags win for the current process but are NOT written back to disk —
    only the interactive slash commands persist. That keeps a one-off
    `--model X` from silently redefining the default for later runs.
    """
    if model:
        config.model_name = model
    if max_iterations:
        config.max_iterations = max_iterations
    if no_stream:
        set_stream_enabled(False)
    if sandbox_tools or config.tool_runtime == "sandbox":
        config.tool_runtime = "sandbox"
        # execution.py reads this when it builds the backend, so it has to be
        # set before the first tool call constructs one.
        os.environ["SWARN_SANDBOX"] = "docker"
    return config


def _tool_runtime_label(config: CLIConfig) -> str:
    return "docker sandbox" if config.tool_runtime == "sandbox" else "local filesystem"


def _make_observability_hooks() -> Optional[ObservabilityHooks]:
    if os.environ.get("SWARN_ENABLE_TRACING") != "1":
        return None
    endpoint = os.environ.get("OTEL_EXPORTER_ENDPOINT")
    console.print(
        f"[info]OpenTelemetry tracing enabled (exporting to {endpoint or 'console'}).[/info]\n"
    )
    return ObservabilityHooks(exporter_endpoint=endpoint)


def _create_agent(
    config: CLIConfig,
    observability_hooks: Optional[ObservabilityHooks] = None,
    approval_callback=None,
    keep_history: bool = False,
) -> AgentLoop:
    """Create an AgentLoop wired to the current config and mode."""
    return AgentLoop(
        model=config.model_name,
        correction_policy=SelfCorrectionPolicy(max_consecutive=3),
        guardrail_policy=GuardrailPolicy(),
        observability_hooks=observability_hooks,
        approval_callback=approval_callback,
        max_iterations=config.max_iterations,
        keep_history=keep_history,
    )


# ── headless mode ──────────────────────────────────────────────────────────


def _run_headless(
    task: str,
    config: CLIConfig,
    use_team: bool = False,
    no_tester: bool = False,
    show_report: bool = True,
    show_progress: bool = True,
) -> int:
    """Run one task unattended and return the process exit code.

    No approval callback is passed, so every tool call runs without a prompt
    — headless mode exists to be scriptable, and a prompt on stdin nobody is
    watching would just hang.
    """
    observability_hooks = _make_observability_hooks()
    atexit.register(close_sandbox)

    reset_headless_display()
    display = get_headless_display(show_progress=show_progress)
    display.start_run(task, config.model_name, "team" if use_team else "single")

    start_time = time.monotonic()

    try:
        if use_team:
            orchestrator = Orchestrator(
                model=config.model_name,
                include_tester=not no_tester,
                guardrail_policy=GuardrailPolicy(),
                observability_hooks=observability_hooks,
            )
            result = orchestrator.run(task)
            if show_report:
                display.print_markdown(result["report_markdown"])
            outcome, session_id = result["final_outcome"], result["session_id"]
        else:
            agent = _create_agent(config, observability_hooks=observability_hooks)
            result = agent.run(task)
            outcome, session_id = result["outcome"], result["session_id"]
    except KeyboardInterrupt:
        display.print_error("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 — headless must not dump a traceback at a script
        display.print_error(str(exc))
        return 1

    display.print_result(outcome, session_id, time.monotonic() - start_time)
    return 0 if outcome == "complete" else 1


# ── interactive mode ───────────────────────────────────────────────────────


class _Approver:
    """Approval prompt for interactive mode.

    Answers: y (this call) · n (refuse) · a (approve everything from here on,
    same as /yolo). A non-tty stdin auto-approves rather than hanging.
    """

    def __init__(self, config: CLIConfig):
        self.config = config

    def __call__(self, tool_name: str, tool_input: dict) -> bool:
        if self.config.yolo_mode:
            print_yolo_approve(1)
            return True
        if not sys.stdin.isatty():
            return True

        print_approval_header(1)
        print_approval_item(1, 1, tool_name, describe_operation(tool_name, tool_input))
        try:
            answer = (
                console.input(f"{indent()}[bold]approve?[/bold] [y/n/a] ")
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False
        if answer in ("a", "all"):
            self.config.yolo_mode = True
            console.print("  [info]auto-approve on for the rest of this session.[/info]")
            return True
        return answer in ("", "y", "yes")


def _cmd_model(arg: str, config: CLIConfig) -> None:
    """/model — show the current model, or record a different one.

    Routing note: agent/llm/router.py hard-routes every call to one deployed
    endpoint, so switching changes what is recorded and displayed, not where
    traffic goes. Saying so plainly beats a switch that appears to work.
    """
    if not arg:
        console.print(f"  [info]Model:    {config.model_name}[/info]")
        console.print(f"  [info]Endpoint: {DEPLOYED_BASE_URL}[/info]")
        if config.model_name != DEPLOYED_MODEL_NAME:
            console.print(
                f"  [yellow]Note:[/yellow] all calls still route to the deployed "
                f"model ({DEPLOYED_MODEL_NAME}) — see agent/llm/router.py."
            )
        console.print("  [dim]/model <id> to record a different model.[/dim]")
        return
    old, config.model_name = config.model_name, arg
    save_config(config)
    console.print(f"  [info]Model: {old} → {arg}  (saved)[/info]")
    if arg != DEPLOYED_MODEL_NAME:
        console.print(
            f"  [yellow]Note:[/yellow] requests still go to the deployed endpoint "
            f"({DEPLOYED_MODEL_NAME} @ {DEPLOYED_BASE_URL}). To change that, set "
            "SWARN_DEPLOYED_MODEL / SWARN_DEPLOYED_BASE_URL."
        )


def _cmd_effort(arg: str, config: CLIConfig) -> None:
    if not arg:
        console.print(f"  [info]Reasoning effort: {config.reasoning_effort or 'default'}[/info]")
        console.print(f"  [dim]/effort <{'|'.join(VALID_EFFORT_LEVELS)}>[/dim]")
        return
    if arg not in VALID_EFFORT_LEVELS:
        console.print(f"  [bold red]Unknown effort level:[/bold red] {arg}")
        return
    config.reasoning_effort = arg
    save_config(config)
    console.print(f"  [info]Reasoning effort: {arg}  (saved)[/info]")


def _cmd_share_traces(arg: str, config: CLIConfig) -> None:
    if not arg:
        console.print(
            f"  [info]Trace visibility: {'public' if config.share_traces else 'private'}[/info]"
        )
        console.print("  [dim]/share-traces <public|private>[/dim]")
        return
    if arg not in ("public", "private"):
        console.print("  [bold red]Expected 'public' or 'private'.[/bold red]")
        return
    config.share_traces = arg == "public"
    save_config(config)
    console.print(f"  [info]Trace visibility: {arg}  (saved)[/info]")


def _cmd_status(config: CLIConfig, agent: AgentLoop, turns: int) -> None:
    console.print()
    console.print(f"  [bold]Model[/bold]           {config.model_name}")
    console.print(f"  [bold]Endpoint[/bold]        {DEPLOYED_BASE_URL}")
    console.print(f"  [bold]Turns[/bold]           {turns}")
    console.print(f"  [bold]History[/bold]         {len(agent.history)} messages")
    console.print(f"  [bold]Tool runtime[/bold]    {_tool_runtime_label(config)}")
    console.print(f"  [bold]Max iterations[/bold]  {agent.max_iterations}")
    console.print(f"  [bold]Auto-approve[/bold]    {'on' if config.yolo_mode else 'off'}")
    console.print(f"  [bold]Effort[/bold]          {config.reasoning_effort or 'default'}")
    console.print(f"  [bold]Traces[/bold]          {'public' if config.share_traces else 'private'}")
    console.print(f"  [bold]Config[/bold]          {DEFAULT_CONFIG_PATH}")
    console.print()


def _cmd_resume(arg: str, agent: AgentLoop) -> None:
    """/resume — reload a past session's task and summary as context for the
    current conversation, so a follow-up can build on an earlier run."""
    from agent.memory import get_session_store

    store = get_session_store()
    if not arg:
        console.print(store.list_sessions(n=10))
        console.print("  [dim]/resume <session id> to load one.[/dim]")
        return
    text = store.recall_as_text(arg)
    if not text or "not found" in text.lower():
        console.print(f"  [bold red]No such session:[/bold red] {arg}")
        return
    agent.history = [
        {"role": "user", "content": f"Context — transcript of an earlier session:\n\n{text}"},
        {"role": "assistant", "content": "Understood. I have that earlier session in mind."},
    ]
    console.print(f"  [info]Resumed session {arg} — its transcript is now in context.[/info]")


def _undo(agent: AgentLoop) -> None:
    """/undo — drop the last user turn and everything after it."""
    if not agent.history:
        console.print("  [info]Nothing to undo.[/info]")
        return
    for i in range(len(agent.history) - 1, -1, -1):
        msg = agent.history[i]
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            del agent.history[i:]
            console.print("  [info]Last turn removed from context.[/info]")
            return
    agent.reset_conversation()
    console.print("  [info]Context cleared.[/info]")


def _clear_workspace() -> None:
    for item in os.listdir(WORKSPACE_DIR):
        if item.startswith("."):
            continue
        path = os.path.join(WORKSPACE_DIR, item)
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    console.print("  [info]Workspace cleared.[/info]")


def _show_recent_sessions(n: int = 3) -> None:
    from agent.memory import get_session_store

    store = get_session_store()
    if store._index:
        console.print(f"\n[dim]Recent sessions (last {min(n, len(store._index))}):[/dim]")
        console.print(store.list_sessions(n=n))
        console.print()


def _run_interactive(config: CLIConfig, no_banner: bool = False) -> None:
    """The REPL: a conversation with tool approval, slash commands, and
    history that carries across turns."""
    observability_hooks = _make_observability_hooks()
    atexit.register(close_sandbox)

    # Built before the banner on purpose: constructing the AgentLoop resolves
    # the LLM client, which may print a routing notice. print_init_done()
    # overwrites the banner's "Tools: loading..." line by walking the cursor
    # back up a fixed number of rows, so nothing may print in between.
    agent = _create_agent(
        config,
        observability_hooks=observability_hooks,
        approval_callback=_Approver(config),
        keep_history=True,
    )

    if not no_banner:
        print_banner(model=config.model_name, tool_runtime=_tool_runtime_label(config))

    print_init_done(tool_count=len(TOOL_REGISTRY))

    # _show_recent_sessions(n=3)
    console.print("\n[dim]Type a task, or '/help' for commands.[/dim]\n")

    last_team_report: Optional[str] = None
    turns = 0

    while True:
        try:
            # lstrip the BOM: piping a UTF-8-with-BOM script into stdin
            # otherwise turns the first "/help" into a task for the agent.
            # The prompt stays flush-left even though agent output is centered.
            raw = console.input("[bold cyan]>[/bold cyan] ").lstrip("﻿").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not raw:
            continue

        head, _, rest = raw.partition(" ")
        cmd, arg = head.lower(), rest.strip()

        if cmd in ("exit", "quit", "/quit", "/exit", "bye"):
            console.print("\n[info]Shutting down...[/info]")
            break

        if cmd in ("/help", "help"):
            print_help()
            continue

        if cmd == "history":
            from agent.memory import get_session_store

            console.print(get_session_store().list_sessions(n=int(arg) if arg.isdigit() else 10))
            continue

        if cmd == "recall" and arg:
            from agent.memory import get_session_store

            console.print(get_session_store().recall_as_text(arg))
            continue

        if cmd == "guardrails":
            console.print(GuardrailPolicy().summary())
            continue

        if cmd == "index" and arg:
            from agent.runtime.tools import index_project

            console.print(index_project(arg))
            continue

        if cmd in ("clear", "/clear"):
            _clear_workspace()
            continue

        if cmd == "report":
            if last_team_report is None:
                console.print("  [info]No 'team' run has completed yet in this session.[/info]")
            else:
                with _cancellable() as cancel_event:
                    asyncio.run(print_markdown(last_team_report, cancel_event=cancel_event))
            continue

        if cmd == "team" and arg:
            orchestrator = Orchestrator(
                model=config.model_name,
                guardrail_policy=GuardrailPolicy(),
                observability_hooks=observability_hooks,
            )
            result = orchestrator.run(arg)
            last_team_report = result["report_markdown"]
            console.print(
                f"\n[info]Multi-agent run finished: {result['final_outcome']}. "
                "Type 'report' to see the full timeline.[/info]"
            )
            continue

        if cmd == "/plan":
            from agent.core.plan import get_current_plan

            if get_current_plan():
                print_plan()
            else:
                console.print("  [info]No plan yet — one appears once the agent lays out steps.[/info]")
            continue

        if cmd == "/new":
            agent.reset_conversation()
            turns = 0
            console.print("  [info]New conversation — previous context dropped.[/info]")
            continue

        if cmd == "/compact":
            before, after = agent.compact_conversation()
            print_compacted(before, after)
            continue

        if cmd == "/undo":
            _undo(agent)
            continue

        if cmd == "/model":
            _cmd_model(arg, config)
            agent.model = config.model_name
            continue

        if cmd == "/effort":
            _cmd_effort(arg.lower(), config)
            continue

        if cmd == "/share-traces":
            _cmd_share_traces(arg.lower(), config)
            continue

        if cmd == "/resume":
            _cmd_resume(arg, agent)
            continue

        if cmd == "/status":
            _cmd_status(config, agent, turns)
            continue

        if cmd == "/yolo":
            config.yolo_mode = not config.yolo_mode
            save_config(config)
            state = "ON — tool calls run without asking" if config.yolo_mode else "OFF — side-effecting tools need approval"
            console.print(f"  [info]Auto-approve {state}  (saved)[/info]")
            continue

        if cmd.startswith("/"):
            console.print(f"  [bold red]Unknown command:[/bold red] {cmd}  [dim](/help)[/dim]")
            continue

        # Anything else is a task for the agent.
        if agent._policy:
            agent._policy.consecutive_errors = 0
        try:
            agent.run(raw)
            turns += 1
        except KeyboardInterrupt:
            from agent.utils.terminal_display import print_interrupted

            print_interrupted()


# ── CLI surface ────────────────────────────────────────────────────────────


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    no_banner: bool = typer.Option(False, "--no-banner", help="Skip the startup banner."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use for this invocation."),
    max_iterations: Optional[int] = typer.Option(None, "--max-iterations", help="Cap on agentic loop iterations."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streamed (typewriter) output."),
    sandbox_tools: bool = typer.Option(False, "--sandbox-tools", help="Run tools in a Docker sandbox instead of the local filesystem."),
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit."),
):
    """
    Swarn — autonomous AI engineering agent.

    Run bare for interactive mode; pass a prompt for a headless one-shot.
    """
    if version:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as get_version

        try:
            console.print(f"swarn v{get_version('swarn')}")
        except PackageNotFoundError:
            console.print("swarn (not installed — running from source)")
        raise typer.Exit(0)

    load_dotenv()
    config = _apply_runtime_options(
        load_config(), model, max_iterations, no_stream, sandbox_tools
    )
    ctx.obj = config

    if ctx.invoked_subcommand:
        return

    _run_interactive(config, no_banner=no_banner)


@app.command()
def run(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="The task for the single agent to perform."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Suppress per-tool progress output."),
):
    """Run one task headlessly (tools auto-approved) and exit 0/1."""
    raise typer.Exit(
        _run_headless(task=task, config=ctx.obj, use_team=False, show_progress=not no_progress)
    )


@app.command()
def team(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="The task for the multi-agent pipeline."),
    no_tester: bool = typer.Option(False, "--no-tester", help="Stop after Reviewer approval, skip the Tester stage."),
    no_report: bool = typer.Option(False, "--no-report", help="Don't print the markdown report."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Suppress per-tool progress output."),
):
    """Run one task headlessly through the Planner→Coder→Reviewer→Tester pipeline."""
    raise typer.Exit(
        _run_headless(
            task=task,
            config=ctx.obj,
            use_team=True,
            no_tester=no_tester,
            show_report=not no_report,
            show_progress=not no_progress,
        )
    )


@app.command()
def solve(
    task: str = typer.Argument(..., help="Full ML task description (target, metric, constraints)."),
    data: str = typer.Option(None, "--data", "-d", help="Directory holding the task's data files (not needed with --resume)."),
    steps: int = typer.Option(20, "--steps", "-s", help="Search budget: number of solution nodes to try."),
    time_limit: int = typer.Option(None, "--time-limit", "-t", help="Wall-clock budget in seconds."),
    drafts: int = typer.Option(4, "--drafts", help="Number of independent initial solutions."),
    search_model: str = typer.Option(None, "--search-model", help="Model for code generation (routing is pinned; display/log only)."),
    feedback_model: str = typer.Option(None, "--feedback-model", help="Model for result review (routing is pinned; display/log only)."),
    exec_timeout: int = typer.Option(600, "--exec-timeout", help="Per-node execution timeout in seconds."),
    workers: int = typer.Option(None, "--workers", "-w", help="How many solution nodes run concurrently (default 1, or SWARN_SEARCH_WORKERS)."),
    token_budget: int = typer.Option(None, "--token-budget", help="Stop the run after this many total LLM tokens."),
    resume: str = typer.Option(None, "--resume", help="Resume a previous run by id; --steps adds that many MORE nodes."),
    no_learn: bool = typer.Option(False, "--no-learn", help="Disable cross-run knowledge: no playbook injection, no reflection."),
):
    """
    Solve an ML task end-to-end via AIDE-style solution tree search
    (draft → debug → improve until the budget is spent). Produces
    runs/<id>/best_solution.py + report.md.
    """
    from pathlib import Path

    from agent.search import SearchConfig, run_search

    if not resume and (not data or not Path(data).is_dir()):
        console.print(f"[bold red]error:[/bold red] data directory not found: {data}")
        raise typer.Exit(code=2)

    kwargs: dict = {
        "steps": steps,
        "time_limit_secs": time_limit,
        "num_drafts": drafts,
        "exec_timeout": exec_timeout,
        "use_knowledge": not no_learn,
        "reflect": not no_learn,
    }
    if workers:
        kwargs["parallel_workers"] = workers
    if token_budget:
        kwargs["token_budget"] = token_budget
    if search_model:
        kwargs["code_model"] = search_model
        kwargs["feedback_model"] = feedback_model or search_model
    elif feedback_model:
        kwargs["feedback_model"] = feedback_model

    result = run_search(task, data_dir=data, config=SearchConfig(**kwargs), resume_run_id=resume)
    if result.best:
        console.print(f"\nBest metric: {result.best.metric:.6g}")
        console.print(f"Solution:    {result.solution_path}")
        console.print(f"Report:      {result.report_path}")
        raise typer.Exit(code=0)
    console.print(f"\nNo working solution found. Report: {result.report_path}")
    raise typer.Exit(code=1)


@app.command()
def sessions(limit: int = typer.Option(10, "--limit", "-n", help="Number of recent sessions to show.")):
    """List recent sessions (Phase 5)."""
    from agent.memory import get_session_store

    console.print(get_session_store().list_sessions(n=limit))


@app.command()
def recall(session_id: str = typer.Argument(..., help="A session ID (or unique prefix) from `swarn sessions`.")):
    """Show one past session's full tool-call log (Phase 5)."""
    from agent.memory import get_session_store

    console.print(get_session_store().recall_as_text(session_id))


@app.command()
def index(path: str = typer.Argument(..., help="Directory to index for semantic search (Phase 3).")):
    """Index a directory into the repo-RAG search index."""
    from agent.runtime.tools import index_project

    console.print(index_project(path))


@app.command()
def config(
    ctx: typer.Context,
    show_path: bool = typer.Option(False, "--path", help="Print the config file path and exit."),
):
    """Show the persisted CLI configuration (~/.config/swarn/cli_agent_config.json)."""
    import json

    if show_path:
        console.print(str(DEFAULT_CONFIG_PATH))
        return
    console.print(f"[dim]{DEFAULT_CONFIG_PATH}"
                  f"{'' if DEFAULT_CONFIG_PATH.exists() else '  (not created yet — defaults shown)'}[/dim]")
    console.print(json.dumps(ctx.obj.to_dict(), indent=2))


@app.command(name="guardrail-benchmark")
def guardrail_benchmark():
    """Run Phase 15's canned prompt-injection detection benchmark."""
    from agent.observability import get_benchmark_harness

    console.print(get_benchmark_harness().run())


@app.command()
def serve(
    port: int = typer.Option(8420, "--port", "-p", help="Port for the dashboard web server."),
    host: str = typer.Option("127.0.0.1", help="Host to bind to."),
):
    """Launch Phase 16's web dashboard."""
    import uvicorn

    console.print(f"[swarn] Dashboard starting at http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run("agent.web.dashboard:app", host=host, port=port, log_level="warning")


@app.command()
def playbook(clear: bool = typer.Option(False, "--clear", help="Erase all learned lessons.")):
    """Show (or clear) the cross-run playbook."""
    from agent.memory.knowledge import KnowledgeStore

    store = KnowledgeStore()
    if clear:
        try:
            os.remove(store.playbook_path)
        except OSError:
            pass
        console.print("Playbook cleared.")
        return
    console.print(store.playbook() or "(playbook is empty -- it fills up as search runs complete)")


@app.command(name="mcp-serve")
def mcp_serve():
    """Run the Swarn MCP server over stdio."""
    from agent.integrations.mcp_server import main as serve_mcp

    serve_mcp()


def _rewrite_bare_prompt(argv: list[str]) -> list[str]:
    """`swarn "do X"` → `swarn run "do X"`.

    Only the first positional argument is considered, and only when it isn't
    already a command name — so `swarn run ...`, `swarn --help` and
    `swarn -m X team ...` all keep their normal meaning.
    """
    commands = set(app_command_names())
    for i, token in enumerate(argv):
        if token.startswith("-"):
            continue
        # An option's value looks positional; skip the token after any option
        # that takes one.
        if i > 0 and argv[i - 1] in ("--model", "-m", "--max-iterations", "--port", "-p", "--host", "-n", "--limit"):
            continue
        if token in commands:
            return argv
        return [*argv[:i], "run", *argv[i:]]
    return argv


def app_command_names() -> list[str]:
    """Names Typer registered, including explicit `name=` overrides."""
    return [cmd.name or cmd.callback.__name__ for cmd in app.registered_commands]


def main() -> None:
    sys.argv[1:] = _rewrite_bare_prompt(sys.argv[1:])
    app()


if __name__ == "__main__":
    main()
