"""
The Swarn CLI: interactive REPL and headless one-shot — Rich terminal UI.

This is the single front end for the whole platform. It lives inside the
package rather than at the repo root because the console-script entry point
has to be importable from an installed copy: a root-level main.py resolves
fine from a source checkout and fails with ModuleNotFoundError once
installed, since it is not part of the `agent` package. Run it as `swarn`
(installed) or `python -m agent.cli` (checkout).

Two modes
──────────
    swarn                        # interactive: a conversation, with tools
                                 # gated behind approval prompts
    swarn "build me a model"     # headless: one prompt, auto-approved, exits
                                 # with a meaningful status code
    swarn run "..."              # the same headless run, stated explicitly
    swarn team "..."             # headless through the multi-agent pipeline
    swarn --help                 # everything else

A bare prompt works because main() rewrites argv: the first positional
argument that isn't a known subcommand is treated as `run <prompt>`. Click
groups cannot carry their own positional argument without swallowing the
subcommand name, so the rewrite happens before Typer sees argv.

Interactive vs headless, concretely
────────────────────────────────────
  • Approval — interactive prompts before anything side-effecting (see
    agent/core/approval_policy.py); headless auto-approves, which is what
    makes it usable from a script or a CI job. `/yolo` turns interactive
    into auto-approve for the rest of the session.
  • Memory — interactive carries message history across turns, so "now add
    tests for that" resolves against the previous turn. Headless is one
    shot: it starts from an empty conversation every time.
  • Output — interactive prints the themed banner and streams; headless
    prints structured panels and a final summary table, which is what you
    want in a log file.

Themes
───────
Two interchangeable terminal skins ship in agent/utils/: the green-on-black
CRT `classic` (default) and `lain`. Pick one with SWARN_THEME=lain. Nothing
in this module knows which is active — it imports from the
agent.utils.terminal_display facade, which forwards to whichever is chosen.

Commands
──────────
  swarn run "<task>" [docs...]    — universal entry point: single agent,
                                     or the document fast path when the
                                     task is just a question about a file
  swarn team "<task>"             — multi-agent pipeline, one-shot
  swarn solve "<task>" --data D   — AIDE-style ML solution tree search
  swarn sessions [--limit N]       — session history
  swarn recall <session_id>         — full tool-call log of one past session
  swarn index <path>                 — repo indexing
  swarn extract-pdf <path>            — PDF → structured JSON (no indexing)
  swarn to-csv <path>                  — PDF's tables → CSV file(s) on disk
  swarn doc-inspect <path>             — PDF/image → fields + bounding boxes
                                          + an annotated image
  swarn ingest <path>                   — parse a document once into stored JSON
  swarn ask "<question>" <path>         — answer a question about a document,
                                          with the evidence boxed and cited
                                          (the explicit form of what
                                           `swarn run` routes to)
  swarn config [--path]               — show the persisted CLI configuration
  swarn serve [--port N]              — the web dashboard (see dashboard.py)
  swarn guardrail-benchmark            — canned guardrail test suite

Exit codes
────────────
`run`/`team` exit 0 on outcome="complete", 1 otherwise — so
`swarn run "..." && echo "ok"` in a shell script behaves the way you'd
expect a build/test step to behave. The document fast path additionally
exits 2 when the question is well-formed but the document cannot answer it:
that is a legitimate outcome, not a failure, and a script should be able to
tell the two apart.
"""

import asyncio
import atexit
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import click
import typer
from dotenv import load_dotenv

from agent.config import (
    CLIConfig,
    DEFAULT_CONFIG_PATH,
    load_config,
    save_config,
)
# Deployed model name — all calls hard-route to the endpoint configured in
# agent/llm/router.py, so --model flags below are display/log only.
from agent.llm import DEFAULT_MODEL, DEPLOYED_BASE_URL, DEPLOYED_MODEL_NAME
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
    print_interrupted,
    print_plan,
    print_yolo_approve,
    read_user_input,
    reset_headless_display,
    set_stream_enabled,
)

console = get_console()

VALID_EFFORT_LEVELS = ("low", "medium", "high")

app = typer.Typer(
    name="swarn",
    help='Swarn — autonomous AI engineering agent. Run bare for interactive '
         'mode, or `swarn "task"` for a headless one-shot.',
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
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


# ═══════════════════════════════════════════════════════════════════════════
# RUNTIME WIRING — shared by both modes
# ═══════════════════════════════════════════════════════════════════════════


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

    Flags win for the current process but are NOT written back to disk — only
    the interactive slash commands persist. That keeps a one-off `--model X`
    from silently redefining the default for every later run.
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


def _make_observability_hooks():
    """Tracing is opt-in: span export adds console noise most runs don't want,
    and most environments have no collector to send spans to anyway."""
    from agent import config as agent_config

    if not agent_config.tracing_enabled():
        return None
    from agent.observability.observability import ObservabilityHooks

    endpoint = agent_config.otel_endpoint()
    console.print(
        f"[info]OpenTelemetry tracing enabled (exporting to {endpoint or 'console'}).[/info]\n"
    )
    return ObservabilityHooks(exporter_endpoint=endpoint)


def _create_agent(
    config: CLIConfig,
    observability_hooks=None,
    approval_callback=None,
    keep_history: bool = False,
    single_session: bool = False,
    on_tool_result=None,
):
    """Create an AgentLoop wired to the current config and mode.

    Imported lazily, like everything else that touches the model stack:
    `swarn --help` should not have to construct an LLM client just to print
    usage text.
    """
    from agent.core.agent_loop import AgentLoop
    from agent.core.self_correction import SelfCorrectionPolicy
    from agent.observability.observability import GuardrailPolicy

    return AgentLoop(
        model=config.model_name,
        correction_policy=SelfCorrectionPolicy(max_consecutive=3),
        guardrail_policy=GuardrailPolicy(),
        observability_hooks=observability_hooks,
        approval_callback=approval_callback,
        max_iterations=config.max_iterations,
        keep_history=keep_history,
        single_session=single_session,
        on_tool_result=on_tool_result or _print_grounded_tool_result,
    )


def _report_run_failure(exc: BaseException) -> None:
    """Turn an exception into one readable line instead of a traceback.

    A rate limit is called out specifically because it is the single most
    common failure and the least self-explanatory: "429" alone tells a user
    nothing about what to do next.
    """
    from agent.llm.base import LLMError
    from agent.utils import ui

    ui.console.print()
    if isinstance(exc, LLMError):
        message = str(exc)
        if "429" in message or "rate limit" in message.lower():
            ui.error(
                "LLM provider rate limit reached. If this is OpenRouter's free tier, "
                "you're capped at 50 requests/day — it resets at 00:00 UTC, or add "
                "$10 in credits to unlock 1000 free requests/day."
            )
        else:
            ui.error(f"LLM call failed: {message.splitlines()[0]}")
    else:
        ui.error(f"Run failed: {type(exc).__name__}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# HEADLESS MODE
# ═══════════════════════════════════════════════════════════════════════════


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
    — headless mode exists to be scriptable, and a prompt written to a stdin
    nobody is watching would simply hang.
    """
    from agent.runtime.sandbox import close_sandbox

    observability_hooks = _make_observability_hooks()
    atexit.register(close_sandbox)

    reset_headless_display()
    display = get_headless_display(show_progress=show_progress)
    display.start_run(task, config.model_name, "team" if use_team else "single")

    start_time = time.monotonic()

    try:
        if use_team:
            from agent.core.orchestrator import Orchestrator
            from agent.observability.observability import GuardrailPolicy

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
            def _count_and_print(tool_name: str, tool_input: dict, raw_result: str) -> None:
                """Feed the summary panel's tool counter as calls go by.

                The agent loop renders its own per-tool lines through
                agent/utils/ui.py, so the headless manager never sees them and
                its counter would otherwise report 0 for every run. This is
                the one point every tool call passes through.
                """
                display.note_tool(tool_name)
                _print_grounded_tool_result(tool_name, tool_input, raw_result)

            agent = _create_agent(
                config,
                observability_hooks=observability_hooks,
                on_tool_result=_count_and_print,
            )
            result = agent.run(task)
            outcome, session_id = result["outcome"], result["session_id"]
    except KeyboardInterrupt:
        display.print_error("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 — headless must never dump a traceback at a script
        display.print_error(str(exc))
        return 1

    display.print_result(outcome, session_id, time.monotonic() - start_time)
    return 0 if outcome == "complete" else 1


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════════════════


class _Approver:
    """The approval prompt for interactive mode.

    Answers: y (this call) · n (refuse) · a (approve everything from here on,
    the same as /yolo). A non-tty stdin auto-approves rather than hanging,
    so `echo "task" | swarn` still works.
    """

    def __init__(self, config: CLIConfig):
        self.config = config

    def __call__(self, tool_name: str, tool_input: dict) -> bool:
        from agent.core.approval_policy import describe_operation

        if self.config.yolo_mode:
            print_yolo_approve(1)
            return True
        if not sys.stdin.isatty():
            return True

        print_approval_header(1)
        print_approval_item(1, 1, tool_name, describe_operation(tool_name, tool_input))
        try:
            answer = (
                console.input(f"{indent()}[bold]approve?[/bold] [y/n/a] ").strip().lower()
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
    the traffic goes. Saying so plainly beats a switch that appears to work.
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


def _cmd_status(config: CLIConfig, agent, turns: int) -> None:
    console.print()
    console.print(f"  [bold]Model[/bold]           {config.model_name}")
    console.print(f"  [bold]Endpoint[/bold]        {DEPLOYED_BASE_URL}")
    console.print(f"  [bold]Turns[/bold]           {turns}")
    console.print(f"  [bold]History[/bold]         {len(agent.history)} messages")
    session = getattr(agent, "_session", None)
    console.print(
        f"  [bold]Session[/bold]         "
        f"{session.id[:8] if session else '— (starts on your first question)'}"
    )
    console.print(f"  [bold]Tool runtime[/bold]    {_tool_runtime_label(config)}")
    console.print(f"  [bold]Max iterations[/bold]  {agent.max_iterations}")
    console.print(f"  [bold]Auto-approve[/bold]    {'on' if config.yolo_mode else 'off'}")
    console.print(f"  [bold]Effort[/bold]          {config.reasoning_effort or 'default'}")
    console.print(f"  [bold]Traces[/bold]          {'public' if config.share_traces else 'private'}")
    console.print(f"  [bold]Config[/bold]          {DEFAULT_CONFIG_PATH}")
    console.print()


def _cmd_resume(arg: str, agent) -> None:
    """/resume — reload a past session's task and summary as context for the
    current conversation, so a follow-up can build on an earlier run."""
    from agent.memory.memory import get_session_store

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


def _undo(agent) -> None:
    """/undo — drop the last user turn and everything after it."""
    if not agent.history:
        console.print("  [info]Nothing to undo.[/info]")
        return
    for i in range(len(agent.history) - 1, -1, -1):
        message = agent.history[i]
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            del agent.history[i:]
            console.print("  [info]Last turn removed from context.[/info]")
            return
    agent.reset_conversation()
    console.print("  [info]Context cleared.[/info]")


def _clear_workspace() -> None:
    from agent.runtime.tools import WORKSPACE_DIR

    for item in os.listdir(WORKSPACE_DIR):
        if item.startswith("."):
            continue
        path = os.path.join(WORKSPACE_DIR, item)
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    console.print("  [info]Workspace cleared.[/info]")


# REPL word → subcommand name. Only the document commands are exposed: a
# document session is naturally iterative — ingest once, then ask several
# questions — and dropping to a second shell between questions breaks that
# rhythm. `run` and `team` are deliberately absent; typing a bare task
# already does the first, and `team <task>` is handled separately so it can
# share the REPL's guardrail and observability instances.
_REPL_SUBCOMMANDS = {
    "ask": "ask",
    "ingest": "ingest",
    "inspect": "doc-inspect",
    "to-csv": "to-csv",
    "extract-pdf": "extract-pdf",
}


def _repl_document_command(cmd: str, arg: str) -> bool:
    """Run a document subcommand typed at the REPL prompt.

    Dispatched through the very same Click command the shell invokes, rather
    than reimplemented against the underlying capability functions. Calling
    those directly is what the first version of this did, and it was wrong in
    two different ways at once (inspect_document returns a dict, not an
    object with .summary(); pdf_to_csv returns a ConversionResult, not a list
    of paths) — mistakes that were possible only because the REPL was
    re-deriving output the subcommand already knows how to render. Going
    through Click means the two surfaces cannot disagree, and every flag the
    subcommand accepts works here too.

    Returns True when `cmd` was ours to handle.
    """
    import shlex

    subcommand = _REPL_SUBCOMMANDS.get(cmd)
    if subcommand is None:
        return False
    if not arg:
        console.print(f"  [bold red]Usage:[/bold red] {cmd} <arguments>  "
                      f"[dim](swarn {subcommand} --help)[/dim]")
        return True

    try:
        argv = shlex.split(arg)
    except ValueError as exc:  # unbalanced quotes
        console.print(f"  [bold red]Could not parse arguments:[/bold red] {exc}")
        return True

    click_command = typer.main.get_command(app)
    try:
        # standalone_mode=False stops Click from calling sys.exit() on
        # completion or on a usage error — which would take the REPL down
        # with it. Exit/UsageError come back as exceptions instead.
        click_command.main(
            args=[subcommand, *argv],
            standalone_mode=False,
            # The document subcommands read their config from ctx.obj only
            # via `or load_config()`, so nothing extra needs threading here.
        )
    except (typer.Exit, SystemExit):
        pass  # a non-zero document exit code is information, not a crash
    except click.UsageError as exc:
        console.print(f"  [bold red]{exc.format_message()}[/bold red]  "
                      f"[dim](swarn {subcommand} --help)[/dim]")
    except Exception as exc:  # noqa: BLE001 — one bad command must not end the session
        console.print(f"  [bold red]{type(exc).__name__}:[/bold red] {exc}")
    return True


def _run_interactive(config: CLIConfig, no_banner: bool = False) -> None:
    """The REPL: a conversation with tool approval, slash commands, and
    history that carries across turns."""
    from agent.observability.observability import GuardrailPolicy
    from agent.runtime.sandbox import close_sandbox
    from agent.runtime.tools import TOOL_REGISTRY

    observability_hooks = _make_observability_hooks()
    atexit.register(close_sandbox)
    guardrails = GuardrailPolicy()

    # Built before the banner on purpose: constructing the AgentLoop resolves
    # the LLM client, which may print a routing notice. print_init_done()
    # overwrites the banner's "Tools: loading..." line by walking the cursor
    # back up a fixed number of rows, so nothing may print in between.
    agent = _create_agent(
        config,
        observability_hooks=observability_hooks,
        approval_callback=_Approver(config),
        keep_history=True,
        # One sitting at this prompt is one session. Every question asked here
        # is recorded as a turn inside it, and it is closed when the REPL
        # exits — so `history` lists one entry per conversation rather than
        # one per question.
        single_session=True,
    )
    # Even a hard exit (an unhandled crash, a closed terminal) should leave a
    # finalized session behind rather than one that looks like it is still
    # running. Turn-by-turn checkpointing already protects the content; this
    # protects the closing state.
    atexit.register(agent.close_session)

    if not no_banner:
        print_banner(model=config.model_name, tool_runtime=_tool_runtime_label(config))

    print_init_done(tool_count=len(TOOL_REGISTRY))
    console.print("\n[dim]Type a task, or '/help' for commands.[/dim]\n")

    last_team_report: Optional[str] = None
    turns = 0

    while True:
        try:
            # lstrip the BOM: piping a UTF-8-with-BOM script into stdin
            # otherwise turns the first "/help" into a task for the agent.
            # Ghost placeholder text sits inside the input line and
            # clears on the first keypress (see read_user_input).
            raw = read_user_input().lstrip("﻿").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            agent.close_session()
            break

        if not raw:
            continue

        head, _, rest = raw.partition(" ")
        cmd, arg = head.lower(), rest.strip()

        if cmd in ("exit", "quit", "/quit", "/exit", "bye"):
            console.print("\n[info]Shutting down...[/info]")
            agent.close_session()
            break

        if cmd in ("/help", "help"):
            print_help()
            continue

        if cmd == "history":
            from agent.memory.memory import get_session_store

            console.print(
                get_session_store().list_sessions(n=int(arg) if arg.isdigit() else 10)
            )
            continue

        if cmd == "recall" and arg:
            from agent.memory.memory import get_session_store

            console.print(get_session_store().recall_as_text(arg))
            continue

        if cmd == "guardrails":
            console.print(guardrails.summary())
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
                from agent.utils import ui

                ui.markdown(last_team_report)
            continue

        if cmd == "team" and arg:
            from agent.core.orchestrator import Orchestrator

            orchestrator = Orchestrator(
                model=config.model_name,
                guardrail_policy=guardrails,
                observability_hooks=observability_hooks,
            )
            result = orchestrator.run(arg)
            last_team_report = result["report_markdown"]
            console.print(
                f"\n[info]Multi-agent run finished: {result['final_outcome']}. "
                "Type 'report' to see the full timeline.[/info]"
            )
            continue

        if _repl_document_command(cmd, arg):
            continue

        if cmd == "/plan":
            from agent.core.plan import get_current_plan

            if get_current_plan():
                print_plan()
            else:
                console.print(
                    "  [info]No plan yet — one appears once the agent lays out steps.[/info]"
                )
            continue

        if cmd == "/new":
            # Closes the current session too, not just the message history:
            # /new means "that conversation is over", and the next question
            # starts a fresh session the same way reopening the REPL would.
            agent.close_session()
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
            state = (
                "ON — tool calls run without asking"
                if config.yolo_mode
                else "OFF — side-effecting tools need approval"
            )
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
            print_interrupted()
        except Exception as exc:  # noqa: BLE001 — one bad turn must not end the session
            _report_run_failure(exc)


# ═══════════════════════════════════════════════════════════════════════════
# CLI SURFACE
# ═══════════════════════════════════════════════════════════════════════════


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    no_banner: bool = typer.Option(False, "--no-banner", help="Skip the startup banner."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to record for this invocation."),
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
    task: str = typer.Argument(..., help="The task, or a question to answer about a document."),
    paths: Optional[List[str]] = typer.Argument(None, help="Optional documents (PDF/image) the task is about."),
    force_ask: bool = typer.Option(False, "--ask", help="Force the document fast path, skipping the agent."),
    force_agent: bool = typer.Option(False, "--agent", help="Force the ReAct agent, even for a plain document question."),
    page: int = typer.Option(None, "--page", help="Document fast path: restrict to a single 1-based page."),
    backend: str = typer.Option(None, "--backend", help="Document fast path: force 'text' or 'ocr'. Default: auto."),
    no_annotate: bool = typer.Option(False, "--no-annotate", help="Skip rendering evidence/annotated images."),
    show_json: bool = typer.Option(False, "--json", help="Document fast path: also print the full result as JSON."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Suppress per-tool progress output."),
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

    raise typer.Exit(
        _run_headless(
            task=agent_task,
            config=ctx.obj or load_config(),
            use_team=False,
            show_progress=not no_progress,
        )
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
            config=ctx.obj or load_config(),
            use_team=True,
            no_tester=no_tester,
            show_report=not no_report,
            show_progress=not no_progress,
        )
    )


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
    from agent.runtime.tools import extract_pdf_document, extract_pdf_structured

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


@app.command(name="config")
def show_config(
    ctx: typer.Context,
    show_path: bool = typer.Option(False, "--path", help="Print the config file path and exit."),
):
    """Show the persisted CLI configuration (~/.config/swarn/cli_agent_config.json)."""
    if show_path:
        console.print(str(DEFAULT_CONFIG_PATH))
        return
    note = "" if DEFAULT_CONFIG_PATH.exists() else "  (not created yet — defaults shown)"
    console.print(f"[dim]{DEFAULT_CONFIG_PATH}{note}[/dim]")
    console.print(json.dumps((ctx.obj or load_config()).to_dict(), indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def app_command_names() -> list[str]:
    """The names Typer registered, including explicit `name=` overrides."""
    return [cmd.name or cmd.callback.__name__ for cmd in app.registered_commands]


# Options that take a value: the token AFTER one of these looks positional but
# isn't, so the bare-prompt rewrite has to skip it.
_VALUE_OPTIONS = frozenset({
    "--model", "-m", "--max-iterations", "--port", "-p", "--host",
    "-n", "--limit", "--page", "--backend", "--data", "-d", "--steps", "-s",
    "--time-limit", "-t", "--drafts", "--exec-timeout", "--workers", "-w",
    "--token-budget", "--resume", "--feedback-model",
})


def _rewrite_bare_prompt(argv: list[str]) -> list[str]:
    """`swarn "do X"` → `swarn run "do X"`.

    Only the first positional argument is considered, and only when it isn't
    already a command name — so `swarn run ...`, `swarn --help` and
    `swarn -m X team ...` all keep their normal meaning. A bare `swarn` with
    no positional argument at all is left alone, which is what drops it into
    interactive mode.
    """
    commands = set(app_command_names())
    for i, token in enumerate(argv):
        if token.startswith("-"):
            continue
        if i > 0 and argv[i - 1] in _VALUE_OPTIONS:
            continue
        if token in commands:
            return argv
        return [*argv[:i], "run", *argv[i:]]
    return argv


def main() -> None:
    sys.argv[1:] = _rewrite_bare_prompt(sys.argv[1:])
    app()


if __name__ == "__main__":
    main()
