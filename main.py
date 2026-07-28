"""
Entry point: REPL for the Swarn agent (Phase 1–16).

Usage:
    python main.py

Special commands:
    exit / quit         — stop the REPL and clean up the sandbox
    history [n]         — show last n sessions (default 5)
    recall <id>         — show a past session's tool-call log
    index <path>        — index a directory for semantic search (Phase 3)
    clear               — clear the workspace
    team <task>         — run the task through the Phase 11 multi-agent
                           pipeline (Planner → Coder → Reviewer → Tester)
                           instead of the single agent
    report              — show the markdown report from the most recent
                           'team' run
    guardrails          — show every prompt-injection pattern flagged
                           so far this session (Phase 15)

Or just type any natural-language task to run it through the single
agent (Phases 1–10's combined toolset, one ReAct loop, exactly as before
— 'team' is a separate, additive mode, not a replacement).

Phase 15 note: the guardrail policy (prompt-injection scanning on every
tool result) is ALWAYS on by default — it's a safety layer with near-
zero cost. OpenTelemetry tracing is opt-in: set SWARN_ENABLE_TRACING=1 in
the environment to wrap every LLM/tool call in an OTel span (exported to
the console unless OTEL_EXPORTER_ENDPOINT is also set). It's off by
default because span export adds console noise most users won't want
on every run, and most environments won't have a real OTel collector
configured to send spans to anyway.

Phase 16 note: this REPL is the interactive front end for ad-hoc local
use. For one-off shell-script-friendly invocations (`swarn run "task"`,
exit code 0/1) or the live web dashboard (`swarn serve`), see cli.py —
install this project with `pip install -e .` to get the `swarn` command,
or run `python -m agent.cli --help` without installing anything.
IMPORTANT: a run started here (via this REPL) will NOT appear live in
the Phase 16 dashboard even if it's running — only runs triggered
through the dashboard's own "Run a task" box (or POST /api/run) stream
live, because the dashboard and this REPL are separate OS processes
with separate in-memory session stores. Both still show up in session
history (`history` here, or the dashboard's session list) once
complete — see dashboard.py's module docstring for why.
"""

import atexit
import os
import shutil

from dotenv import load_dotenv

from agent                 import ui
from agent.agent_loop      import AgentLoop
from agent.sandbox         import close_sandbox
from agent.self_correction import SelfCorrectionPolicy
from agent.tools           import WORKSPACE_DIR


def _show_recent_sessions(n: int = 3) -> None:
    """Print a short session history at startup if any sessions exist."""
    from agent.memory import get_session_store
    store = get_session_store()
    if store._index:
        print(f"\nRecent sessions (last {min(n, len(store._index))}):")
        print(store.list_sessions(n=n))
        print()


def main():
    load_dotenv()

    # BYO-LLM was removed: every LLM call is hard-routed to the deployed
    # model endpoint configured in agent/llm/router.py ("PRODUCTION
    # ENDPOINT — CHANGE HERE" banner, or SWARN_DEPLOYED_* env vars).
    # No provider API key is required.
    from agent.llm import DEPLOYED_BASE_URL, DEPLOYED_MODEL_NAME

    # Ensure Docker sandbox is stopped cleanly on any exit
    atexit.register(close_sandbox)

    ui.banner(
        "Swarn Agent — Phases 1–16",
        [
            "file I/O · sandbox · repo-RAG · self-correction",
            "structured memory · session recall",
            "data ingestion+validation · feature engineering · ML training",
            "evaluation+visualization · deployment automation",
            "multi-agent orchestration (type 'team <task>' to use it)",
            "multi-modal RAG · LLM fine-tuning · guardrails+observability",
            "CLI ('swarn' command) + web dashboard ('swarn serve')",
            "",
            f"LLM: {DEPLOYED_MODEL_NAME} @ {DEPLOYED_BASE_URL}",
            f"Workspace: {os.path.relpath(WORKSPACE_DIR)}",
        ],
        footer="Type a task, or 'exit' to quit.",
    )

    # Show session history if there are past runs
    _show_recent_sessions(n=3)

    # Phase 4: correction policy (max 3 consecutive errors before abort)
    policy = SelfCorrectionPolicy(max_consecutive=3)

    # Phase 15: guardrail policy is on by default (near-zero cost, scans
    # tool results for prompt-injection patterns before Claude sees them).
    # Observability (OTel spans) is opt-in via env var — see module
    # docstring above for why it isn't on by default.
    from agent.observability import GuardrailPolicy
    guardrails = GuardrailPolicy()

    observability_hooks = None
    if os.environ.get("SWARN_ENABLE_TRACING") == "1":
        from agent.observability import ObservabilityHooks
        observability_hooks = ObservabilityHooks(
            exporter_endpoint=os.environ.get("OTEL_EXPORTER_ENDPOINT")
        )
        print("[agent] OpenTelemetry tracing enabled "
              f"(exporting to {os.environ.get('OTEL_EXPORTER_ENDPOINT') or 'console'}).\n")

    agent = AgentLoop(
        correction_policy=policy,
        guardrail_policy=guardrails,
        observability_hooks=observability_hooks,
    )

    # Phase 11: holds the markdown report from the most recent 'team' run,
    # so the 'report' command has something to show. None until a team
    # run has actually happened in this process.
    last_team_report: str | None = None

    while True:
        try:
            task = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not task:
            continue

        lower = task.lower()

        if lower in ("exit", "quit"):
            print("[agent] Shutting down…")
            break

        if lower.startswith("history"):
            parts = lower.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            from agent.memory import get_session_store
            print(get_session_store().list_sessions(n=n))
            continue

        if lower.startswith("recall "):
            sid = task[7:].strip()
            from agent.memory import get_session_store
            print(get_session_store().recall_as_text(sid))
            continue

        if lower == "guardrails":
            print(guardrails.summary())
            continue

        if lower.startswith("index "):
            path = task[6:].strip()
            from agent.tools import index_project
            print(index_project(path))
            continue

        if lower == "clear":
            for item in os.listdir(WORKSPACE_DIR):
                if item.startswith("."):
                    continue
                p = os.path.join(WORKSPACE_DIR, item)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            print("[agent] Workspace cleared.")
            continue

        if lower == "report":
            if last_team_report is None:
                print("[agent] No 'team' run has completed yet in this session.")
            else:
                print(last_team_report)
            continue

        if lower.startswith("team "):
            # Phase 11: route through the multi-agent pipeline instead of
            # the single AgentLoop. Imported lazily, same lazy-import
            # convention used everywhere else in this file, so importing
            # main.py for its functions doesn't force-load orchestrator.py.
            # Phase 15: the SAME guardrails/observability instances used
            # by the single agent are passed through here too, so
            # "guardrails" findings and trace spans accumulate across
            # BOTH single-agent and team runs in one process, rather than
            # the team pipeline silently running unguarded.
            team_task = task[5:].strip()
            from agent.orchestrator import Orchestrator
            orchestrator = Orchestrator(
                guardrail_policy=guardrails,
                observability_hooks=observability_hooks,
            )
            result = orchestrator.run(team_task)
            last_team_report = result["report_markdown"]
            print(
                f"\n[agent] Multi-agent run finished: {result['final_outcome']}. "
                f"Type 'report' to see the full timeline."
            )
            continue

        # Reset the correction policy's consecutive counter between tasks
        # (total_corrections accumulates across the session for Phase 5 metrics)
        policy.consecutive_errors = 0

        agent.run(task)


if __name__ == "__main__":
    main()
