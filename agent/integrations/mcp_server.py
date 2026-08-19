"""
Swarn MCP server — expose the platform to ANY MCP client.

heyneo ships `neo-mcp` (pip package) so Claude Code, Cursor, Windsurf, Zed,
Codex CLI etc. can drive its agent through four tools. This module is our
equivalent, running over stdio with the exact same task-lifecycle shape:

  swarn_submit_task   start a task in the background → returns a task id
  swarn_task_status   poll it (running / complete / failed)
  swarn_get_messages  read the transcript + final output
  swarn_list_tasks    see everything submitted this session

Two execution modes, chosen per task:
  mode="solve"  the tree-search engine (needs data_dir) — long ML tasks
  mode="agent"  the ReAct agent with the full Phase 1–15 toolset
  mode="auto"   solve when data_dir is given, agent otherwise

Register with Claude Code:
  claude mcp add swarn -- python -m agent.integrations.mcp_server
Or in any client's mcp.json:
  {"mcpServers": {"swarn": {"command": "python", "args": ["-m", "agent.integrations.mcp_server"]}}}
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from contextlib import redirect_stdout
from dataclasses import dataclass, field

from mcp.server.fastmcp import FastMCP

from agent.llm import DEFAULT_MODEL  # deployed model (see agent/llm/router.py)

mcp = FastMCP("swarn")


@dataclass
class _TaskRecord:
    id: str
    task: str
    mode: str
    status: str = "running"          # running | complete | failed
    result: str = ""
    messages: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float = 0.0


_TASKS: dict[str, _TaskRecord] = {}
_LOCK = threading.Lock()


def _run_task(rec: _TaskRecord, data_dir: str, steps: int, model: str):
    try:
        if rec.mode == "solve":
            from agent.search import SearchConfig, run_search

            kwargs: dict = {"steps": steps, "reflect": True}
            if model:
                kwargs["code_model"] = model
                kwargs["feedback_model"] = model
            cfg = SearchConfig(**kwargs)

            def on_step(node, journal):
                status = ("BUGGY" if node.is_buggy
                          else f"metric={node.metric}")
                rec.messages.append(
                    f"node {node.step + 1} [{node.stage}] → {status}: "
                    f"{(node.analysis or node.plan)[:200]}")

            result = run_search(rec.task, data_dir=data_dir or None,
                                config=cfg, on_step=on_step)
            if result.best:
                rec.result = (f"Best metric: {result.best.metric}\n"
                              f"Solution: {result.solution_path}\n"
                              f"Report: {result.report_path}")
            else:
                rec.result = f"No working solution. Report: {result.report_path}"
        else:
            from agent.core.agent_loop import AgentLoop
            from agent.core.self_correction import SelfCorrectionPolicy
            from agent.observability.observability import GuardrailPolicy

            # model arg is display-only — calls hard-route to the deployed
            # endpoint configured in agent/llm/router.py
            loop = AgentLoop(model=model or DEFAULT_MODEL,
                             correction_policy=SelfCorrectionPolicy(),
                             guardrail_policy=GuardrailPolicy())
            buf = io.StringIO()
            with redirect_stdout(buf):
                outcome = loop.run(rec.task)
            transcript = buf.getvalue()
            rec.messages.extend(transcript.splitlines()[-200:])
            rec.result = (f"Outcome: {outcome['outcome']}\n"
                          f"Summary: {outcome.get('summary') or '(none)'}")
        rec.status = "complete"
    except Exception as e:  # noqa: BLE001
        rec.status = "failed"
        rec.result = f"{type(e).__name__}: {e}"
    finally:
        rec.finished = time.time()


@mcp.tool()
def swarn_submit_task(task: str, data_dir: str = "", steps: int = 12,
                    mode: str = "auto", model: str = "") -> str:
    """Submit an ML/AI engineering task to Swarn. Runs in the background;
    returns a task_id to poll with swarn_task_status.

    Args:
        task: the goal in natural language (include target + metric for ML tasks)
        data_dir: absolute path to the data directory (enables tree-search mode)
        steps: search budget in solution attempts (solve mode)
        mode: "solve" (tree search), "agent" (ReAct toolset), or "auto"
        model: optional model spec, e.g. "openai:gpt-4o" or "ollama:llama3.1"
    """
    if mode == "auto":
        mode = "solve" if data_dir else "agent"
    rec = _TaskRecord(id=uuid.uuid4().hex[:12], task=task, mode=mode)
    with _LOCK:
        _TASKS[rec.id] = rec
    threading.Thread(target=_run_task, args=(rec, data_dir, steps, model),
                     daemon=True).start()
    return (f"Task {rec.id} submitted (mode={mode}). "
            f"Poll with swarn_task_status(task_id=\"{rec.id}\").")


@mcp.tool()
def swarn_task_status(task_id: str) -> str:
    """Check the status of a submitted Swarn task."""
    rec = _TASKS.get(task_id)
    if not rec:
        return f"Unknown task id: {task_id}"
    elapsed = (rec.finished or time.time()) - rec.started
    lines = [f"Task {rec.id}: {rec.status} ({elapsed:.0f}s, mode={rec.mode})"]
    if rec.messages:
        lines.append(f"Latest: {rec.messages[-1]}")
    if rec.status != "running":
        lines.append(rec.result)
    return "\n".join(lines)


@mcp.tool()
def swarn_get_messages(task_id: str) -> str:
    """Read the full progress transcript and final output of a Swarn task."""
    rec = _TASKS.get(task_id)
    if not rec:
        return f"Unknown task id: {task_id}"
    out = [f"# Task {rec.id} ({rec.status})", rec.task, ""]
    out.extend(rec.messages[-100:])
    if rec.result:
        out.extend(["", "# Result", rec.result])
    return "\n".join(out)


@mcp.tool()
def swarn_list_tasks() -> str:
    """List every Swarn task submitted in this session."""
    if not _TASKS:
        return "No tasks submitted yet."
    lines = []
    for rec in _TASKS.values():
        lines.append(f"{rec.id} · {rec.status} · [{rec.mode}] {rec.task[:80]}")
    return "\n".join(lines)


def main():
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
