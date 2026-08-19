"""
Terminal UI layer (Rich) — the one module that owns console output for
agent runs.

Before this existed, agent_loop.py / orchestrator.py / cli.py each did
raw print()s: unstyled JSON tool inputs, 400-char result dumps, and
guardrail warnings all ran together into one undifferentiated wall of
text. Every display concern now routes through here so the visual
language is consistent everywhere the agent renders to a terminal:

  • agent reasoning   → bordered panel, markdown-rendered
  • tool call         → single dim "→ name {args…}" line
  • tool result       → dim, indented, hard-truncated
  • warnings          → one ⚠ line per kind (guardrail / doom-loop /
                        correction), colored so they can't be missed
  • session/pipeline  → rules (horizontal dividers) with accent colors
  • final outcome     → green/red panel with the summary inside

Rich degrades gracefully when stdout is not a TTY (piped to a file, CI
log): colors and boxes are dropped automatically, and the NO_COLOR env
var is respected — so `swarn run … > log.txt` stays script-friendly.

Role accent colors are stable per role so interleaved multi-agent
output (Phase 11) is scannable at a glance.
"""

import json

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# highlight=False: Rich's auto-highlighting colors every number/string
# in tool results, which is exactly the kind of noise this module exists
# to remove.
console = Console(highlight=False)

_ROLE_STYLES = {
    "planner":  "bright_cyan",
    "coder":    "bright_green",
    "reviewer": "bright_magenta",
    "tester":   "bright_yellow",
}
_DEFAULT_ACCENT = "bright_blue"

_TOOL_ARGS_PREVIEW = 120
_RESULT_PREVIEW    = 400


def _accent(role: str | None) -> str:
    return _ROLE_STYLES.get((role or "").lower(), _DEFAULT_ACCENT)


def _title(role: str | None, label: str) -> str:
    return f"{role} · {label}" if role else label


# ────────────────────────────────────────────────── agent-loop events

def session_header(session_id: str, task: str, role: str | None = None) -> None:
    accent = _accent(role)
    console.print()
    console.print(Rule(Text(_title(role, f"session {session_id[:8]}"), style=f"bold {accent}"),
                       style=accent, align="left"))
    console.print(Text(task if len(task) <= 200 else task[:200] + "…", style="italic dim"))


def agent_text(text: str, role: str | None = None) -> None:
    accent = _accent(role)
    console.print(Panel(
        Markdown(text),
        title=Text(_title(role, "agent"), style=accent),
        title_align="left",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def tool_call(name: str, tool_input: dict, role: str | None = None) -> None:
    try:
        args = json.dumps(tool_input, ensure_ascii=False)
    except (TypeError, ValueError):
        args = str(tool_input)
    if len(args) > _TOOL_ARGS_PREVIEW:
        args = args[:_TOOL_ARGS_PREVIEW] + "…"
    line = Text()
    line.append("→ ", style="dim")
    line.append(name, style=f"bold {_accent(role)}")
    line.append(f"  {args}", style="dim")
    console.print(Padding(line, (0, 0, 0, 1)))


def tool_result(result: str, limit: int = _RESULT_PREVIEW) -> None:
    result = str(result)
    body = result[:limit].rstrip()
    if len(result) > limit:
        body += f" … (+{len(result) - limit:,} chars)"
    console.print(Padding(Text(body, style="dim"), (0, 0, 0, 3)))


def warn(kind: str, message: str) -> None:
    """One visual grammar for every mid-run alert (guardrail hit,
    doom-loop trip, self-correction attempt): a single ⚠ line, colored
    so it separates from the dim tool output around it."""
    line = Text()
    line.append("⚠ ", style="bold yellow")
    line.append(kind, style="bold black on yellow")
    line.append(f" {message}", style="yellow")
    console.print(Padding(line, (0, 0, 0, 1)))


def error(message: str) -> None:
    console.print(Padding(Text(f"⛔ {message}", style="bold red"), (0, 0, 0, 1)))


def info(message: str) -> None:
    console.print(Padding(Text(f"· {message}", style="dim"), (0, 0, 0, 1)))


def outcome(outcome: str, session_id: str, summary: str | None = None) -> None:
    ok = outcome == "complete"
    style = "green" if ok else "red"
    mark = "✔" if ok else "✘"
    console.print(Panel(
        Markdown(summary) if summary else Text(outcome, style=style),
        title=Text(f"{mark} {outcome} · session {session_id[:8]}", style=f"bold {style}"),
        title_align="left",
        border_style=style,
        box=box.ROUNDED,
        padding=(0, 1),
    ))


# ────────────────────────────────────────────── orchestrator / cli / repl

def section(title: str, subtitle: str | None = None, style: str = _DEFAULT_ACCENT) -> None:
    """A horizontal rule with a title — pipeline start/finish markers."""
    console.print()
    console.print(Rule(Text(title, style=f"bold {style}"), style=style))
    if subtitle:
        console.print(Text(subtitle if len(subtitle) <= 200 else subtitle[:200] + "…",
                           style="italic dim"))


def markdown(md: str) -> None:
    console.print(Markdown(md))


def banner(title: str, lines: list[str], footer: str | None = None) -> None:
    body = Text()
    for i, line in enumerate(lines):
        if i:
            body.append("\n")
        body.append(line, style="dim")
    if footer:
        body.append("\n\n" if lines else "")
        body.append(footer, style="italic")
    console.print(Panel(
        body,
        title=Text(title, style=f"bold {_DEFAULT_ACCENT}"),
        title_align="left",
        border_style=_DEFAULT_ACCENT,
        box=box.ROUNDED,
        padding=(0, 1),
    ))
