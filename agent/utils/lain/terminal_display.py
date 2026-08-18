"""
Lain theme display — hot pink on black, NAVI braille art, Wired protocol UI.

Same public surface as `agent.utils.classic.terminal_display`, so
`agent.utils.terminal_display` can swap between the two without any caller
noticing. Everything visual is driven by theme.yaml (see `theme.py`);
structural pieces that carry no styling (the sub-agent live region, the
headless run manager, plan formatting) are reused from the classic module
rather than copied.
"""

from __future__ import annotations

import asyncio
import os
import random
import time

from rich.console import Console
from rich.markup import escape
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from agent.utils.classic.terminal_display import (
    HeadlessDisplayManager,
    SubAgentDisplayManager,
    _truncate,
    format_plan_display,
    format_plan_tool_output,
)
from agent.utils.lain import theme as T

__all__ = [
    "HELP_ROWS",
    "HeadlessDisplayManager",
    "format_help_text",
    "format_plan_display",
    "format_plan_tool_output",
    "get_console",
    "get_headless_display",
    "print_approval_header",
    "print_approval_item",
    "print_banner",
    "print_compacted",
    "print_error",
    "print_help",
    "print_init_done",
    "print_interrupted",
    "print_markdown",
    "print_plan",
    "print_tool_call",
    "print_tool_log",
    "print_tool_output",
    "print_turn_complete",
    "print_yolo_approve",
    "reset_headless_display",
    "set_stream_enabled",
    "stream_enabled",
]


_ACCENT = T.color("ui_accent")
_TITLE = T.color("banner_title")
_DIM = T.color("banner_dim")
_TEXT = T.color("banner_text")
_LABEL = T.color("ui_label")
_OK = T.color("ui_ok")
_ERR = T.color("ui_error")
_BORDER = T.color("banner_border")

_THEME = Theme(
    {
        "tool.name": f"bold {_ACCENT}",
        "tool.args": "dim",
        "tool.ok": f"dim {_OK}",
        "tool.fail": f"dim {_ERR}",
        "info": "dim",
        "muted": f"{_DIM}",
        "markdown.strong": f"bold {_TITLE}",
        "markdown.emphasis": f"italic {_TEXT}",
        "markdown.code": _ACCENT,
        "markdown.code_block": _ACCENT,
        "markdown.link": f"underline {_TITLE}",
        "markdown.h1": f"bold {_ACCENT}",
        "markdown.h2": f"bold {_TITLE}",
        "markdown.h3": f"bold {_DIM}",
    }
)

_console = Console(theme=_THEME, highlight=False)

# Indent prefix for all agent output (aligns under the prompt symbol)
_I = "  "

_stream_enabled = True


def indent() -> str:
    """Left margin for agent output. The classic theme centers its column;
    this theme stays flush-left, but callers share one accessor."""
    return _I


def get_console() -> Console:
    return _console


def set_stream_enabled(enabled: bool) -> None:
    global _stream_enabled
    _stream_enabled = enabled


def stream_enabled() -> bool:
    return _stream_enabled and _console.is_terminal


# ── Banner ─────────────────────────────────────────────────────────────

_TOOLS_PLACEHOLDER = "Tools: loading..."
_HINT = "/help for commands · /model to switch · /quit to exit"

# Rows between the tools placeholder and the cursor once print_banner() has
# finished; None means the row arithmetic can't be trusted (piped output).
_tools_row_offset: int | None = None
_TOOLS_INDEX = 4

# Left padding the banner's info block was centered with. print_init_done()
# reuses it so the retyped "Tools: N loaded" lands in the same column.
_info_pad = "  "


def _banner_lines(model_label: str, user_label: str, runtime_label: str) -> list[tuple[str, str]]:
    """Banner body as (text, rich_style) pairs. Index 4 is the placeholder."""
    return [
        ("Connecting to the World...", f"bold {_ACCENT}"),
        (f"User: {user_label}", _LABEL),
        (f"Model: {model_label}", _LABEL),
        (f"Tool runtime: {runtime_label}", _LABEL),
        (_TOOLS_PLACEHOLDER, _LABEL),
        ("", ""),
        (_HINT, _DIM),
    ]


def _terminal_speaks_braille() -> bool:
    """Whether the console encoding can represent the NAVI braille art.

    Probed up front: once Rich fails to encode a frame the un-writable text is
    still in its buffer and the *next* print raises too.
    """
    try:
        "⠀⣿◈".encode(_console.file.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError, AttributeError):
        return False
    return True


def _art_enabled() -> bool:
    return (
        _console.is_terminal
        and os.environ.get("SWARN_NO_BOOT_ANIM") != "1"
        and _terminal_speaks_braille()
    )


def _cell_len(markup: str) -> int:
    """Visible width of a markup string, in terminal cells.

    Braille and the box-drawing glyphs are full-width-sensitive, so measure
    with Rich rather than len().
    """
    return Text.from_markup(markup).cell_len


def _block_pad(markup_lines: list[str]) -> str:
    """Left padding that centers a block of lines *as a block*.

    Centering each line independently would shear ASCII art apart — every row
    must shift by the same amount, so the offset comes from the widest line.
    """
    if not markup_lines:
        return ""
    block_width = max(_cell_len(line) for line in markup_lines)
    return " " * max(0, (_console.width - block_width) // 2)


def _print_centered_block(markup_lines: list[str]) -> None:
    pad = _block_pad(markup_lines)
    for line in markup_lines:
        _console.print(f"{pad}{line}" if line else "")


def print_banner(
    model: str | None = None,
    hf_user: str | None = None,
    tool_runtime: str | None = None,
) -> None:
    """Print the Lain boot screen: logo, NAVI hero art, then the info block.

    Everything is centered horizontally against the terminal width, each group
    as its own block so the art and the info column each stay internally
    aligned.
    """
    global _tools_row_offset

    lines = _banner_lines(
        model_label=model or "unknown",
        user_label=hf_user or "not logged in",
        runtime_label=tool_runtime or "local filesystem",
    )

    _console.print()
    if _art_enabled():
        try:
            _print_centered_block(T.banner_logo_lines())
            _console.print()
        except Exception:  # noqa: BLE001 — decoration must never block startup
            pass

    title = T.branding("agent_name", "ML Engineer")
    rule = "─" * max(8, len(title))
    info_block = [f"[bold {_TITLE}]{escape(title)}[/]", f"[{_DIM}]{rule}[/]"] + [
        f"[{style}]{escape(text)}[/]" if style else escape(text)
        for text, style in lines
    ]
    # One pad for the whole block: the "Tools: loading..." row must sit at the
    # same column as the rest, since print_init_done() retypes it in place.
    pad = _block_pad(info_block)
    global _info_pad
    _info_pad = pad
    for line in info_block:
        _console.print(f"{pad}{line}" if _cell_len(line) else "")
    _console.print()

    welcome = T.branding("welcome")
    if welcome:
        _print_centered_block([welcome])
    _console.print()

    if not _console.is_terminal:
        _tools_row_offset = None
        return
    # Remaining banner lines, the two trailing blanks + welcome line, plus one
    # to land on the cursor's own row.
    trailing = 3 if welcome else 2
    _tools_row_offset = (len(lines) - 1 - _TOOLS_INDEX) + trailing + 1


# ── Init progress ──────────────────────────────────────────────────────


def _retype_in_place(text: str, rows_up: int, style: str) -> None:
    """Overwrite a line `rows_up` rows above the cursor, then come back down."""
    f = _console.file
    f.write(f"\033[{rows_up}A\r\033[2K")
    f.flush()

    if stream_enabled():
        _console.print(_info_pad, end="")
        for ch in text:
            _console.print(escape(ch), style=style, end="")
            f.flush()
            time.sleep(0.012)
    else:
        _console.print(f"{_info_pad}{escape(text)}", style=style, end="")

    f.write(f"\033[{rows_up}B\r")
    f.flush()


def print_init_done(tool_count: int = 0) -> None:
    global _tools_row_offset

    label = f"Tools: {tool_count} loaded"

    if _tools_row_offset is not None:
        _retype_in_place(label, _tools_row_offset, style=_LABEL)
        _tools_row_offset = None
    else:
        _console.print(f"{_info_pad}[{_LABEL}]{label}[/]")


# ── Tool calls ─────────────────────────────────────────────────────────


def _hex_to_ansi(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m"


_ANSI_ACCENT = _hex_to_ansi(_ACCENT)
_ANSI_RESET = "\033[0m"


def print_tool_call(tool_name: str, args_preview: str) -> None:
    glyph = T.tool_emoji(tool_name)
    prefix = f"{T.tool_prefix()} {glyph}"
    f = _console.file
    if not stream_enabled():
        _console.print(
            f"{_I}[tool.name]{escape(prefix)} {tool_name}[/tool.name]  [tool.args]{args_preview}[/tool.args]"
        )
        return
    f.write(f"{_I}{_ANSI_ACCENT}{prefix} ")
    for ch in tool_name:
        f.write(ch)
        f.flush()
        time.sleep(0.015)
    f.write(f"{_ANSI_RESET}  \033[2m{args_preview}{_ANSI_RESET}\n")
    f.flush()


def print_tool_output(output: str, success: bool, truncate: bool = True) -> None:
    if truncate:
        output = _truncate(output, max_lines=10)
    style = "tool.ok" if success else "tool.fail"
    indented = "\n".join(f"{_I}  {line}" for line in output.split("\n"))
    _console.print(f"[{style}]{indented}[/{style}]")


_subagent_display = SubAgentDisplayManager()


def print_tool_log(tool: str, log: str, agent_id: str = "", label: str = "") -> None:
    """Handle tool log events — sub-agent calls get the rolling display."""
    if tool == "research":
        aid = agent_id or "research"
        if log == "Starting research sub-agent...":
            _subagent_display.start(aid, label or "research")
        elif log == "Research complete.":
            _subagent_display.clear(aid)
        elif log.startswith("tokens:"):
            _subagent_display.set_tokens(aid, int(log[7:]))
        elif log.startswith("tools:"):
            _subagent_display.set_tool_count(aid, int(log[6:]))
        else:
            _subagent_display.add_call(aid, log)
    else:
        _console.print(f"{_I}[{_DIM}]{tool}: {log}[/]")


# ── Messages ───────────────────────────────────────────────────────────


async def print_markdown(
    text: str,
    cancel_event: "asyncio.Event | None" = None,
    instant: bool = False,
) -> None:
    import io

    from rich.padding import Padding

    _console.print()

    # Render to a buffer so the text can be typed out. StringIO is not a TTY,
    # so force terminal rendering to keep the ANSI style codes.
    buf = io.StringIO()
    buf_console = Console(
        file=buf,
        width=_console.width,
        highlight=False,
        theme=_THEME,
        force_terminal=True,
        color_system=_console.color_system or "truecolor",
    )
    buf_console.print(Padding(Markdown(text), (0, 0, 0, 2)))
    rendered = "\n".join(line.rstrip() for line in buf.getvalue().split("\n"))

    f = _console.file

    label = T.branding("response_label")
    if label:
        _console.print(f"{_I}[bold {_TITLE}]{escape(label.strip())}[/]")

    if instant or not stream_enabled():
        f.write(rendered)
        f.write("\n")
        f.flush()
        return

    rng = random.Random(42)
    cancelled = False
    for ch in rendered:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        f.write(ch)
        f.flush()
        if ch in ("\n", " "):
            await asyncio.sleep(0.002)
        elif rng.random() < 0.03:
            await asyncio.sleep(0.015)
        else:
            await asyncio.sleep(0.004)
    f.write("\033[0m\n" if cancelled else "\n")
    f.flush()


def print_error(message: str) -> None:
    _console.print(f"\n{_I}[bold {_ERR}]Error:[/] {message}")


def print_turn_complete() -> None:
    pass  # no separator — clean output


def print_interrupted() -> None:
    _console.print(f"\n{_I}[{_DIM} italic]signal lost[/]")


def print_compacted(old_tokens: int, new_tokens: int) -> None:
    _console.print(
        f"{_I}[{_DIM}]layer compacted: {old_tokens:,} → {new_tokens:,} tokens[/]"
    )


# ── Approval ───────────────────────────────────────────────────────────


def print_approval_header(count: int) -> None:
    label = f"Approval required — {count} item{'s' if count != 1 else ''}"
    _console.print()
    _console.print(
        f"{_I}",
        Panel(f"[bold {_TITLE}]{label}[/]", border_style=_BORDER, expand=False),
    )


def print_approval_item(index: int, total: int, tool_name: str, operation: str) -> None:
    _console.print(
        f"\n{_I}[bold]\\[{index}/{total}][/bold]  [tool.name]{tool_name}[/tool.name]  {operation}"
    )


def print_yolo_approve(count: int) -> None:
    _console.print(f"{_I}[bold {_TITLE}]yolo →[/] auto-approved {count} item(s)")


# ── Help ───────────────────────────────────────────────────────────────

from agent.utils.classic.terminal_display import HELP_ROWS  # noqa: E402


def _help_column_widths(rows) -> tuple[int, int]:
    return (
        max(len(command) for command, _, _ in rows),
        max(len(args) for _, args, _ in rows),
    )


def _format_help_row(command, args, description, command_width, args_width) -> str:
    command_gap = " " * (command_width - len(command) + 2)
    args_gap = " " * (args_width - len(args) + 2)
    command_markup = f"[{_ACCENT}]{escape(command)}[/]"
    args_markup = f"[muted]{escape(args)}[/muted]" if args else ""
    return f"{_I}  {command_markup}{command_gap}{args_markup}{args_gap}{description}"


def format_help_text(rows=None) -> str:
    help_rows = HELP_ROWS if rows is None else rows
    command_width, args_width = _help_column_widths(help_rows)
    header = T.branding("help_header", "Available Protocols")
    return "\n".join(
        [f"{_I}[bold {_TITLE}]{escape(header)}[/]"]
        + [
            _format_help_row(command, args, description, command_width, args_width)
            for command, args, description in help_rows
        ]
    )


def print_help() -> None:
    _console.print()
    _console.print(format_help_text())
    _console.print()


# ── Plan display ───────────────────────────────────────────────────────


def print_plan() -> None:
    plan_str = format_plan_display()
    if plan_str:
        _console.print(plan_str)


# ── Headless mode display ──────────────────────────────────────────────

_headless_display: HeadlessDisplayManager | None = None


def get_headless_display(
    console: Console | None = None, show_progress: bool = True
) -> HeadlessDisplayManager:
    """Get or create the global headless display manager (Lain-themed console)."""
    global _headless_display
    if _headless_display is None:
        _headless_display = HeadlessDisplayManager(
            console=console or _console, show_progress=show_progress
        )
    return _headless_display


def reset_headless_display() -> None:
    global _headless_display
    _headless_display = None
