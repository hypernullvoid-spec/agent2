"""
Terminal display utilities — rich-powered CLI formatting.
"""

import asyncio
import os
import re
import random
import threading
import time

from contextlib import nullcontext

from rich._spinners import SPINNERS
from rich.console import Console
from rich.markup import escape
from rich.markdown import Heading, Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from agent.utils.classic.crt_boot import run_boot_sequence
from agent.utils.classic.particle_logo import run_particle_logo


class _LeftHeading(Heading):
    """Rich's default Markdown renders h1/h2 centered via Align.center.
    Yield the styled text directly so headings stay left-aligned."""

    def __rich_console__(self, console, options):
        self.text.justify = "left"
        yield self.text


Markdown.elements["heading_open"] = _LeftHeading


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _clip_to_width(s: str, width: int) -> str:
    """Truncate a string to `width` visible columns, preserving ANSI styles.

    Needed for the sub-agent live redraw: cursor-up-and-erase assumes one
    logical line == one terminal row. If a line wraps, cursor-up undershoots
    and the next redraw corrupts the display. Truncating prevents wrap.
    """
    if width <= 0:
        return s
    out: list[str] = []
    visible = 0
    i = 0
    # Reserve 1 char for the trailing ellipsis
    limit = width - 1
    truncated = False
    while i < len(s):
        m = _ANSI_RE.match(s, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        if visible >= limit:
            truncated = True
            break
        out.append(s[i])
        visible += 1
        i += 1
    if truncated:
        # Strip styles (so ellipsis isn't left hanging inside a style run)
        out.append("\033[0m…")
    return "".join(out)


_THEME = Theme(
    {
        "tool.name": "bold rgb(255,200,80)",
        "tool.args": "dim",
        "tool.ok": "dim green",
        "tool.fail": "dim red",
        "info": "dim",
        "muted": "dim",
        # Markdown emphasis colors
        "markdown.strong": "bold rgb(255,200,80)",
        "markdown.emphasis": "italic rgb(180,140,40)",
        "markdown.code": "rgb(120,220,255)",
        "markdown.code_block": "rgb(120,220,255)",
        "markdown.link": "underline rgb(90,180,255)",
        "markdown.h1": "bold rgb(255,200,80)",
        "markdown.h2": "bold rgb(240,180,95)",
        "markdown.h3": "bold rgb(220,165,100)",
    }
)

_console = Console(theme=_THEME, highlight=False)

# The UI renders as a fixed-width column centered in the terminal rather than
# flush-left. _MAX_CONTENT_WIDTH caps how wide that column grows on a very wide
# terminal — past that, prose gets hard to read and the centering is lost.
_MAX_CONTENT_WIDTH = 90
_MIN_CONTENT_WIDTH = 20


def content_width() -> int:
    """Width of the centered content column, in columns."""
    return max(_MIN_CONTENT_WIDTH, min(_MAX_CONTENT_WIDTH, _console.width))


def indent() -> str:
    """Left margin that centers the content column in the terminal.

    Recomputed on every call so a mid-session resize is picked up. Every piece
    of agent output is prefixed with this, so the whole UI moves together.
    """
    return " " * max(0, (_console.width - content_width()) // 2)


def _i() -> str:
    return indent()


def get_console() -> Console:
    return _console


# ── Prompt placeholder ─────────────────────────────────────────────────

# Example tasks shown dimmed above the input line. console.input() gives no
# keypress hook, so a true vanishing ghost inside the line isn't possible
# without a raw-mode editor; a hint line above the prompt is the honest
# equivalent and never collides with what the user types.
_PLACEHOLDERS = [
    "analyze sales.csv and show me what stands out",
    "train a model to predict churn from customers.csv",
    "extract the tables from this PDF into a CSV",
    "explain what agent/core/agent_loop.py does",
    "clean the missing values in data/raw.csv",
    "plot revenue by month and save the chart",
    "find every place we call the LLM directly",
    "compare two models and tell me which to ship",
]


# One session for the whole REPL, so in-memory history (↑/↓) carries
# across turns instead of resetting on every prompt.
_pt_session = None


def prompt_placeholder() -> str:
    """A random example task for the hint line above the prompt."""
    return random.choice(_PLACEHOLDERS)


def read_user_input(prompt: str = "> ") -> str:
    """Read one line with ghost placeholder text inside the input line.

    prompt_toolkit draws a dim example task where the cursor sits and
    clears it the moment a key is pressed, while keeping real line editing
    (backspace, arrows, history). Falls back to Rich's plain input when
    prompt_toolkit is unavailable or stdout isn't a terminal (piped input,
    CI), where a placeholder would be meaningless anyway.
    """
    if not _console.is_terminal:
        return _console.input(prompt)
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import FormattedText
    except ImportError:
        return _console.input(prompt)

    global _pt_session
    if _pt_session is None:
        _pt_session = PromptSession()
    return _pt_session.prompt(
        FormattedText([("bold ansicyan", prompt)]),
        placeholder=FormattedText([("ansibrightblack", prompt_placeholder())]),
    )


# ── Thinking status ────────────────────────────────────────────────────

_THINK_COLOR = "rgb(255,200,80)"

# Frames are fixed-width and end with a single trailing space so the verb
# sits right beside the dots instead of a ragged gap.
SPINNERS["swarn.classic"] = {
    "interval": 110,
    "frames": ["·   ", "•   ", "●   ", "●·  ", "●•· ", "●●● ", " ●● ", "  ● ", "    "],
}

_THINK_VERBS = [
    "Thinking",
    "Reasoning",
    "Working it out",
    "Considering",
    "Planning",
    "Digging in",
    "Weighing options",
    "Piecing it together",
    "Reading the room",
    "Crunching",
]

# How long one verb stays on screen before the next is drawn.
_THINK_VERB_SECONDS = 1.5


class _RotatingStatus:
    """Rich's status text is fixed for the life of the context, so a long
    wait would sit on one phrase. This wraps it in a daemon thread that
    swaps in a new random verb every few seconds."""

    def __init__(self, verbs: list[str], period: float):
        self._verbs  = verbs
        self._period = period
        self._status = _console.status(self._next_text(), spinner="swarn.classic",
                                       spinner_style=_THINK_COLOR)
        self._stop   = threading.Event()
        self._thread = None

    def _next_text(self) -> Text:
        return Text(f"{random.choice(self._verbs)}…", style=_THINK_COLOR)

    def _rotate(self) -> None:
        while not self._stop.wait(self._period):
            try:
                self._status.update(status=self._next_text())
            except Exception:  # noqa: BLE001 — display must never kill a run
                return

    def __enter__(self):
        self._status.__enter__()
        if len(self._verbs) > 1:
            self._thread = threading.Thread(target=self._rotate, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        return self._status.__exit__(*exc)


def thinking_status(label: str | None = None):
    """Animated status for the stretch where the agent is waiting on the
    model. The verb is drawn at random and re-drawn every
    `_THINK_VERB_SECONDS`, so even one long call cycles through phrases
    instead of freezing on one. No-op when stdout isn't a TTY."""
    if not _console.is_terminal:
        return nullcontext()
    if label:
        return _console.status(Text(f"{label}…", style=_THINK_COLOR),
                               spinner="swarn.classic", spinner_style=_THINK_COLOR)
    return _RotatingStatus(_THINK_VERBS, _THINK_VERB_SECONDS)


# Streaming (the CRT typewriter effect) is on by default and turned off by
# `--no-stream`, or automatically when stdout isn't a terminal — typing text
# out one character at a time into a pipe or a CI log is pure waste.
_stream_enabled = True


def set_stream_enabled(enabled: bool) -> None:
    global _stream_enabled
    _stream_enabled = enabled


def stream_enabled() -> bool:
    return _stream_enabled and _console.is_terminal


# ── Banner ─────────────────────────────────────────────────────────────


_GOLD = "rgb(255,200,80)"
_TOOLS_PLACEHOLDER = "Tools: loading..."
_HINT = "/help for commands · /model to switch · /quit to exit"

# How many rows above the cursor the "Tools: loading..." placeholder sits once
# print_banner() has finished. print_init_done() walks back up by this many
# rows to overwrite it in place. None means "just append" — either the banner
# never ran, stdout isn't a terminal, or something else has printed since and
# the row arithmetic can no longer be trusted.
_tools_row_offset: int | None = None


def _banner_lines(model_label: str, user_label: str, runtime_label: str) -> list[tuple[str, str]]:
    """The banner body as (text, rich_style) pairs — the shape crt_boot wants.

    Index 4 is the tools placeholder; _TOOLS_INDEX below must track it.
    """
    return [
        ("Initializing agent runtime...", f"bold {_GOLD}"),
        (f"User: {user_label}", "dim"),
        (f"Model: {model_label}", "dim"),
        (f"Tool runtime: {runtime_label}", "dim"),
        (_TOOLS_PLACEHOLDER, "dim"),
        ("", ""),
        (_HINT, _GOLD),
    ]


_TOOLS_INDEX = 4


def _terminal_speaks_braille() -> bool:
    """Whether the console encoding can represent the logo's braille glyphs.

    Probed up front rather than caught after the fact: once Rich fails to
    encode a frame, the un-writable text is still sitting in its buffer and
    the *next* print raises too, taking the static fallback down with it.
    """
    try:
        "⠀⣿".encode(_console.file.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError, AttributeError):
        return False
    return True


def _boot_animation_enabled() -> bool:
    """The particle logo and CRT typing are ~7s of pure decoration. Skip them
    when nobody is watching a real terminal, when the terminal can't render
    the glyphs, or when asked to."""
    return (
        _console.is_terminal
        and os.environ.get("SWARN_NO_BOOT_ANIM") != "1"
        and _terminal_speaks_braille()
    )


def _wipe_screen() -> None:
    _console.file.write("\033[2J\033[H")
    _console.file.flush()


def _print_banner_static(lines: list[tuple[str, str]]) -> int | None:
    """Print the banner with no animation. Returns the tools-row offset, or
    None when the cursor can't be moved (piped output, CI log)."""
    _console.print()
    for text, style in lines:
        _console.print(f"{_i()}{escape(text)}" if text else "", style=style or None)
    _console.print()
    if not _console.is_terminal:
        return None
    # Rows below the placeholder: the remaining lines, plus the trailing blank,
    # plus one to land on the cursor's own row.
    return (len(lines) - 1 - _TOOLS_INDEX) + 2


def print_banner(
    model: str | None = None,
    hf_user: str | None = None,
    tool_runtime: str | None = None,
) -> None:
    """Boot the CLI: particle logo, screen wipe, then the CRT-typed banner.

    Falls back to a plain static banner when the animation is disabled or
    interrupted, so a pipe, a CI log or an impatient Ctrl+C all still get the
    same information — just instantly.
    """
    global _tools_row_offset

    lines = _banner_lines(
        model_label=model or "unknown",
        user_label=hf_user or "not logged in",
        runtime_label=tool_runtime or "local filesystem",
    )

    if not _boot_animation_enabled():
        _tools_row_offset = _print_banner_static(lines)
        return

    try:
        run_particle_logo(_console, converge_seconds=1.5, hold_seconds=2.0)
        _wipe_screen()
        run_boot_sequence(_console, lines, indent=_i())
    except KeyboardInterrupt:
        # Ctrl+C through the animation means "get on with it", not "quit".
        _wipe_screen()
        _tools_row_offset = _print_banner_static(lines)
        return
    except Exception:  # noqa: BLE001 — decoration must never block startup
        # Most likely a legacy console that can't encode braille (cp1252
        # raises UnicodeEncodeError on U+2800+), or a terminal too small for
        # the canvas. Either way: show the plain banner and carry on.
        _wipe_screen()
        _tools_row_offset = _print_banner_static(lines)
        return

    # run_boot_sequence's final frame ends every line with "\n" and then
    # console.print adds one more, so the cursor sits two rows below the last
    # banner line.
    _tools_row_offset = (len(lines) - 1 - _TOOLS_INDEX) + 2


# ── Init progress ──────────────────────────────────────────────────────


def _retype_in_place(text: str, rows_up: int, style: str) -> None:
    """Overwrite a line `rows_up` rows above the cursor, then come back down.

    Used to turn the banner's "Tools: loading..." placeholder into the real
    count without reprinting the whole banner.
    """
    f = _console.file
    f.write(f"\033[{rows_up}A\r\033[2K")
    f.flush()

    if stream_enabled():
        # Retype it character by character, matching the CRT boot cadence.
        # Written raw: Rich trims a whitespace-only print, which would drop the
        # centering margin entirely.
        f.write(_i())
        for ch in text:
            _console.print(escape(ch), style=style, end="")
            f.flush()
            time.sleep(0.012)
    else:
        _console.print(f"{_i()}{escape(text)}", style=style, end="")

    f.write(f"\033[{rows_up}B\r")
    f.flush()


def print_init_done(tool_count: int = 0) -> None:
    global _tools_row_offset

    label = f"Tools: {tool_count} loaded"

    if _tools_row_offset is not None:
        _retype_in_place(label, _tools_row_offset, style="dim")
        _tools_row_offset = None
    else:
        _console.print(f"{_i()}[dim]{label}[/dim]")

    _console.print()
    _console.print(f"{_i()}[{_GOLD}]Ready. Let's build something impressive.[/{_GOLD}]")
    _console.print()


# ── Tool calls ─────────────────────────────────────────────────────────


def print_tool_call(tool_name: str, args_preview: str) -> None:
    import time

    f = _console.file
    # CRT-style: type out tool name in HF yellow
    gold = "\033[38;2;255;200;80m"
    reset = "\033[0m"
    if not stream_enabled():
        _console.print(f"{_i()}[tool.name]▸ {tool_name}[/tool.name]  [tool.args]{args_preview}[/tool.args]")
        return
    f.write(f"{_i()}{gold}▸ ")
    for ch in tool_name:
        f.write(ch)
        f.flush()
        time.sleep(0.015)
    f.write(f"{reset}  \033[2m{args_preview}{reset}\n")
    f.flush()


def print_tool_output(output: str, success: bool, truncate: bool = True) -> None:
    if truncate:
        output = _truncate(output, max_lines=10)
    style = "tool.ok" if success else "tool.fail"
    # Indent each line of tool output
    indented = "\n".join(f"{_i()}  {line}" for line in output.split("\n"))
    _console.print(f"[{style}]{indented}[/{style}]")


class SubAgentDisplayManager:
    """Manages multiple concurrent sub-agent displays.

    Each agent gets its own stats and rolling tool-call log.
    All agents are rendered together so terminal escape-code
    erase/redraw stays consistent.
    """

    _MAX_VISIBLE = 4  # tool-call lines shown per agent

    def __init__(self):
        self._agents: dict[str, dict] = {}  # agent_id -> state dict
        self._lines_on_screen = 0

    def start(self, agent_id: str, label: str = "research") -> None:
        import time

        self._agents[agent_id] = {
            "label": label,
            "calls": [],
            "tool_count": 0,
            "token_count": 0,
            "start_time": time.monotonic(),
        }
        self._redraw()

    def set_tokens(self, agent_id: str, tokens: int) -> None:
        if agent_id in self._agents:
            self._agents[agent_id]["token_count"] = tokens

    def set_tool_count(self, agent_id: str, count: int) -> None:
        if agent_id in self._agents:
            self._agents[agent_id]["tool_count"] = count

    def add_call(self, agent_id: str, tool_desc: str) -> None:
        if agent_id in self._agents:
            self._agents[agent_id]["calls"].append(tool_desc)
            self._redraw()

    def clear(self, agent_id: str) -> None:
        # On completion: erase the live region, freeze a single-line summary
        # for this agent ("✓ research: … (stats)") above the live region so
        # the user sees each sub-agent finish cleanly without the tool-call
        # noise, then redraw remaining live agents.
        agent = self._agents.pop(agent_id, None)
        self._erase()
        if agent is not None:
            width = max(10, len(_i()) + content_width())
            line = _clip_to_width(self._render_completion_line(agent), width)
            _console.file.write(line + "\n")
            _console.file.flush()
        self._lines_on_screen = 0
        if self._agents:
            self._redraw()

    @staticmethod
    def _render_completion_line(agent: dict) -> str:
        stats = SubAgentDisplayManager._format_stats(agent)
        label = agent["label"]
        # dim green check + dim label; stats in parens
        line = f"{_i()}\033[38;2;120;200;140m✓\033[0m \033[2m{label}\033[0m"
        if stats:
            line += f"  \033[2m({stats})\033[0m"
        return line

    @staticmethod
    def _format_stats(agent: dict) -> str:
        import time

        start = agent["start_time"]
        if start is None:
            return ""
        elapsed = time.monotonic() - start
        if elapsed < 60:
            time_str = f"{elapsed:.0f}s"
        else:
            time_str = f"{elapsed / 60:.0f}m {elapsed % 60:.0f}s"
        tok = agent["token_count"]
        tok_str = f"{tok / 1000:.1f}k" if tok >= 1000 else str(tok)
        return f"{agent['tool_count']} tool uses · {tok_str} tokens · {time_str}"

    def _erase(self) -> None:
        if self._lines_on_screen > 0:
            f = _console.file
            for _ in range(self._lines_on_screen):
                f.write("\033[A\033[K")
            f.flush()

    def _render_agent_lines(self, agent: dict, compact: bool = False) -> list[str]:
        """Render one agent's block.

        compact=True → single line (label + stats + most-recent tool name);
        compact=False → header + up to _MAX_VISIBLE rolling tool-call lines.
        We use compact mode when multiple agents are live so the total live
        region stays small enough to fit on one screen. Otherwise cursor-up
        can't reach lines that have scrolled into scrollback, and every
        redraw pollutes history with a stale copy.
        """
        stats = self._format_stats(agent)
        label = agent["label"]
        header = f"{_i()}\033[38;2;255;200;80m▸ {label}\033[0m"
        if stats:
            header += f"  \033[2m({stats})\033[0m"
        if compact:
            latest = agent["calls"][-1] if agent["calls"] else ""
            if latest:
                # Strip long json tails for the inline view
                short = latest.split("  ")[0] if "  " in latest else latest
                header += f" \033[2m·\033[0m \033[2m{short}\033[0m"
            return [header]
        lines = [header]
        visible = agent["calls"][-self._MAX_VISIBLE :]
        for desc in visible:
            lines.append(f"{_i()}  \033[2m{desc}\033[0m")
        return lines

    def _redraw(self) -> None:
        f = _console.file
        self._erase()
        compact = len(self._agents) > 1
        width = max(10, len(_i()) + content_width())
        lines: list[str] = []
        for agent in self._agents.values():
            for ln in self._render_agent_lines(agent, compact=compact):
                lines.append(_clip_to_width(ln, width))
        for line in lines:
            f.write(line + "\n")
        f.flush()
        self._lines_on_screen = len(lines)


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
        _console.print(f"{_i()}[dim]{tool}: {log}[/dim]")


# ── Messages ───────────────────────────────────────────────────────────


async def print_markdown(
    text: str,
    cancel_event: "asyncio.Event | None" = None,
    instant: bool = False,
) -> None:
    import io
    import random
    from rich.padding import Padding

    _console.print()

    # Render markdown to a string buffer so we can type it out
    buf = io.StringIO()
    margin = len(_i())
    # Important: StringIO is not a TTY, so Rich would normally strip styles.
    # Force terminal rendering so ANSI style codes are preserved for typewriter output.
    buf_console = Console(
        file=buf,
        width=margin + content_width(),
        highlight=False,
        theme=_THEME,
        force_terminal=True,
        color_system=_console.color_system or "truecolor",
    )
    # Left padding is the centering margin, so the prose block lines up with
    # the rest of the UI instead of hugging the left edge.
    buf_console.print(Padding(Markdown(text), (0, 0, 0, margin)))
    rendered = buf.getvalue()

    # Strip trailing whitespace from each line so we don't type across the full width
    lines = rendered.split("\n")
    rendered = "\n".join(line.rstrip() for line in lines)

    f = _console.file

    # Headless / non-interactive / --no-stream: dump it in one write.
    if instant or not stream_enabled():
        f.write(rendered)
        f.write("\n")
        f.flush()
        return

    # CRT typewriter effect — async so the event loop can service signal
    # handlers (Ctrl+C during streaming) between characters. If cancelled
    # mid-type, stop cleanly: write an ANSI reset so half-open color state
    # doesn't bleed onto the "interrupted" line, and return.
    rng = random.Random(42)
    cancelled = False
    for ch in rendered:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        f.write(ch)
        f.flush()
        if ch == "\n":
            await asyncio.sleep(0.002)
        elif ch == " ":
            await asyncio.sleep(0.002)
        elif rng.random() < 0.03:
            await asyncio.sleep(0.015)
        else:
            await asyncio.sleep(0.004)
    f.write("\033[0m\n" if cancelled else "\n")
    f.flush()


def print_error(message: str) -> None:
    _console.print(f"\n{_i()}[bold red]Error:[/bold red] {message}")


def print_turn_complete() -> None:
    pass  # no separator — clean output


def print_interrupted() -> None:
    _console.print(f"\n{_i()}[dim italic]interrupted[/dim italic]")


def print_compacted(old_tokens: int, new_tokens: int) -> None:
    _console.print(
        f"{_i()}[dim]context compacted: {old_tokens:,} → {new_tokens:,} tokens[/dim]"
    )


# ── Approval ───────────────────────────────────────────────────────────


def print_approval_header(count: int) -> None:
    label = f"Approval required — {count} item{'s' if count != 1 else ''}"
    _console.print()
    _console.print(
        f"{_i()}",
        Panel(
            f"[bold yellow]{label}[/bold yellow]", border_style="yellow", expand=False
        ),
    )


def print_approval_item(index: int, total: int, tool_name: str, operation: str) -> None:
    _console.print(
        f"\n{_i()}[bold]\\[{index}/{total}][/bold]  [tool.name]{tool_name}[/tool.name]  {operation}"
    )


def print_yolo_approve(count: int) -> None:
    _console.print(
        f"{_i()}[bold yellow]yolo →[/bold yellow] auto-approved {count} item(s)"
    )


# ── Help ───────────────────────────────────────────────────────────────

HELP_ROWS: tuple[tuple[str, str, str], ...] = (
    ("/help", "", "Show this help"),
    ("/new", "", "Start a fresh conversation"),
    ("/clear", "", "Clear the workspace directory"),
    ("/undo", "", "Drop the last turn from context"),
    ("/compact", "", "Compact the context window"),
    ("/resume", "[session id]", "Load a past session as context"),
    ("/model", "[id]", "Show or switch the model"),
    ("/effort", "[low|medium|high]", "Set reasoning effort preference"),
    ("/yolo", "", "Toggle auto-approve mode"),
    ("/status", "", "Model, turns, and current settings"),
    ("/plan", "", "Show the current plan"),
    ("/share-traces", "[public|private]", "Show or change trace visibility"),
    ("/quit", "", "Exit"),
    ("history", "[n]", "List recent sessions"),
    ("recall", "<session id>", "Replay a past session's tool log"),
    ("index", "<path>", "Index a directory for semantic search"),
    ("team", "<task>", "Run the multi-agent pipeline"),
    ("report", "", "Show the last team run's report"),
    ("guardrails", "", "Show flagged prompt-injection patterns"),
    ('ask', '"<question>" <doc>', "Answer a question about a document"),
    ("ingest", "<path>", "Parse a document once into stored JSON"),
    ("inspect", "<path>", "Extract fields + boxes from a PDF/image"),
    ("to-csv", "<path>", "Write a PDF's tables out as CSV files"),
    ("extract-pdf", "<path>", "Convert a PDF into structured data"),
)


def _help_column_widths(
    rows: tuple[tuple[str, str, str], ...],
) -> tuple[int, int]:
    return (
        max(len(command) for command, _, _ in rows),
        max(len(args) for _, args, _ in rows),
    )


def _format_help_row(
    command: str,
    args: str,
    description: str,
    command_width: int,
    args_width: int,
) -> str:
    command_gap = " " * (command_width - len(command) + 2)
    args_gap = " " * (args_width - len(args) + 2)
    command_markup = f"[cyan]{escape(command)}[/cyan]"
    args_markup = f"[muted]{escape(args)}[/muted]" if args else ""
    return f"{_i()}  {command_markup}{command_gap}{args_markup}{args_gap}{description}"


def format_help_text(rows: tuple[tuple[str, str, str], ...] | None = None) -> str:
    help_rows = HELP_ROWS if rows is None else rows
    command_width, args_width = _help_column_widths(help_rows)
    return "\n".join(
        [f"{_i()}[bold]Commands[/bold]"]
        + [
            _format_help_row(
                command,
                args,
                description,
                command_width,
                args_width,
            )
            for command, args, description in help_rows
        ]
    )


def print_help() -> None:
    _console.print()
    _console.print(format_help_text())
    _console.print()


# ── Plan display ───────────────────────────────────────────────────────


def format_plan_display() -> str:
    """Format the current plan for display."""
    from agent.core.plan import get_current_plan

    plan = get_current_plan()
    if not plan:
        return ""

    completed = [t for t in plan if t["status"] == "completed"]
    in_progress = [t for t in plan if t["status"] == "in_progress"]
    pending = [t for t in plan if t["status"] == "pending"]

    lines = []
    for t in completed:
        lines.append(f"{_i()}[green]✓[/green] [dim]{t['content']}[/dim]")
    for t in in_progress:
        lines.append(f"{_i()}[yellow]▸[/yellow] {t['content']}")
    for t in pending:
        lines.append(f"{_i()}[dim]○ {t['content']}[/dim]")

    summary = f"[dim]{len(completed)}/{len(plan)} done[/dim]"
    lines.append(f"{_i()}{summary}")
    return "\n".join(lines)


def print_plan() -> None:
    plan_str = format_plan_display()
    if plan_str:
        _console.print(plan_str)


# ── Formatting for plan_tool output (used by plan_tool handler) ────────


def format_plan_tool_output(todos: list) -> str:
    if not todos:
        return "Plan is empty."

    lines = ["Plan updated:", ""]
    completed = [t for t in todos if t["status"] == "completed"]
    in_progress = [t for t in todos if t["status"] == "in_progress"]
    pending = [t for t in todos if t["status"] == "pending"]

    for t in completed:
        lines.append(f"  [x] {t['id']}. {t['content']}")
    for t in in_progress:
        lines.append(f"  [~] {t['id']}. {t['content']}")
    for t in pending:
        lines.append(f"  [ ] {t['id']}. {t['content']}")

    lines.append(f"\n{len(completed)}/{len(todos)} done")
    return "\n".join(lines)


# ── Internal helpers ───────────────────────────────────────────────────


def _truncate(text: str, max_lines: int = 6) -> str:
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


# ── Headless mode display ──────────────────────────────────────────────


class HeadlessDisplayManager:
    """Manages rich display output for headless (one-shot) execution mode.

    Provides structured output with panels, progress indicators, and summary
    panels suitable for scripting and CI/CD pipelines.
    """

    def __init__(self, console: Console | None = None, show_progress: bool = True):
        self.console = console or _console
        self.show_progress = show_progress
        self._tool_count = 0
        self._start_time = 0.0
        self._current_tool: str | None = None

    def start_run(self, task: str, model: str, mode: str = "single") -> None:
        """Print the run header with task info."""
        import time
        self._start_time = time.monotonic()

        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Task", task[:100] + ("..." if len(task) > 100 else ""))
        table.add_row("Model", model)
        table.add_row("Mode", mode.capitalize())

        self.console.print()
        self.console.print(Panel(table, title="[bold]Swarn Agent Run[/bold]", border_style="cyan", expand=False))
        self.console.print()

    def note_tool(self, tool_name: str) -> None:
        """Count a tool call that something else already rendered.

        The agent loop draws its own per-tool lines (agent/utils/ui.py), so
        start_tool() would print a second, duplicate line for the same call.
        This records the call for the "Tools used" row of the summary panel
        and prints nothing — without it that row reports 0 on every run.
        """
        self._tool_count += 1
        self._current_tool = tool_name

    def start_tool(self, tool_name: str, args_preview: str = "") -> None:
        """Show tool invocation with optional args preview."""
        import time
        self._tool_count += 1
        self._current_tool = tool_name

        if self.show_progress:
            self.console.print(f"  [tool.name]> {tool_name}[/tool.name]  [dim]{args_preview}[/dim]")

    def finish_tool(self, output: str, success: bool, truncate: bool = True) -> None:
        """Show tool completion with output."""
        if truncate:
            output = _truncate(output, max_lines=8)

        style = "tool.ok" if success else "tool.fail"
        status_icon = "OK" if success else "FAIL"

        if self.show_progress and output.strip():
            indented = "\n".join(f"    {line}" for line in output.split("\n"))
            self.console.print(f"  [{style}][{status_icon}] {self._current_tool}[/{style}]")
            self.console.print(f"[{style}]{indented}[/{style}]")
        elif self.show_progress:
            self.console.print(f"  [{style}][{status_icon}] {self._current_tool}[/{style}]")

        self._current_tool = None

    def print_markdown(self, text: str, instant: bool = True) -> None:
        """Print markdown output (instant mode for headless)."""
        from rich.markdown import Markdown
        from rich.padding import Padding

        self.console.print()
        self.console.print(Padding(Markdown(text), (0, 0, 0, 2)))

    def print_result(self, outcome: str, session_id: str, duration: float | None = None) -> None:
        """Print the final result summary panel."""
        from rich.table import Table

        status_style = "green" if outcome == "complete" else "red"
        status_text = "SUCCESS" if outcome == "complete" else "FAILED"

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Outcome", f"[{status_style}]{status_text} {outcome}[/{status_style}]")
        table.add_row("Session", session_id[:8])
        if duration is not None:
            table.add_row("Duration", f"{duration:.1f}s")
        table.add_row("Tools used", str(self._tool_count))

        self.console.print()
        self.console.print(
            Panel(
                table,
                title=f"[bold {status_style}]Run Complete[/bold {status_style}]",
                border_style=status_style,
                expand=False,
            )
        )

    def print_error(self, message: str) -> None:
        """Print an error message in a panel."""
        self.console.print(
            Panel(
                f"[bold red]Error:[/bold red] {message}",
                title="[bold red]Failed[/bold red]",
                border_style="red",
                expand=False,
            )
        )


# Global headless display instance
_headless_display: HeadlessDisplayManager | None = None


def get_headless_display(console: Console | None = None, show_progress: bool = True) -> HeadlessDisplayManager:
    """Get or create the global headless display manager."""
    global _headless_display
    if _headless_display is None:
        _headless_display = HeadlessDisplayManager(console=console, show_progress=show_progress)
    return _headless_display


def reset_headless_display() -> None:
    """Reset the global headless display manager."""
    global _headless_display
    _headless_display = None
