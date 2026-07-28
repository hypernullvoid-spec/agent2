"""
Phase 5: Structured Memory & Traces

Replaces the flat runs/*.json files (Phase 1) with a proper session model
that makes every agent run inspectable, comparable, and recallable.

What gets recorded per session
───────────────────────────────
  • PLAN         — Claude's reasoning/planning text
  • TOOL_CALL    — tool name + full input (logged BEFORE execution)
  • TOOL_RESULT  — raw tool output (logged AFTER execution)
  • CORRECTION   — self-correction event: kind + attempt number (Phase 4)
  • COMPLETE     — finish_task was called with a summary
  • ERROR        — run ended abnormally (max_corrections, max_iterations)

Storage layout
───────────────
  sessions/
    index.json                      ← lightweight index, last 100 sessions
    <uuid>/
      trace.json                    ← full structured trace (machine-readable)
      summary.md                    ← human-readable replay

Two agent tools expose this to the agent itself (added to tools.py):
  list_sessions()         — tabular view of recent runs
  recall_session(id)      — detailed tool-call log for a past session
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"


# ─── step model ───────────────────────────────────────────────────────────────

class StepKind(str, Enum):
    PLAN        = "plan"
    TOOL_CALL   = "tool_call"
    TOOL_RESULT = "tool_result"
    CORRECTION  = "correction"
    COMPLETE    = "complete"
    ERROR       = "error"


@dataclass
class Step:
    kind:      StepKind
    timestamp: float
    data:      dict

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "time": self.timestamp, "data": self.data}


# ─── session model ────────────────────────────────────────────────────────────

@dataclass
class Session:
    id:          str
    task:        str
    model:       str
    started_at:  float
    steps:       list[Step] = field(default_factory=list)
    ended_at:    Optional[float] = None
    outcome:     Optional[str]   = None    # complete | max_iterations | max_corrections | error
    summary:     Optional[str]   = None    # from finish_task
    corrections: int             = 0

    # Phase 16: optional live-step subscribers — NOT persisted, NOT part
    # of to_dict()/to_markdown(). trace.json/summary.md are only written
    # once, at close_session(), which means nothing reading from disk
    # can show a session "live" while it's still running. on_step is a
    # parallel, in-memory notification: the dashboard's websocket layer
    # appends a callback here when a client starts watching a session,
    # and add_step() below fires every callback immediately, synchronously,
    # on every step — so a still-running session is observable in real
    # time without changing anything about how/when this class persists
    # to disk. A callback raising never breaks the agent loop (see
    # add_step's try/except) — a dashboard bug must never be able to
    # crash an agent run.
    on_step: list = field(default_factory=list, repr=False, compare=False)

    # ── step helpers ──────────────────────────────────────────────────

    def add_step(self, kind: StepKind, **data) -> "Step":
        step = Step(kind=kind, timestamp=time.time(), data=data)
        self.steps.append(step)
        for callback in self.on_step:
            try:
                callback(self, step)
            except Exception:
                pass   # a broken dashboard subscriber must never break the agent run
        return step

    # ── derived metrics ───────────────────────────────────────────────

    def duration_s(self) -> Optional[float]:
        if self.ended_at:
            return round(self.ended_at - self.started_at, 1)
        return None

    def tool_call_steps(self) -> list[Step]:
        return [s for s in self.steps if s.kind == StepKind.TOOL_CALL]

    def tool_call_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.tool_call_steps():
            name = s.data.get("tool", "?")
            counts[name] = counts.get(name, 0) + 1
        return counts

    # ── serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "task":        self.task,
            "model":       self.model,
            "started_at":  self.started_at,
            "ended_at":    self.ended_at,
            "outcome":     self.outcome,
            "summary":     self.summary,
            "corrections": self.corrections,
            "duration_s":  self.duration_s(),
            "tool_counts": self.tool_call_counts(),
            "steps":       [s.to_dict() for s in self.steps],
        }

    def to_markdown(self) -> str:
        """Human-readable session replay — useful for reviewing what happened."""
        ts  = datetime.fromtimestamp(self.started_at).strftime("%Y-%m-%d %H:%M:%S")
        dur = f"{self.duration_s()}s" if self.duration_s() else "—"

        lines = [
            f"# Session `{self.id[:8]}`",
            "",
            f"| | |",
            f"|---|---|",
            f"| **Task** | {self.task} |",
            f"| **Model** | {self.model} |",
            f"| **Started** | {ts} |",
            f"| **Duration** | {dur} |",
            f"| **Outcome** | {self.outcome or 'in progress'} |",
            f"| **Self-corrections** | {self.corrections} |",
        ]

        if self.summary:
            lines += ["", f"> **Summary:** {self.summary}"]

        counts = self.tool_call_counts()
        if counts:
            lines += ["", "## Tool usage"]
            for tool, n in sorted(counts.items(), key=lambda x: -x[1]):
                lines.append(f"- `{tool}` × {n}")

        lines += ["", "## Trace"]

        n = 0
        for s in self.steps:
            n  += 1
            ts2 = datetime.fromtimestamp(s.timestamp).strftime("%H:%M:%S")

            if s.kind == StepKind.PLAN:
                text = s.data.get("text", "")[:400]
                lines += ["", f"### {n}. 💭 Plan  `{ts2}`", "", text]

            elif s.kind == StepKind.TOOL_CALL:
                inp = json.dumps(s.data.get("input", {}), ensure_ascii=False)
                # Truncate very long inputs (e.g. write_file with large content)
                if len(inp) > 300:
                    inp = inp[:300] + "…"
                lines += [
                    "",
                    f"### {n}. 🔧 `{s.data.get('tool', '?')}`  `{ts2}`",
                    f"```json",
                    inp,
                    "```",
                ]

            elif s.kind == StepKind.TOOL_RESULT:
                res = str(s.data.get("result", ""))[:500]
                if len(str(s.data.get("result", ""))) > 500:
                    res += "…"
                lines += ["```", res, "```"]

            elif s.kind == StepKind.CORRECTION:
                kind  = s.data.get("error_kind", "?")
                att   = s.data.get("attempt", "?")
                lines += [
                    "",
                    f"### {n}. ⚠️  Correction  `{ts2}`",
                    f"Error kind: `{kind}`  •  attempt {att}",
                ]

            elif s.kind == StepKind.COMPLETE:
                lines += [
                    "",
                    f"### {n}. ✅ Complete  `{ts2}`",
                    "",
                    s.data.get("summary", ""),
                ]

            elif s.kind == StepKind.ERROR:
                lines += [
                    "",
                    f"### {n}. ❌ Stopped  `{ts2}`",
                    s.data.get("reason", ""),
                ]

        return "\n".join(lines)


# ─── session store ────────────────────────────────────────────────────────────

class SessionStore:
    """
    Manages the sessions/ directory: creates sessions, persists them,
    maintains the index, and serves lookups.

    Called by AgentLoop at the start and end of every run(), and
    by the list_sessions / recall_session tools (added in tools.py).
    """

    def __init__(self, sessions_dir: Path = SESSIONS_DIR):
        self.dir = sessions_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.dir / "index.json"
        self._index: list[dict] = self._load_index()
        # Phase 16: store-level subscribers, applied to every NEW session
        # automatically. The dashboard doesn't know a session's UUID
        # before it's created (new_session() generates it), so it
        # subscribes here, at the store level, rather than trying to
        # attach to a specific not-yet-existent Session.
        self._global_step_subscribers: list = []

    def subscribe_to_all_sessions(self, callback) -> None:
        """
        Register a callback(session, step) that fires on every step of
        every session created from this point forward — this is how
        Phase 16's dashboard gets live updates without polling the
        filesystem or needing trace.json to exist yet (it doesn't, until
        close_session() runs). Does not affect already-created Session
        objects retroactively.
        """
        self._global_step_subscribers.append(callback)

    # ── lifecycle ─────────────────────────────────────────────────────

    def new_session(self, task: str, model: str) -> Session:
        session = Session(
            id         = str(uuid.uuid4()),
            task       = task,
            model      = model,
            started_at = time.time(),
        )
        session.on_step.extend(self._global_step_subscribers)
        return session

    def close_session(self, session: Session) -> None:
        session.ended_at = time.time()
        self._persist(session)
        self._update_index(session)
        n_calls = len(session.tool_call_steps())
        print(
            f"\n[memory] ✓ Session {session.id[:8]}  "
            f"saved  ({session.duration_s()}s  •  "
            f"{n_calls} tool call{'s' if n_calls != 1 else ''}  •  "
            f"{session.corrections} correction{'s' if session.corrections != 1 else ''})"
        )

    # ── querying ──────────────────────────────────────────────────────

    def list_sessions(self, n: int = 10) -> str:
        if not self._index:
            return "No sessions recorded yet."

        rows = self._index[:n]
        hdr  = f"{'ID':<10}  {'Outcome':<14}  {'Dur':>6}  {'Calls':>5}  {'Corr':>4}  Task"
        sep  = "─" * 76
        lines = [hdr, sep]

        for e in rows:
            sid     = e["id"][:8]
            outcome = (e.get("outcome") or "?")[:12]
            dur     = f"{e.get('duration_s', '?')}s".rjust(6)
            calls   = str(e.get("tool_calls", "?")).rjust(5)
            corr    = str(e.get("corrections", 0)).rjust(4)
            task    = (e.get("task") or "")[:36]
            lines.append(f"{sid:<10}  {outcome:<14}  {dur}  {calls}  {corr}  {task}")

        return "\n".join(lines)

    def get_session(self, session_id: str) -> Optional[dict]:
        """Load a full trace by UUID prefix (min 4 chars)."""
        for entry in self._index:
            if entry["id"].startswith(session_id):
                path = self.dir / entry["id"] / "trace.json"
                if path.exists():
                    with open(path) as f:
                        return json.load(f)
        return None

    def recall_as_text(self, session_id: str) -> str:
        """
        Human-readable summary of a past session for the agent to read.
        Enough detail to understand what happened without the full JSON.
        """
        data = self.get_session(session_id)
        if not data:
            return (
                f"No session found with ID starting '{session_id}'. "
                "Use list_sessions() to see available session IDs."
            )

        lines = [
            f"Session {data['id'][:8]}",
            f"Task     : {data['task']}",
            f"Outcome  : {data.get('outcome', '?')}",
            f"Duration : {data.get('duration_s', '?')}s",
            f"Corrections: {data.get('corrections', 0)}",
        ]
        if data.get("summary"):
            lines.append(f"Summary  : {data['summary']}")

        counts = data.get("tool_counts", {})
        if counts:
            lines.append("\nTool usage:")
            for tool, n in sorted(counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {tool} × {n}")

        steps = data.get("steps", [])
        call_steps = [s for s in steps if s["kind"] == "tool_call"]
        if call_steps:
            lines.append(f"\nTool calls ({len(call_steps)} total):")
            for s in call_steps[:20]:   # cap at 20 for readability
                inp = json.dumps(s["data"].get("input", {}), ensure_ascii=False)
                if len(inp) > 120:
                    inp = inp[:120] + "…"
                lines.append(f"  {s['data'].get('tool', '?')}  {inp}")
            if len(call_steps) > 20:
                lines.append(f"  … and {len(call_steps) - 20} more")

        return "\n".join(lines)

    # ── persistence ───────────────────────────────────────────────────

    def _persist(self, session: Session) -> None:
        session_dir = self.dir / session.id
        session_dir.mkdir(exist_ok=True)

        with open(session_dir / "trace.json", "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, default=str)

        with open(session_dir / "summary.md", "w", encoding="utf-8") as f:
            f.write(session.to_markdown())

    def _load_index(self) -> list[dict]:
        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _update_index(self, session: Session) -> None:
        entry = {
            "id":          session.id,
            "task":        session.task[:80],
            "model":       session.model,
            "outcome":     session.outcome,
            "duration_s":  session.duration_s(),
            "tool_calls":  len(session.tool_call_steps()),
            "corrections": session.corrections,
            "started_at":  session.started_at,
        }
        # Upsert: remove existing entry for this session, prepend updated one
        self._index = [e for e in self._index if e["id"] != session.id]
        self._index.insert(0, entry)
        self._index = self._index[:100]         # keep last 100

        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, default=str)


# ─── singleton ────────────────────────────────────────────────────────────────

_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
