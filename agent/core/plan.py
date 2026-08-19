"""In-memory plan store backing the interactive `/plan` view.

terminal_display.format_plan_display() used to import this from
`agent.tools.plan_tool`, a module that never existed — the old agent/tools.py
(now agent/runtime/tools.py) is a single module, not a package, so `/plan`
raised ModuleNotFoundError. The state lives here instead: todo dicts, each
{"id": int, "content": str, "status": "pending"|"in_progress"|"completed"}.
"""

from __future__ import annotations

_VALID_STATUS = ("pending", "in_progress", "completed")

_plan: list[dict] = []


def get_current_plan() -> list[dict]:
    """Return the current plan (a copy — callers must not mutate the store)."""
    return list(_plan)


def set_current_plan(todos: list[dict]) -> list[dict]:
    """Replace the plan. Entries are normalized: missing ids are assigned in
    order, and an unrecognized status falls back to "pending"."""
    global _plan
    normalized: list[dict] = []
    for i, todo in enumerate(todos or [], start=1):
        status = todo.get("status", "pending")
        normalized.append(
            {
                "id": todo.get("id", i),
                "content": str(todo.get("content", "")),
                "status": status if status in _VALID_STATUS else "pending",
            }
        )
    _plan = normalized
    return get_current_plan()


def clear_plan() -> None:
    global _plan
    _plan = []
