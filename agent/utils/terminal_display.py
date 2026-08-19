"""
Theme selector for the CLI display layer.

Two interchangeable implementations live side by side:

  agent/utils/classic/  — the original green-on-black CRT UI (default)
  agent/utils/lain/     — Serial Experiments Lain theme, driven by theme.yaml

Callers keep importing `agent.utils.terminal_display`; this module forwards
every public name to whichever theme is active. Pick one with:

    SWARN_THEME=lain      (or "classic", the default)
"""

import os

_THEMES = {"classic", "lain"}


def active_theme_name() -> str:
    name = os.environ.get("SWARN_THEME", "classic").strip().lower()
    return name if name in _THEMES else "classic"


if active_theme_name() == "lain":
    from agent.utils.lain.terminal_display import *  # noqa: F401,F403
    from agent.utils.lain import terminal_display as _impl
else:
    from agent.utils.classic.terminal_display import *  # noqa: F401,F403
    from agent.utils.classic import terminal_display as _impl


def __getattr__(name: str):
    """Forward anything the star-import skipped (underscore-prefixed helpers,
    names not listed in __all__) to the active theme module."""
    try:
        return getattr(_impl, name)
    except AttributeError as exc:  # pragma: no cover - mirrors normal lookup
        raise AttributeError(name) from exc
