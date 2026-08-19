"""
Loader for the Lain theme definition (theme.yaml).

The YAML is the single source of truth for colors, branding strings, spinner
frames and the banner art; this module just parses it once and exposes typed
accessors so the display code never hardcodes a hex value.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_THEME_PATH = Path(__file__).with_name("theme.yaml")


@lru_cache(maxsize=1)
def load_theme() -> dict[str, Any]:
    """Parse theme.yaml. Cached — the file never changes at runtime."""
    with _THEME_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def color(key: str, default: str = "#FF1493") -> str:
    return load_theme().get("colors", {}).get(key, default)


def branding(key: str, default: str = "") -> str:
    return load_theme().get("branding", {}).get(key, default)


def spinner(key: str) -> list:
    return load_theme().get("spinner", {}).get(key, [])


def spinner_opt(key: str, default=None):
    """Scalar entry under `spinner:` (label, interval, color) — the list
    accessor above would return [] for these."""
    return load_theme().get("spinner", {}).get(key, default)


def prompt_placeholders() -> list[str]:
    return load_theme().get("prompt_placeholders", [])


def tool_emoji(tool_name: str, default: str = "◈") -> str:
    return load_theme().get("tool_emojis", {}).get(tool_name, default)


def tool_prefix() -> str:
    return load_theme().get("tool_prefix", "┊")


def banner_logo_lines() -> list[str]:
    return load_theme().get("banner_logo", "").rstrip("\n").split("\n")


def banner_hero_lines() -> list[str]:
    return load_theme().get("banner_hero", "").rstrip("\n").split("\n")
