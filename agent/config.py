"""
Configuration — the one place environment variables are read.

Every `SWARN_*` knob the platform honours is declared here, with its default
and a one-line explanation, so `.env.example` and the docs have a single
source to track. Modules import from this module instead of calling
`os.environ.get` inline; nothing else in the package should read the
environment directly.

Two access shapes, deliberately:

  * **Module-level constants** (`MAX_ITERATIONS`, `DEFAULT_TIMEOUT`, …) for
    settings that were already frozen at import time. Reading them here
    preserves that timing exactly.
  * **Functions** (`knowledge_dir()`, `sandbox_mode()`, …) for settings that
    were read at call time, where a test or a long-lived process can change
    the environment between calls. Turning these into constants would be a
    silent behaviour change, so they stay lazy.
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load .env BEFORE any constant below is evaluated. This module freezes
# module-level settings at import time, so the .env must be in the
# environment by then — entry points that call load_dotenv() later (or
# not at all) would otherwise silently get the defaults.
load_dotenv()

# ─── helpers ──────────────────────────────────────────────────────────────────


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    """Read an integer setting, falling back to `default` on a malformed value."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _flag(name: str) -> bool:
    """A switch is on when set to 1/true/yes/on (case-insensitive)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ═══ deployed model endpoint ══════════════════════════════════════════════════
# ALL LLM calls route to one deployed, OpenAI-compatible endpoint; see
# agent/llm/router.py ("PRODUCTION ENDPOINT — CHANGE HERE").
# TODO(production): defaults below point at the TEST deployment.

DEPLOYED_MODEL = _str("SWARN_DEPLOYED_MODEL", "qwen3.5-9b")
DEPLOYED_BASE_URL = _str(
    "SWARN_DEPLOYED_BASE_URL",
    "https://secretaryalumniaffairs-mmm--qwen35-9b-serve.modal.run/v1",
)
DEPLOYED_API_KEY = _str("SWARN_DEPLOYED_API_KEY", "dummy")


def openai_api_key() -> Optional[str]:
    """Key for the OpenAI-compatible client; local servers ignore it."""
    return os.environ.get("OPENAI_API_KEY")


# ═══ agent loop ═══════════════════════════════════════════════════════════════

MAX_ITERATIONS = _int("SWARN_MAX_ITERATIONS", 30)          # ReAct iteration cap
CONTEXT_CHAR_BUDGET = _int("SWARN_CONTEXT_CHAR_BUDGET", 400_000)  # compaction threshold


# ═══ execution ════════════════════════════════════════════════════════════════

DEFAULT_TIMEOUT = _int("SWARN_EXEC_TIMEOUT", 300)          # per-call timeout (seconds)
SANDBOX_IMAGE = _str("SWARN_SANDBOX_IMAGE", "python:3.11-slim")


def sandbox_mode() -> str:
    """"docker" | "subprocess" to force a backend; "" means auto-detect."""
    return _str("SWARN_SANDBOX").lower()


# ═══ search engine ════════════════════════════════════════════════════════════


def search_workers() -> int:
    """Parallel draft/debug/improve nodes."""
    return _int("SWARN_SEARCH_WORKERS", 1)


def code_model(default: str) -> str:
    """Display label for code generation; "mock:<name>" runs offline."""
    return _str("SWARN_CODE_MODEL", default)


def feedback_model(default: str) -> str:
    """Display label for feedback/review; "mock:<name>" runs offline."""
    return _str("SWARN_FEEDBACK_MODEL", default)


# ═══ knowledge store ══════════════════════════════════════════════════════════


def knowledge_dir() -> Optional[str]:
    """Override for the playbook + run archive location (see agent/paths.py)."""
    return os.environ.get("SWARN_KNOWLEDGE_DIR")


# ═══ observability ════════════════════════════════════════════════════════════


def tracing_enabled() -> bool:
    return _flag("SWARN_ENABLE_TRACING")


def otel_endpoint() -> Optional[str]:
    """OTLP collector endpoint; None exports spans to the console."""
    return os.environ.get("OTEL_EXPORTER_ENDPOINT")
