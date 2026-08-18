"""
Project directory locations, resolved once.

Modules used to compute these inline as `os.path.join(os.path.dirname(
__file__), "..", "workspace")`, which silently depended on every module
sitting exactly one level below the repo root — so moving a file into a
subpackage repointed its workspace at `agent/` instead. Deriving the paths
here, from a module whose own depth is fixed, makes that failure mode
impossible: nothing else needs to know how deep it is.
"""

from __future__ import annotations

import os
from pathlib import Path

# agent/paths.py -> agent/ -> repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE_DIR = str(PROJECT_ROOT / "workspace")
RUNS_DIR = str(PROJECT_ROOT / "runs")
SESSIONS_DIR = PROJECT_ROOT / "sessions"
KNOWLEDGE_DIR = str(PROJECT_ROOT / "knowledge")
CHROMA_DIR = str(PROJECT_ROOT / ".chroma")


def project_path(*parts: str) -> str:
    """Absolute path to something under the repo root."""
    return os.path.join(str(PROJECT_ROOT), *parts)
