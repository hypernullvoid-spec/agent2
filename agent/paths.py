"""
Canonical filesystem locations for the platform.

Every runtime directory the agent writes to hangs off the repository root
(the parent of this package), *not* off the module that happens to need it.
Before this module existed, seven modules each re-derived their own
`os.path.join(os.path.dirname(__file__), "..", "workspace")` — which silently
pointed somewhere else the moment a module moved into a subpackage.

Also the one place the workspace path guard lives: `safe_path()` resolves a
caller-supplied relative path inside WORKSPACE_DIR and rejects traversal.
"""

import os
from pathlib import Path

# agent/paths.py → agent/ → repository root
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent

WORKSPACE_DIR = str(ROOT_DIR / "workspace")   # everything the agent creates
RUNS_DIR = str(ROOT_DIR / "runs")             # Phase 1 flat run logs
SESSIONS_DIR = ROOT_DIR / "sessions"          # Phase 5 session traces (Path)
KNOWLEDGE_DIR = str(ROOT_DIR / "knowledge")   # cross-run playbook + run archive

# workspace subdirectories owned by individual phases
PLOTS_SUBDIR = "plots"
DEPLOYMENTS_SUBDIR = "deployments"
FINETUNE_SUBDIR = "finetune"


def safe_path(path: str, root: str = WORKSPACE_DIR) -> str:
    """Resolve a relative path inside `root`. Rejects path traversal."""
    full = os.path.abspath(os.path.join(root, path))
    if not (full == root or full.startswith(root + os.sep)):
        raise ValueError(f"Path '{path}' escapes the workspace directory")
    return full


def safe_filename(name: str) -> str:
    """Reduce an arbitrary identifier to a filesystem-safe directory name."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
