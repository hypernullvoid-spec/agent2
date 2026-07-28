"""
Back-compat shim — the Phase 2 Sandbox API now delegates to
agent/execution.py, which adds:

  • a real cross-platform SubprocessBackend (the old fallback hardcoded
    "python3" and couldn't run shell commands — broken on Windows)
  • per-call timeouts (ML training needs minutes; a data peek needs seconds)
  • structured ExecResult (exit code + timing) for the search engine

tools.py keeps calling get_sandbox().exec_python(...) -> str unchanged.
"""

from typing import Optional

from agent.execution import (  # noqa: F401 — re-exported for existing imports
    WORKSPACE_DIR, get_backend, close_backend,
)


class Sandbox:
    """String-in/string-out facade over the active execution backend."""

    def exec_python(self, code: str, timeout: Optional[int] = None) -> str:
        return get_backend().exec_python(code, timeout=timeout).as_text()

    def exec_shell(self, command: str, timeout: Optional[int] = None) -> str:
        return get_backend().exec_shell(command, timeout=timeout).as_text()

    def close(self):
        close_backend()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_sandbox: Optional[Sandbox] = None


def get_sandbox() -> Sandbox:
    global _sandbox
    if _sandbox is None:
        _sandbox = Sandbox()
    return _sandbox


def close_sandbox():
    global _sandbox
    if _sandbox:
        _sandbox.close()
        _sandbox = None
