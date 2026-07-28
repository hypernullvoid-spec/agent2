"""
Phase 4: Self-Correction Loop

When a tool call returns an error, the SelfCorrectionPolicy:

  1. Detects whether the result is genuinely a failure (vs. code that
     intentionally prints "Error: ..." or exits non-zero as part of its
     own logic — we use precise signal matching to avoid false positives)
  2. Classifies the error kind: syntax / import / file / runtime / timeout /
     shell / generic
  3. Enriches the tool_result string with a structured hint so Claude must
     diagnose before it retries — instead of re-running the exact same code
  4. Tracks consecutive errors (errors without a success between them) and
     signals ABORT when the budget is exhausted

The policy itself never retries anything — retrying is Claude's job.
What this phase gives the agent is:
  - A clear label saying "this was an error, here is the kind, here is
    the guidance" prepended to the raw error text
  - A retry counter ("attempt 2 of 3, 1 remaining") so Claude knows how
    urgent it is to get it right this time
  - A hard stop when it's clear the agent is stuck in a loop

Integration with agent_loop.py:
  AgentLoop receives an optional SelfCorrectionPolicy instance and calls
  policy.assess(tool_name, result) after every tool call. The returned
  (is_error, enriched_result) drives logging and abort decisions.
"""

import re
from enum import Enum

MAX_CONSECUTIVE_ERRORS = 3     # abort if this many errors occur back-to-back


class ErrorKind(str, Enum):
    SYNTAX   = "syntax"
    IMPORT   = "import"
    FILE     = "file"
    RUNTIME  = "runtime"
    TIMEOUT  = "timeout"
    SHELL    = "shell"
    GENERIC  = "generic"


# Structured hint injected per error kind.
# Phrased as instructions rather than explanations so Claude acts on them.
HINTS: dict[ErrorKind, str] = {
    ErrorKind.SYNTAX: (
        "The code has a Python syntax error. Read the message and identify "
        "the exact line and column. Rewrite only the affected part to fix "
        "the syntax, then retry."
    ),
    ErrorKind.IMPORT: (
        "A Python module is not installed. Call install_package('<name>') "
        "first, then retry the code. Do not assume the package is present."
    ),
    ErrorKind.FILE: (
        "A file or directory was not found. Call list_files('.') to verify "
        "what exists, correct the path, then retry."
    ),
    ErrorKind.RUNTIME: (
        "A runtime exception occurred. Read the full traceback from the "
        "bottom up to find the root cause. Fix the specific issue — do not "
        "re-run the same code without a concrete change."
    ),
    ErrorKind.TIMEOUT: (
        "The command timed out. Break the operation into smaller pieces, "
        "reduce the dataset size, or restructure the logic to be faster "
        "before retrying."
    ),
    ErrorKind.SHELL: (
        "The shell command exited with a non-zero code. Read stderr "
        "carefully. Fix or replace the command — do not repeat the exact "
        "same invocation."
    ),
    ErrorKind.GENERIC: (
        "An error occurred. Diagnose what went wrong before trying a "
        "different approach. Do not repeat the same action."
    ),
}


class SelfCorrectionPolicy:
    """
    Stateful error-detection and correction-guidance policy.

    One instance lives for the duration of an agent session. The agent loop
    calls assess() after every tool result.

    State tracked:
      consecutive_errors  — errors without a success in between; resets on
                            any successful tool call
      total_corrections   — total error events across the session
      correction_log      — ordered list of {tool, error_kind, attempt}
    """

    def __init__(self, max_consecutive: int = MAX_CONSECUTIVE_ERRORS):
        self.max_consecutive    = max_consecutive
        self.consecutive_errors = 0
        self.total_corrections  = 0
        self.correction_log: list[dict] = []

    # ─────────────────────────────────────────────────── public API

    def assess(self, tool_name: str, result: str) -> tuple[bool, str]:
        """
        Assess a tool result for errors.

        Returns (is_error, enriched_result).
        - If not an error: resets the consecutive counter, returns the
          result unchanged.
        - If an error: increments counters, classifies, injects hint,
          returns the enriched result for Claude to read.
        """
        if not self._is_error(tool_name, result):
            self.consecutive_errors = 0   # success resets the run
            return False, result

        self.consecutive_errors += 1
        self.total_corrections  += 1

        kind      = self._classify(result)
        attempt   = self.consecutive_errors
        remaining = max(0, self.max_consecutive - attempt)

        self.correction_log.append({
            "tool":       tool_name,
            "error_kind": kind.value,
            "attempt":    attempt,
        })

        enriched = (
            f"{result}\n\n"
            f"━━ SELF-CORRECTION  [attempt {attempt}/{self.max_consecutive}"
            f" • {remaining} remaining] ━━\n"
            f"Error type : {kind.value}\n"
            f"Guidance   : {HINTS[kind]}"
        )
        return True, enriched

    def should_abort(self) -> bool:
        """True when the consecutive error budget is exhausted."""
        return self.consecutive_errors >= self.max_consecutive

    def last_error_kind(self) -> str:
        if self.correction_log:
            return self.correction_log[-1]["error_kind"]
        return "unknown"

    # ─────────────────────────────────────────────────── detection

    # All Python built-in exception names that appear as "ExcName:" or
    # "ExcName\n" in tracebacks.
    _PY_EXCEPTIONS = (
        "SyntaxError", "IndentationError", "TabError",
        "NameError", "UnboundLocalError",
        "TypeError", "ValueError", "AttributeError",
        "ImportError", "ModuleNotFoundError",
        "FileNotFoundError", "IsADirectoryError", "NotADirectoryError",
        "PermissionError", "FileExistsError",
        "RuntimeError", "RecursionError", "MemoryError",
        "ZeroDivisionError", "IndexError", "KeyError",
        "AssertionError", "StopIteration", "OverflowError",
    )

    def _is_error(self, tool_name: str, result: str) -> bool:
        # Python traceback — definitive
        if "Traceback (most recent call last)" in result:
            return True
        # Our run_tool exception wrapper: "Error running '<name>': ..."
        if result.startswith(f"Error running '{tool_name}'"):
            return True
        # Sandbox / file tool error prefix
        if result.startswith("Error: ") or result.startswith("Error executing"):
            return True
        # Non-zero shell exit from the sandbox (e.g. "[exit code 2]")
        if re.search(r"\[exit(?:\s+code)?\s+[1-9]\d*\]", result):
            return True
        # Timeout (both sandbox and subprocess fallback)
        if "timed out after" in result.lower():
            return True
        # Python exception class names appearing as in a traceback
        for exc in self._PY_EXCEPTIONS:
            if f"{exc}:" in result or f"{exc}\n" in result:
                return True
        return False

    # ─────────────────────────────────────────────────── classification

    def _classify(self, result: str) -> ErrorKind:
        if any(e in result for e in ("SyntaxError", "IndentationError", "TabError")):
            return ErrorKind.SYNTAX
        if any(e in result for e in ("ImportError", "ModuleNotFoundError", "No module named")):
            return ErrorKind.IMPORT
        if any(e in result for e in ("FileNotFoundError", "No such file or directory",
                                      "IsADirectoryError")):
            return ErrorKind.FILE
        if "timed out" in result.lower():
            return ErrorKind.TIMEOUT
        if "Traceback" in result or any(
            f"{e}:" in result
            for e in ("TypeError", "NameError", "AttributeError",
                       "ValueError", "RuntimeError", "KeyError",
                       "IndexError", "AssertionError")
        ):
            return ErrorKind.RUNTIME
        if re.search(r"\[exit(?:\s+code)?\s+[1-9]\d*\]", result):
            return ErrorKind.SHELL
        return ErrorKind.GENERIC
