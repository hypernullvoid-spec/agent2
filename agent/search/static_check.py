"""
Static gate — catch broken code BEFORE burning a sandbox execution.

Borrowed from opencode's "diagnostics before you run" pattern: a syntax
error, an obviously-missing metric print, or a forbidden pattern costs one
`compile()` call to detect here, versus a full sandbox round-trip (and, on
real ML tasks, minutes of training time) to detect at runtime. A gated node
is marked buggy immediately with a synthetic "execution output" describing
the problem, so the normal debug branch of the tree search fixes it — the
gate changes *when* the failure is caught, not how it's handled.
"""

from __future__ import annotations

import ast
import re
from typing import Optional

METRIC_PRINT_RE = re.compile(r"Final Validation Metric", re.IGNORECASE)


def static_check(code: str, require_metric_print: bool = True) -> Optional[str]:
    """Return a synthetic error string if the code is certain to fail
    (or certain to be scored buggy), else None.

    Only rejects things that are *guaranteed* problems — this must never
    veto a script that could have worked.
    """
    if not code.strip():
        return "StaticCheckError: empty script (no code block was produced)."

    # 1. Syntax — the big one. compile() is exact, instant, and free.
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        line = (code.splitlines()[e.lineno - 1].strip()
                if e.lineno and e.lineno <= len(code.splitlines()) else "")
        return (f"StaticCheckError: SyntaxError: {e.msg} (line {e.lineno})\n"
                f"    {line}\nFix the syntax and resubmit the complete script.")

    # 2. A solution that never prints the metric line will always be scored
    #    buggy — reject it one round-trip earlier.
    if require_metric_print and not METRIC_PRINT_RE.search(code):
        return ("StaticCheckError: the script never prints "
                "'Final Validation Metric: <number>'. Every solution must "
                "evaluate itself on a held-out split and print that exact line.")

    # 3. Cheap AST scans for guaranteed-fatal patterns.
    for node in ast.walk(tree):
        # `input()` blocks forever in a non-interactive sandbox
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "input"):
            return ("StaticCheckError: the script calls input(), which hangs "
                    "forever in the sandbox. Remove all interactive prompts.")
    return None
