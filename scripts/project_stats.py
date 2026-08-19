#!/usr/bin/env python3
"""
Regenerate the headline numbers in PROJECT_STATUS.md.

Counts source lines by area, registered agent tools, CLI commands, HTTP
endpoints and test functions — so the status report can be re-verified
instead of trusted. Reads the repo only; runs no tests and makes no
network calls.

    python scripts/project_stats.py            # human-readable table
    python scripts/project_stats.py --json     # machine-readable
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}

AREAS = {
    "agent/ (core platform)": ["agent"],
    "swarn/ (capabilities)": ["swarn"],
    "tests/": ["tests"],
    "examples/ + root": ["examples", "."],
}


def py_files(rel: str) -> list[Path]:
    base = ROOT / rel
    if rel == ".":
        return sorted(p for p in base.glob("*.py"))
    if not base.is_dir():
        return []
    return sorted(
        p for p in base.rglob("*.py")
        if not (SKIP_DIRS & set(p.relative_to(ROOT).parts))
    )


def lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def collect() -> dict:
    per_area, all_files = {}, []
    for label, rels in AREAS.items():
        files = [f for rel in rels for f in py_files(rel)]
        per_area[label] = {"files": len(files), "lines": sum(lines(f) for f in files)}
        all_files += files

    tools = sorted(_registered_tools())
    cli = (ROOT / "agent" / "cli.py").read_text(encoding="utf-8")
    dash_path = ROOT / "agent" / "dashboard.py"
    dash = dash_path.read_text(encoding="utf-8") if dash_path.exists() else ""

    test_files = py_files("tests")
    per_suite = {}
    for f in test_files:
        n = len(re.findall(r"^def test_", f.read_text(encoding="utf-8"), re.M))
        if n:
            per_suite[f.name] = n

    return {
        "python_files": len(all_files),
        "python_lines": sum(v["lines"] for v in per_area.values()),
        "by_area": per_area,
        "agent_tools": {"count": len(tools), "names": tools},
        "cli_commands": len(re.findall(r"^@app\.command\(", cli, re.M)),
        "http_endpoints": len(re.findall(r"^@app\.(get|post|websocket)\(", dash, re.M)),
        "tests": {"total": sum(per_suite.values()), "by_suite": dict(sorted(
            per_suite.items(), key=lambda kv: -kv[1]))},
    }


def _registered_tools() -> list[str]:
    """Prefer the live registry; fall back to parsing @tool decorators."""
    sys.path.insert(0, str(ROOT))
    try:
        from agent.runtime.tools import TOOL_REGISTRY  # noqa: PLC0415
        return list(TOOL_REGISTRY)
    except Exception:  # noqa: BLE001 — deps may not be installed
        src = (ROOT / "agent" / "tools.py").read_text(encoding="utf-8")
        return re.findall(r"^@tool[^\n]*\n(?:@[^\n]*\n)*def (\w+)", src, re.M)


def main() -> int:
    stats = collect()
    if "--json" in sys.argv:
        print(json.dumps(stats, indent=2))
        return 0

    print(f"\nSwarn — project stats  ({ROOT})\n")
    print(f"  Python source     {stats['python_lines']:>7,} lines "
          f"across {stats['python_files']} files")
    for label, v in stats["by_area"].items():
        print(f"    {label:<26} {v['lines']:>7,} lines  ({v['files']} files)")
    print()
    print(f"  Agent tools       {stats['agent_tools']['count']:>7}")
    print(f"  CLI commands      {stats['cli_commands']:>7}")
    print(f"  HTTP endpoints    {stats['http_endpoints']:>7}")
    print(f"  Tests             {stats['tests']['total']:>7}")
    for name, n in stats["tests"]["by_suite"].items():
        print(f"    {name:<32} {n:>4}")
    print("\n  Run `python tests/run_tests.py` for pass/fail.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
