"""Approval policy — which tool calls need a human "yes" before they run.

Interactive mode gates side-effecting tools on an explicit approval prompt;
headless mode auto-approves everything (that is the whole point of headless,
and matches ml-intern, where a bare prompt runs unattended). `/yolo` flips
interactive mode into auto-approve for the rest of the session.
"""

from __future__ import annotations

import json

# Tools that write to disk, execute code, spend money, or reach the network.
# Everything not listed here is treated as read-only and runs unprompted.
SIDE_EFFECTING_TOOLS: frozenset[str] = frozenset(
    {
        "write_file",
        "run_python",
        "run_shell",
        "install_package",
        "save_dataset",
        "engineer_features",
        "train_models",
        "tune_hyperparameters",
        "package_model",
        "connect_mcp_server",
        "disconnect_mcp_server",
        "prepare_finetune_dataset",
        "fine_tune",
        "merge_and_export_model",
        "solve_ml_task",
        "index_project",
        "index_pdf",
        "index_image",
        "index_audio",
        "load_sql",
        "load_cloud_data",
    }
)


def requires_approval(tool_name: str) -> bool:
    """True when `tool_name` mutates state or reaches outside the process."""
    return tool_name in SIDE_EFFECTING_TOOLS


def describe_operation(tool_name: str, tool_input: dict) -> str:
    """One-line, human-readable summary of what a call is about to do —
    shown in the approval prompt."""
    inp = tool_input or {}
    if tool_name == "write_file":
        return f"write {inp.get('path', '?')}"
    if tool_name == "run_shell":
        return str(inp.get("command", ""))[:160]
    if tool_name == "run_python":
        code = str(inp.get("code", ""))
        first = next((ln for ln in code.splitlines() if ln.strip()), "")
        return f"run python: {first[:120]}" + (" …" if len(code.splitlines()) > 1 else "")
    if tool_name == "install_package":
        return f"pip install {inp.get('packages', '?')}"
    try:
        return json.dumps(inp, ensure_ascii=False)[:160]
    except (TypeError, ValueError):
        return str(inp)[:160]


def is_scheduled_operation(operation: str | None) -> bool:
    """Check if an HF Jobs operation is a scheduled/recurring operation."""
    if not operation:
        return False
    scheduled_ops = {"create_schedule", "update_schedule", "delete_schedule"}
    return operation in scheduled_ops
