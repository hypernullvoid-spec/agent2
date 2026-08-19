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
        # writes to disk / executes code / installs
        "write_file",
        "write_report",
        "run_python",
        "run_shell",
        "install_package",
        # dataset mutation and persistence
        "save_dataset",
        "clean_dataset",
        "apply_cleaning",
        "engineer_features",
        # model lifecycle
        "train_models",
        "tune_hyperparameters",
        "save_model",
        "delete_model",
        "package_model",
        "predict",
        "solve_ml_task",
        # fine-tuning
        "prepare_finetune_dataset",
        "fine_tune",
        "merge_and_export_model",
        # plots — each one renders a PNG into the workspace
        "plot_column",
        "plot_relationship",
        "plot_confusion_matrix",
        "plot_residuals",
        "plot_roc_curve",
        # indexing — writes an index to disk and may reach the network
        "index_project",
        "index_pdf",
        "index_image",
        "index_audio",
        # document capabilities that write JSON/CSV/annotated images
        "swarn_doc_ingest",
        "swarn_pdf_to_csv",
        "extract_pdf_structured",
        # network / external processes
        "connect_mcp_server",
        "disconnect_mcp_server",
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
    if tool_name in ("save_dataset", "clean_dataset", "apply_cleaning"):
        return f"{tool_name.replace('_', ' ')} {inp.get('name', inp.get('path', '?'))}"
    if tool_name in ("swarn_doc_ingest", "swarn_pdf_to_csv", "extract_pdf_structured"):
        return f"{tool_name.replace('_', ' ')} {inp.get('path', '?')}"
    if tool_name.startswith("plot_"):
        return f"render {tool_name[5:]} plot -> workspace"
    if tool_name in ("train_models", "tune_hyperparameters"):
        return f"{tool_name.replace('_', ' ')} on {inp.get('dataset', '?')}"
    try:
        return json.dumps(inp, ensure_ascii=False)[:160]
    except (TypeError, ValueError):
        return str(inp)[:160]

