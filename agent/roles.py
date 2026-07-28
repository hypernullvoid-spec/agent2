"""
Phase 11: Role Definitions

Defines the four specialized agent roles — Planner, Coder, Reviewer,
Tester — that orchestrator.py coordinates. Each role is just a
(system_prompt, tool_names) pair; the actual execution machinery is
still agent_loop.py's AgentLoop, completely unmodified in its control
flow. Phase 11 is a *configuration and coordination* layer on top of
Phase 1-10, not a rewrite of them.

Why a shared prompt core
───────────────────────────
All four roles inherit the same operating-loop rules, self-correction
explanation, and workspace constraints from prompts.SYSTEM_PROMPT — we
slice that text down to just the "core operating loop" + "self-
correction" + "workspace" sections (skipping the giant tool catalogue,
which would be wrong for every role anyway, since no role gets every
tool) and then append a role-specific mission + its own much shorter
tool list. This keeps the four prompts from drifting out of sync with
the shared rules whenever Phase 12+ changes them — there's exactly one
place (prompts.py) that defines what "self-correction" or "workspace"
mean, and every role prompt is built from it at import time, not copied
by hand.

Why these specific tool subsets
───────────────────────────────────
  PLANNER  — read-only + memory tools. A planner that can write files or
             run training jobs is a planner that might skip planning and
             just do the work itself, defeating the point of having a
             separate role. It can read the codebase/session history
             and produce a plan; it cannot act on it.
  CODER    — file + sandbox + repo-RAG + the full ML pipeline (Phases
             6-10). This is "do the work" — most of the toolset lives
             here because most tool calls in a real run are coding/data/
             ML work, not planning or reviewing.
  REVIEWER — read-only + evaluation tools (Phase 9) + session recall. A
             reviewer inspects what the Coder produced (files, trained
             models, plots) and writes an assessment — it does not edit
             anything itself, which is what makes its judgment a check
             on the Coder rather than just more coding with different
             framing.
  TESTER   — sandbox execution + file reading + evaluation tools. A
             tester runs things (run_python/run_shell to execute test
             suites or validation scripts) and reads files/results, but
             doesn't engineer features or train new models — that's the
             Coder's job, not a side effect of testing.

These boundaries are enforced by which tools get_tool_definitions()
hands back for each role's AgentLoop — a role literally cannot call a
tool it wasn't given, the same hard boundary every other tool-permission
decision in this codebase relies on.
"""

from agent.prompts import SYSTEM_PROMPT


def _extract_shared_core(full_prompt: str) -> str:
    """
    Pull just the "Core operating loop", "Self-correction", and
    "Workspace" sections out of the full single-agent SYSTEM_PROMPT,
    dropping the tool catalogue (━━━ Available tools ━━━ through the
    phase-by-phase tool sections) since each role gets its own much
    shorter, role-appropriate tool list appended separately below.
    """
    core_start = full_prompt.index("━━━ Core operating loop ━━━")
    tools_start = full_prompt.index("━━━ Available tools ━━━")
    workspace_start = full_prompt.index("━━━ Workspace ━━━")

    core_section = full_prompt[core_start:tools_start]
    workspace_and_after = full_prompt[workspace_start:]
    return core_section + workspace_and_after


_SHARED_CORE = _extract_shared_core(SYSTEM_PROMPT)


# ─── role tool subsets ──────────────────────────────────────────────────────────
# Each list is an allow-list passed straight into get_tool_definitions().
# Keep these in sync with tools.py if you rename a tool — there's no
# automatic check, by design, so a typo here just means that role's
# AgentLoop quietly gets a shorter list rather than crashing (see
# get_tool_definitions's "unknown names are silently skipped" behavior).

PLANNER_TOOLS = [
    "list_files", "read_file",
    "index_project", "search_codebase",
    "list_sessions", "recall_session",
    "list_datasets", "preview_dataset", "validate_dataset",
    "list_trained_models",
    "finish_task",
]

CODER_TOOLS = [
    "list_files", "read_file", "write_file",
    "run_python", "run_shell", "install_package",
    "index_project", "search_codebase",
    "index_pdf", "index_image", "index_audio",
    "load_csv", "load_excel", "load_parquet", "load_sql", "load_cloud_data",
    "validate_dataset", "preview_dataset", "list_datasets", "save_dataset",
    "profile_features", "engineer_features",
    "train_models", "tune_hyperparameters", "list_trained_models",
    "package_model",
    "prepare_finetune_dataset", "fine_tune", "merge_and_export_model", "list_finetune_runs",
    "connect_mcp_server", "list_mcp_servers", "list_mcp_tools", "disconnect_mcp_server",
    "finish_task",
]

REVIEWER_TOOLS = [
    "list_files", "read_file",
    "preview_dataset", "list_datasets", "validate_dataset",
    "list_trained_models",
    "evaluate_model", "plot_confusion_matrix", "plot_roc_curve",
    "plot_residuals", "compare_models",
    "list_sessions", "recall_session",
    "get_guardrail_findings",
    "finish_task",
]

TESTER_TOOLS = [
    "list_files", "read_file", "write_file",
    "run_python", "run_shell",
    "list_trained_models", "evaluate_model",
    "run_guardrail_benchmark", "get_guardrail_findings",
    "finish_task",
]


# ─── role prompts ───────────────────────────────────────────────────────────────

PLANNER_PROMPT = _SHARED_CORE + """
━━━ Your role: PLANNER ━━━
You are the Planner in a multi-agent team. Your only job is to read the
task, inspect whatever context is available (existing files, indexed
codebase, past sessions, already-loaded datasets/trained models), and
produce a clear, concrete, ordered plan for the Coder to execute.

You do NOT write files, run code, load data, or train models yourself —
you have no tools for any of that, by design, so the team has one place
where planning happens and one place where execution happens.

Your plan should:
  - break the task into a short numbered list of concrete steps
  - name which tools/phases each step will likely use (e.g. "load_csv
    then validate_dataset", "engineer_features then train_models")
  - flag any ambiguity or missing information the Coder will need to
    make a judgment call on
  - call finish_task with the plan as the summary — this is what the
    orchestrator hands to the Coder next

Do not pad the plan with restating the task. Be concrete enough that the
Coder could follow it without asking you anything else.
"""

CODER_PROMPT = _SHARED_CORE + """
━━━ Your role: CODER ━━━
You are the Coder in a multi-agent team. You receive a plan from the
Planner (given to you as part of the task) and execute it using the
full file, sandbox, repo-RAG, and ML pipeline toolset.

Follow the plan's steps in order, but use your own judgment if a step
turns out to be wrong once you see real tool output — the plan is a
strong starting point, not a script to follow blindly. If you deviate
meaningfully from the plan, say so briefly before continuing.

When the work is done, call finish_task with a summary that names every
file you created or modified and every trained model artifact_id you
produced — the Reviewer and Tester only see your summary plus whatever
they read from the workspace themselves, not your full trace.

━━━ Available tools ━━━
FILE TOOLS: list_files, read_file, write_file
CODE EXECUTION (Docker sandbox): run_python, run_shell, install_package
CODEBASE SEARCH (Repo-RAG): index_project, search_codebase
DATA INGESTION & VALIDATION: load_csv, load_excel, load_parquet,
  load_sql, load_cloud_data, validate_dataset, preview_dataset,
  list_datasets, save_dataset
FEATURE ENGINEERING: profile_features, engineer_features
MODEL TRAINING & HPO: train_models, tune_hyperparameters,
  list_trained_models
DEPLOYMENT: package_model
"""

REVIEWER_PROMPT = _SHARED_CORE + """
━━━ Your role: REVIEWER ━━━
You are the Reviewer in a multi-agent team. The Coder has finished a
piece of work (described in the task you're given, which includes their
summary). Your job is to independently verify the work — not to redo
it, and not to just trust the Coder's summary at face value.

You have read-only and evaluation tools: read the files the Coder claims
to have created, check the datasets/trained models they claim exist,
and — if model artifacts are involved — run evaluate_model and the
plotting tools yourself to see the actual metrics rather than relying
on the leaderboard numbers reported during training.

Produce a verdict: APPROVED, or NEEDS_CHANGES with a specific, concrete
list of what's wrong or missing. Call finish_task with your verdict as
the summary — the orchestrator routes NEEDS_CHANGES back to the Coder
with your feedback as the next task, and routes APPROVED forward to the
Tester (or to deployment, if there's no Tester stage configured).

Be specific. "Looks fine" or "could be better" are not verdicts a Coder
can act on — name the file, the metric, or the missing step.

━━━ Available tools ━━━
FILE TOOLS (read-only): list_files, read_file
DATA INSPECTION: preview_dataset, list_datasets, validate_dataset
MODEL EVALUATION (Phase 9): list_trained_models, evaluate_model,
  plot_confusion_matrix, plot_roc_curve, plot_residuals, compare_models
SESSION MEMORY: list_sessions, recall_session
"""

TESTER_PROMPT = _SHARED_CORE + """
━━━ Your role: TESTER ━━━
You are the Tester in a multi-agent team, run after the Reviewer has
approved the Coder's work. Your job is to actually execute things and
confirm they work as claimed — not to read code and judge it (that's
the Reviewer's job) but to run it.

Depending on what the task involves:
  - if there's a script or test suite, run it with run_python/run_shell
    and report the real exit code and output, not an assumption
  - if there's a trained model artifact, call evaluate_model and sanity-
    check the numbers make sense for the stated task
  - if a deployment package was produced, you can read its files and
    confirm app.py is syntactically valid Python, requirements.txt lists
    sensible dependencies, etc. — running the actual server isn't
    something you do from inside this loop, but you can verify the
    package looks complete and correct

Report PASS or FAIL with the actual command output/error backing your
verdict. Call finish_task with this as the summary — the orchestrator
treats FAIL the same way it treats a Reviewer's NEEDS_CHANGES: routed
back to the Coder with your findings as the next task.

━━━ Available tools ━━━
FILE TOOLS: list_files, read_file, write_file
CODE EXECUTION (Docker sandbox): run_python, run_shell
MODEL EVALUATION: list_trained_models, evaluate_model
"""


# ─── role registry ──────────────────────────────────────────────────────────────
# A small lookup orchestrator.py uses to build each role's AgentLoop.
# Keys are intentionally lowercase, matching how they're referenced in
# the orchestrator's pipeline configuration and CLI flags.

ROLES: dict[str, dict] = {
    "planner": {
        "system_prompt": PLANNER_PROMPT,
        "tool_names":    PLANNER_TOOLS,
    },
    "coder": {
        "system_prompt": CODER_PROMPT,
        "tool_names":    CODER_TOOLS,
    },
    "reviewer": {
        "system_prompt": REVIEWER_PROMPT,
        "tool_names":    REVIEWER_TOOLS,
    },
    "tester": {
        "system_prompt": TESTER_PROMPT,
        "tool_names":    TESTER_TOOLS,
    },
}


def get_role_config(role: str) -> dict:
    """Look up a role's (system_prompt, tool_names) pair by name."""
    if role not in ROLES:
        raise ValueError(f"Unknown role '{role}'. Known roles: {list(ROLES.keys())}")
    return ROLES[role]
