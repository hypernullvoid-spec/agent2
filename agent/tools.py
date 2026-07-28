"""
Tool registry (Phase 1 + Phase 2 + Phase 3)

One @tool decorator registers any Python function as an agent-callable tool.
get_tool_definitions() converts the registry into Anthropic's `tools` format.
run_tool() dispatches a tool_use block back to the matching function.

Phase breakdown:
  Phase 1  list_files, read_file, write_file, finish_task
           run_python (subprocess, now replaced below)
  Phase 2  run_python  ← now backed by sandbox.py (Docker)
           run_shell   ← new: arbitrary shell in sandbox
           install_package ← convenience: pip install inside sandbox
  Phase 3  index_project  ← walk a directory, embed, persist to ChromaDB
           search_codebase ← semantic nearest-neighbour lookup
  Phase 6  load_csv, load_excel, load_parquet, load_sql, load_cloud_data
           validate_dataset, preview_dataset, list_datasets, save_dataset
           ← backed by data_pipeline.py
  Phase 7  profile_features, engineer_features
           ← backed by feature_engineering.py
  Phase 8  train_models, tune_hyperparameters, list_trained_models
           ← backed by model_training.py
  Phase 9  evaluate_model, plot_confusion_matrix, plot_roc_curve,
           plot_residuals, compare_models
           ← backed by evaluation.py
  Phase 10 package_model
           ← backed by deployment.py
  Phase 12 connect_mcp_server, list_mcp_servers, list_mcp_tools,
           disconnect_mcp_server
           ← backed by mcp_integration.py. Connecting a server also
             dynamically registers EVERY tool that server exposes,
             right into this same TOOL_REGISTRY, under the name
             "mcp_<server_name>_<tool_name>" — so once connected, those
             tools show up in get_tool_definitions() and are callable
             via run_tool() exactly like every tool defined in this
             file. (Phase 11's multi-agent roles live in roles.py +
             orchestrator.py, not here — they reuse this same registry,
             just with a restricted allow-list per role.)
  Phase 13 index_pdf, index_image, index_audio
           ← backed by multimodal_rag.py. These feed the SAME ChromaDB
             collection Phase 3's index_project/search_codebase use —
             search_codebase (unmodified) returns a blend of code, PDF
             text/tables, image OCR/captions, and audio transcripts,
             ranked purely by relevance, with the chunk `type` field
             telling you which modality a given result came from.
  Phase 14 prepare_finetune_dataset, fine_tune, merge_and_export_model,
           list_finetune_runs
           ← backed by finetuning.py. LoRA/QLoRA fine-tuning of a small
             LOCAL open-weight model (not Claude — Claude is accessed
             via the API and isn't fine-tuned by this project). Useful
             for a cheap, specialized model handling a narrow, repetitive
             subtask once the pattern is well-established, per the
             original blueprint.
  Phase 15 run_guardrail_benchmark, get_guardrail_findings
           ← backed by observability.py. GuardrailPolicy itself isn't a
             tool — it's wired directly into agent_loop.py, scanning
             every tool result for prompt-injection patterns before
             Claude sees it (see agent_loop.py's Phase 15 notes for why
             ordering relative to Phase 4's correction policy matters).
             ObservabilityHooks similarly isn't a tool — it's passed
             into AgentLoop the same optional way correction_policy is,
             wrapping LLM/tool calls in OpenTelemetry spans. The two
             tools below are the agent-facing parts: running the canned
             benchmark suite, and inspecting what's been flagged so far.

agent_loop.py needs zero changes — it calls get_tool_definitions() and
run_tool() the same way it always has.
"""

import os

# ─── constants ────────────────────────────────────────────────────────────────

WORKSPACE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "workspace")
)
os.makedirs(WORKSPACE_DIR, exist_ok=True)


# ─── registry machinery ───────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, dict] = {}


def tool(description: str, schema: dict):
    """Decorator that registers a function as an agent-callable tool."""
    def decorator(func):
        TOOL_REGISTRY[func.__name__] = {
            "description": description,
            "schema":      schema,
            "func":        func,
        }
        return func
    return decorator


MCP_TOOL_PREFIX = "mcp_"   # see get_tool_definitions's note on dynamic MCP tool visibility


def get_tool_definitions(names: "list[str] | None" = None) -> list[dict]:
    """
    Return registered tools in Anthropic's `tools` API format.

    names: optional allow-list of tool names. Omit to get every
    registered tool (the original Phase 1–10 behavior, still the
    default everywhere it's already called). Phase 11 passes a
    role-specific subset here — e.g. the Planner role only needs
    list_files/read_file/list_sessions, not run_python or train_models —
    so each role sees a smaller, more relevant tool list instead of all
    31+ tools every single role would otherwise have to consider.
    Unknown names are silently skipped rather than raising, since a role
    config listing a tool that doesn't happen to be registered shouldn't
    crash the whole orchestrator.

    Dynamic MCP tool visibility (Phase 12 + Phase 11 interaction):
    connect_mcp_server registers new tools into TOOL_REGISTRY *while a
    role is already running* — their names (mcp_<server>_<tool>) cannot
    possibly appear in a role's static tool_names list, since that list
    was written before any server was ever connected. Without special
    handling, a role would be able to call connect_mcp_server but then
    never see the tools it just registered — a result the rest of this
    codebase very deliberately doesn't want (see mcp_integration.py's
    "no special-casing anywhere else" design goal).

    The fix: if names is given AND it includes "connect_mcp_server" —
    i.e. this role is explicitly permitted to manage MCP connections —
    then every currently-registered tool whose name starts with "mcp_"
    is included automatically, on top of the explicit allow-list. A
    role that was never given connect_mcp_server in the first place
    (e.g. the Reviewer) still only sees exactly its static list; this
    only widens visibility for roles already trusted to open
    connections, which is exactly the trust boundary that should govern
    whether they can also use what those connections produce.
    """
    if names is None:
        selected = TOOL_REGISTRY.items()
    elif "connect_mcp_server" in names:
        wanted = set(names)
        selected = (
            (n, meta) for n, meta in TOOL_REGISTRY.items()
            if n in wanted or n.startswith(MCP_TOOL_PREFIX)
        )
    else:
        selected = ((n, TOOL_REGISTRY[n]) for n in names if n in TOOL_REGISTRY)
    return [
        {
            "name":         name,
            "description":  meta["description"],
            "input_schema": meta["schema"],
        }
        for name, meta in selected
    ]


def run_tool(name: str, tool_input: dict) -> str:
    """
    Execute a registered tool by name.
    Never raises — errors are returned as strings so the agent can see
    and react to them (the hook Phase 4's self-correction loop builds on).
    """
    if name not in TOOL_REGISTRY:
        return f"Error: unknown tool '{name}'"
    try:
        return TOOL_REGISTRY[name]["func"](**tool_input)
    except Exception as e:
        return f"Error running '{name}': {type(e).__name__}: {e}"


# ─── path guard ───────────────────────────────────────────────────────────────

def _safe_path(path: str) -> str:
    """Resolve a relative path inside WORKSPACE_DIR. Rejects path traversal."""
    full = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not (full == WORKSPACE_DIR or full.startswith(WORKSPACE_DIR + os.sep)):
        raise ValueError(f"Path '{path}' escapes the workspace directory")
    return full


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 TOOLS — file I/O and task completion
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description="List files and subdirectories inside the agent workspace.",
    schema={
        "type": "object",
        "properties": {
            "path": {
                "type":        "string",
                "description": "Relative path inside the workspace. Use '.' for the root.",
            }
        },
        "required": ["path"],
    },
)
def list_files(path: str = ".") -> str:
    target = _safe_path(path)
    if not os.path.exists(target):
        return f"Path not found: {path}"
    entries = sorted(os.listdir(target))
    return "\n".join(entries) if entries else "(empty directory)"


@tool(
    description="Read the full text content of a file inside the workspace.",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path inside the workspace."}
        },
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    target = _safe_path(path)
    if not os.path.exists(target):
        return f"File not found: {path}"
    with open(target, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


@tool(
    description=(
        "Write content to a file inside the workspace, creating parent "
        "directories as needed. Overwrites the file if it already exists."
    ),
    schema={
        "type": "object",
        "properties": {
            "path":    {"type": "string", "description": "Relative path inside the workspace."},
            "content": {"type": "string", "description": "Full text content to write."},
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str) -> str:
    target = _safe_path(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content):,} characters to {path}"


@tool(
    description=(
        "Call this ONLY when the task is fully complete. "
        "Provide a short summary of what was done and any output files. "
        "This signal ends the current task run."
    ),
    schema={
        "type": "object",
        "properties": {
            "summary": {
                "type":        "string",
                "description": "Concise summary of completed work and resulting files.",
            }
        },
        "required": ["summary"],
    },
)
def finish_task(summary: str) -> str:
    return f"TASK_COMPLETE: {summary}"


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 TOOLS — sandboxed code and shell execution
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Execute Python source code in a sandboxed Docker container and return "
        "its stdout + stderr. The sandbox has the workspace mounted at /workspace "
        "and a per-call timeout (default 300s, SWARN_EXEC_TIMEOUT). Falls back to subprocess if Docker is "
        "unavailable. For one-off scripts, prefer this over run_shell."
    ),
    schema={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to execute."}
        },
        "required": ["code"],
    },
)
def run_python(code: str) -> str:
    from agent.sandbox import get_sandbox
    return get_sandbox().exec_python(code)


@tool(
    description=(
        "Execute any shell command in the sandboxed Docker container "
        "(bash, git, curl, pip, etc.). Requires Docker. Use this to install "
        "packages, run scripts, or inspect the environment. "
        "Workspace is at /workspace inside the container."
    ),
    schema={
        "type": "object",
        "properties": {
            "command": {
                "type":        "string",
                "description": "Shell command to run (executed via bash -c).",
            }
        },
        "required": ["command"],
    },
)
def run_shell(command: str) -> str:
    from agent.sandbox import get_sandbox
    return get_sandbox().exec_shell(command)


@tool(
    description=(
        "Install one or more Python packages inside the sandbox using pip. "
        "Packages persist for the duration of this agent session. "
        "Example: install_package('pandas scikit-learn matplotlib')"
    ),
    schema={
        "type": "object",
        "properties": {
            "packages": {
                "type":        "string",
                "description": "Space-separated list of package names (e.g. 'numpy pandas').",
            }
        },
        "required": ["packages"],
    },
)
def install_package(packages: str) -> str:
    from agent.sandbox import get_sandbox
    cmd = f"pip install {packages} --quiet"
    result = get_sandbox().exec_shell(cmd)
    # pip install success produces no stdout with --quiet; check stderr for errors
    if "ERROR" in result.upper() or "exit 1" in result.lower():
        return f"Install may have failed:\n{result}"
    return f"Installed: {packages}\n{result}" if result.strip() else f"✓ Installed: {packages}"


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 TOOLS — project context and semantic search
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Index a project directory so you can search it semantically. "
        "Walks the directory recursively, chunks Python files by function/class "
        "(using the AST) and other files by fixed-size sliding window, embeds "
        "every chunk locally, and persists to a ChromaDB vector store. "
        "Call this BEFORE search_codebase. Re-calling updates the index. "
        "Use '.' to index the workspace, or an absolute path for an external project."
    ),
    schema={
        "type": "object",
        "properties": {
            "directory": {
                "type":        "string",
                "description": (
                    "Path to the directory to index. Use '.' for the agent workspace, "
                    "or supply an absolute path to an external project."
                ),
            }
        },
        "required": ["directory"],
    },
)
def index_project(directory: str) -> str:
    from agent.context_engine import get_context_engine

    # Resolve relative paths against the workspace
    if not os.path.isabs(directory):
        directory = os.path.join(WORKSPACE_DIR, directory)

    return get_context_engine().index_directory(directory)


@tool(
    description=(
        "Search the indexed codebase using a natural-language query. "
        "Returns the most semantically relevant code/doc chunks with file paths "
        "and line numbers. You must call index_project first. "
        "Use this to find: which file handles X, where Y is defined, "
        "how Z is implemented — before reading files directly."
    ),
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type":        "string",
                "description": "Natural-language description of what you're looking for.",
            },
            "n_results": {
                "type":        "integer",
                "description": "Number of results to return (default 6, max 20).",
                "default":     6,
            },
        },
        "required": ["query"],
    },
)
def search_codebase(query: str, n_results: int = 6) -> str:
    from agent.context_engine import get_context_engine
    return get_context_engine().search(query, n_results=min(n_results, 20))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 TOOLS — session memory and recall
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "List recent agent sessions with their outcome, duration, number of "
        "tool calls, and self-correction count. Useful for understanding what "
        "the agent has done before and whether similar tasks were completed "
        "successfully. Use recall_session to dig into a specific run."
    ),
    schema={
        "type": "object",
        "properties": {
            "n": {
                "type":        "integer",
                "description": "How many recent sessions to show (default 10).",
                "default":     10,
            }
        },
        "required": [],
    },
)
def list_sessions(n: int = 10) -> str:
    from agent.memory import get_session_store
    return get_session_store().list_sessions(n=n)


@tool(
    description=(
        "Recall the detailed tool-call log of a past session by its 8-character "
        "ID prefix (visible in list_sessions output). Shows every tool call "
        "that was made, in order, so you can understand what worked before or "
        "continue from where a prior session left off."
    ),
    schema={
        "type": "object",
        "properties": {
            "session_id": {
                "type":        "string",
                "description": "8-character session ID prefix from list_sessions.",
            }
        },
        "required": ["session_id"],
    },
)
def recall_session(session_id: str) -> str:
    from agent.memory import get_session_store
    return get_session_store().recall_as_text(session_id)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 TOOLS — data ingestion and validation
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Load a CSV file from the workspace into the data pipeline's dataset "
        "registry under the given name, so it can be validated, profiled, "
        "feature-engineered, and trained on. Always call validate_dataset "
        "right after loading."
    ),
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the CSV inside the workspace."},
            "name": {"type": "string", "description": "Name to register this dataset under (e.g. 'train')."},
        },
        "required": ["path", "name"],
    },
)
def load_csv(path: str, name: str) -> str:
    from agent.data_pipeline import get_data_pipeline
    return get_data_pipeline().load_csv(path, name)


@tool(
    description=(
        "Load an Excel file (.xlsx/.xls) from the workspace into the dataset "
        "registry. If the file has multiple sheets and sheet_name is omitted, "
        "returns the list of sheet names instead of loading — call again with "
        "sheet_name set."
    ),
    schema={
        "type": "object",
        "properties": {
            "path":       {"type": "string", "description": "Relative path to the Excel file inside the workspace."},
            "name":       {"type": "string", "description": "Name to register this dataset under."},
            "sheet_name": {"type": "string", "description": "Sheet to load. Omit to load the first/only sheet."},
        },
        "required": ["path", "name"],
    },
)
def load_excel(path: str, name: str, sheet_name: str = None) -> str:
    from agent.data_pipeline import get_data_pipeline
    kwargs = {"sheet_name": sheet_name} if sheet_name is not None else {}
    return get_data_pipeline().load_excel(path, name, **kwargs)


@tool(
    description="Load a Parquet file from the workspace into the dataset registry.",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the Parquet file inside the workspace."},
            "name": {"type": "string", "description": "Name to register this dataset under."},
        },
        "required": ["path", "name"],
    },
)
def load_parquet(path: str, name: str) -> str:
    from agent.data_pipeline import get_data_pipeline
    return get_data_pipeline().load_parquet(path, name)


@tool(
    description=(
        "Run a SQL query against a database and load the result into the "
        "dataset registry. connection_string follows SQLAlchemy URL format "
        "(e.g. 'postgresql://user:pass@host:5432/dbname' or 'sqlite:///file.db')."
    ),
    schema={
        "type": "object",
        "properties": {
            "connection_string": {"type": "string", "description": "SQLAlchemy-style database connection URL."},
            "query":             {"type": "string", "description": "SQL query to execute."},
            "name":              {"type": "string", "description": "Name to register the result under."},
        },
        "required": ["connection_string", "query", "name"],
    },
)
def load_sql(connection_string: str, query: str, name: str) -> str:
    from agent.data_pipeline import get_data_pipeline
    return get_data_pipeline().load_sql(connection_string, query, name)


@tool(
    description=(
        "Load a CSV or Parquet object directly from S3 (s3://...) or GCS "
        "(gs://...) into the dataset registry. Requires cloud credentials to "
        "already be configured in the environment."
    ),
    schema={
        "type": "object",
        "properties": {
            "uri":  {"type": "string", "description": "s3:// or gs:// URI of the object to load."},
            "name": {"type": "string", "description": "Name to register this dataset under."},
        },
        "required": ["uri", "name"],
    },
)
def load_cloud_data(uri: str, name: str) -> str:
    from agent.data_pipeline import get_data_pipeline
    return get_data_pipeline().load_cloud_storage(uri, name)


@tool(
    description=(
        "Profile a loaded dataset for data-quality issues: dtypes, null "
        "counts/percentages, duplicate rows, numeric outliers (z-score "
        "method), and a schema check via pandera if installed. ALWAYS call "
        "this right after loading a dataset and before engineer_features."
    ),
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of a dataset already in the registry."}
        },
        "required": ["name"],
    },
)
def validate_dataset(name: str) -> str:
    from agent.data_pipeline import get_data_pipeline
    return get_data_pipeline().validate_dataset(name)


@tool(
    description="Preview the first rows of a loaded dataset to sanity-check its contents.",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of a dataset already in the registry."},
            "n":    {"type": "integer", "description": "Number of rows to preview (default 10).", "default": 10},
        },
        "required": ["name"],
    },
)
def preview_dataset(name: str, n: int = 10) -> str:
    from agent.data_pipeline import get_data_pipeline
    return get_data_pipeline().preview_dataset(name, n=n)


@tool(
    description="List all datasets currently loaded in the data pipeline registry, with their shapes.",
    schema={"type": "object", "properties": {}, "required": []},
)
def list_datasets() -> str:
    from agent.data_pipeline import get_data_pipeline
    return get_data_pipeline().list_datasets()


@tool(
    description=(
        "Save a loaded/transformed dataset from the registry to a file in "
        "the workspace (CSV or Parquet, inferred from the path's extension)."
    ),
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of a dataset already in the registry."},
            "path": {"type": "string", "description": "Relative path to save to inside the workspace."},
        },
        "required": ["name", "path"],
    },
)
def save_dataset(name: str, path: str) -> str:
    from agent.data_pipeline import get_data_pipeline
    return get_data_pipeline().save_dataset(name, path)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7 TOOLS — automated feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Profile a dataset's columns to decide feature-engineering roles: "
        "numeric vs categorical (low/high cardinality) vs datetime vs "
        "constant/ID columns to drop. If target_col is given, also infers "
        "the likely task type (regression / binary / multiclass "
        "classification). Call this BEFORE engineer_features so you can "
        "pass an informed drop_cols list."
    ),
    schema={
        "type": "object",
        "properties": {
            "name":       {"type": "string", "description": "Name of a dataset already in the registry."},
            "target_col": {"type": "string", "description": "Optional: the prediction target column."},
        },
        "required": ["name"],
    },
)
def profile_features(name: str, target_col: str = None) -> str:
    from agent.feature_engineering import get_feature_engine
    return get_feature_engine().profile_dataset(name, target_col=target_col)


@tool(
    description=(
        "Transform a dataset into a model-ready numeric feature matrix: "
        "imputes + scales numeric columns, one-hot encodes low-cardinality "
        "categoricals, frequency-encodes high-cardinality categoricals, and "
        "decomposes datetime columns into year/month/day/dayofweek/is_weekend. "
        "Registers the result under a new dataset name (default "
        "'<name>_features') ready for train_models. Carries target_col "
        "through unchanged if given."
    ),
    schema={
        "type": "object",
        "properties": {
            "name":        {"type": "string", "description": "Name of the source dataset in the registry."},
            "target_col":  {"type": "string", "description": "Prediction target column to preserve and exclude from transformation."},
            "drop_cols":   {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns to drop before transforming (e.g. ID columns flagged by profile_features).",
            },
            "output_name": {"type": "string", "description": "Name for the resulting dataset (default '<name>_features')."},
        },
        "required": ["name"],
    },
)
def engineer_features(name: str, target_col: str = None, drop_cols: list = None, output_name: str = None) -> str:
    from agent.feature_engineering import get_feature_engine
    return get_feature_engine().engineer_features(
        name, target_col=target_col, drop_cols=drop_cols, output_name=output_name
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8 TOOLS — model training and hyperparameter optimization
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Train and evaluate candidate models on a feature-engineered dataset "
        "(must be all-numeric — run engineer_features first). Auto-detects "
        "regression vs binary vs multiclass classification from the target "
        "column and tries sensible candidates (linear/logistic, random "
        "forest, XGBoost, LightGBM, a small PyTorch MLP) unless a subset is "
        "specified. Returns a leaderboard ranked by the appropriate metric "
        "(RMSE for regression, accuracy for classification) and keeps the "
        "best model in memory as a trained artifact for later evaluation/"
        "deployment phases."
    ),
    schema={
        "type": "object",
        "properties": {
            "name":       {"type": "string", "description": "Name of the feature-engineered dataset in the registry."},
            "target_col": {"type": "string", "description": "Name of the target/label column within that dataset."},
            "candidates": {
                "type": "array",
                "items": {"type": "string", "enum": ["linear", "logistic", "random_forest", "xgboost", "lightgbm", "mlp"]},
                "description": "Subset of model families to try. Omit to try all available for the detected task type.",
            },
            "test_size": {"type": "number", "description": "Fraction of data held out for evaluation (default 0.2)."},
        },
        "required": ["name", "target_col"],
    },
)
def train_models(name: str, target_col: str, candidates: list = None, test_size: float = 0.2) -> str:
    from agent.model_training import get_model_trainer
    return get_model_trainer().train_models(name, target_col, candidates=candidates, test_size=test_size)


@tool(
    description=(
        "Run Optuna hyperparameter search (default 25 trials) over a fixed "
        "search space for one model family (random_forest, xgboost, or "
        "lightgbm) on the given dataset/target. Registers the best-found "
        "model as a trained artifact named '<name>__<candidate>_tuned'. "
        "Use after train_models has identified which family performs best."
    ),
    schema={
        "type": "object",
        "properties": {
            "name":       {"type": "string", "description": "Name of the feature-engineered dataset in the registry."},
            "target_col": {"type": "string", "description": "Name of the target/label column."},
            "candidate":  {
                "type": "string",
                "enum": ["random_forest", "xgboost", "lightgbm"],
                "description": "Which model family to tune.",
            },
            "n_trials": {"type": "integer", "description": "Number of Optuna trials to run (default 25)."},
        },
        "required": ["name", "target_col"],
    },
)
def tune_hyperparameters(name: str, target_col: str, candidate: str = "xgboost", n_trials: int = 25) -> str:
    from agent.model_training import get_model_trainer
    return get_model_trainer().tune_hyperparameters(name, target_col, candidate=candidate, n_trials=n_trials)


@tool(
    description="List all trained model artifacts currently held in memory, with their metrics.",
    schema={"type": "object", "properties": {}, "required": []},
)
def list_trained_models() -> str:
    from agent.model_training import get_model_trainer
    return get_model_trainer().list_trained_models()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 9 TOOLS — evaluation and visualization
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Full diagnostic report for a trained model artifact: re-evaluates "
        "it on its stored held-out test split with extra detail beyond the "
        "leaderboard's headline numbers — per-class precision/recall for "
        "classification, residual distribution stats for regression. Use "
        "this before deciding whether a model is good enough to deploy."
    ),
    schema={
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "Artifact key from train_models/tune_hyperparameters output (see list_trained_models).",
            }
        },
        "required": ["artifact_id"],
    },
)
def evaluate_model(artifact_id: str) -> str:
    from agent.evaluation import get_model_evaluator
    return get_model_evaluator().evaluate_model(artifact_id)


@tool(
    description=(
        "Generate and save a confusion matrix plot (PNG, in workspace/plots/) "
        "for a classification model artifact, showing where it confuses classes."
    ),
    schema={
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "description": "Artifact key from train_models/tune_hyperparameters."}
        },
        "required": ["artifact_id"],
    },
)
def plot_confusion_matrix(artifact_id: str) -> str:
    from agent.evaluation import get_model_evaluator
    return get_model_evaluator().plot_confusion_matrix(artifact_id)


@tool(
    description=(
        "Generate and save an ROC curve plot with AUC (PNG, in workspace/plots/) "
        "for a binary classification model artifact that supports predict_proba."
    ),
    schema={
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "description": "Artifact key from train_models/tune_hyperparameters."}
        },
        "required": ["artifact_id"],
    },
)
def plot_roc_curve(artifact_id: str) -> str:
    from agent.evaluation import get_model_evaluator
    return get_model_evaluator().plot_roc_curve(artifact_id)


@tool(
    description=(
        "Generate and save residual diagnostic plots (PNG, in workspace/plots/) "
        "— residuals-vs-predicted and residual distribution — for a regression "
        "model artifact. Look for a random scatter around 0 and a bell-shaped "
        "histogram; patterns indicate the model is missing structure in the data."
    ),
    schema={
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "description": "Artifact key from train_models/tune_hyperparameters."}
        },
        "required": ["artifact_id"],
    },
)
def plot_residuals(artifact_id: str) -> str:
    from agent.evaluation import get_model_evaluator
    return get_model_evaluator().plot_residuals(artifact_id)


@tool(
    description=(
        "Generate and save a bar-chart comparison (PNG, in workspace/plots/) of "
        "every trained model artifact currently in memory, grouped by task type "
        "and ranked by the appropriate primary metric. Use this to see all "
        "candidates and tuning runs side by side before picking one to deploy."
    ),
    schema={"type": "object", "properties": {}, "required": []},
)
def compare_models() -> str:
    from agent.evaluation import get_model_evaluator
    return get_model_evaluator().compare_models()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 10 TOOL — deployment automation
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Package a trained model artifact for deployment: serializes the "
        "model (pickle by default, or ONNX for sklearn-native estimators if "
        "skl2onnx is installed — falls back to pickle otherwise), generates "
        "a FastAPI service with a typed /predict endpoint and auto OpenAPI "
        "docs built from the model's feature columns, plus a requirements.txt "
        "and Dockerfile. Everything is written to "
        "workspace/deployments/<artifact_id>/. Call evaluate_model first to "
        "confirm the artifact is actually good enough to ship."
    ),
    schema={
        "type": "object",
        "properties": {
            "artifact_id":    {"type": "string", "description": "Artifact key from train_models/tune_hyperparameters."},
            "export_format":  {
                "type": "string",
                "enum": ["pickle", "onnx"],
                "description": "Serialization format. 'onnx' only applies to sklearn-native models and falls back to pickle otherwise (default: pickle).",
            },
            "api_title": {"type": "string", "description": "Optional title for the generated FastAPI service's OpenAPI docs."},
        },
        "required": ["artifact_id"],
    },
)
def package_model(artifact_id: str, export_format: str = "pickle", api_title: str = None) -> str:
    from agent.deployment import get_deployment_packager
    return get_deployment_packager().package_model(artifact_id, export_format=export_format, api_title=api_title)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 12 TOOLS — tool ecosystem & MCP integration
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Connect to an external MCP (Model Context Protocol) server as a "
        "subprocess and register every tool it exposes, immediately and "
        "automatically, under names like 'mcp_<server_name>_<tool_name>'. "
        "After this call succeeds, those remote tools work exactly like any "
        "other tool here — call them directly by name. command/args follow "
        "the same shape as Claude Desktop's MCP config, e.g. command='npx', "
        "args=['-y', '@modelcontextprotocol/server-github'], or for a local "
        "Python MCP server: command='python3', args=['my_server.py']."
    ),
    schema={
        "type": "object",
        "properties": {
            "server_name": {
                "type": "string",
                "description": "A short name you choose to identify this server (used as a prefix for its tool names).",
            },
            "command": {"type": "string", "description": "The executable to launch the server (e.g. 'npx', 'python3')."},
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Command-line arguments for launching the server.",
            },
        },
        "required": ["server_name", "command"],
    },
)
def connect_mcp_server(server_name: str, command: str, args: list = None) -> str:
    from agent.mcp_integration import get_mcp_manager
    return get_mcp_manager().connect_server(server_name, command, args=args)


@tool(
    description="List all currently connected MCP servers and how many tools each one exposes.",
    schema={"type": "object", "properties": {}, "required": []},
)
def list_mcp_servers() -> str:
    from agent.mcp_integration import get_mcp_manager
    return get_mcp_manager().list_mcp_servers()


@tool(
    description=(
        "List the tools currently registered from connected MCP servers "
        "(optionally filtered to one server). These tool names are what you "
        "call directly — e.g. 'mcp_github_create_issue' — once connected."
    ),
    schema={
        "type": "object",
        "properties": {
            "server_name": {"type": "string", "description": "Optional: only list tools from this server."},
        },
        "required": [],
    },
)
def list_mcp_tools(server_name: str = None) -> str:
    from agent.mcp_integration import get_mcp_manager
    return get_mcp_manager().list_mcp_tools(server_name=server_name)


@tool(
    description=(
        "Disconnect a previously connected MCP server and remove all of its "
        "tools from the registry. Call this before reconnecting the same "
        "server_name, or to free resources when you're done with it."
    ),
    schema={
        "type": "object",
        "properties": {
            "server_name": {"type": "string", "description": "Name of a server previously connected via connect_mcp_server."},
        },
        "required": ["server_name"],
    },
)
def disconnect_mcp_server(server_name: str) -> str:
    from agent.mcp_integration import get_mcp_manager
    return get_mcp_manager().disconnect_server(server_name)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 13 TOOLS — multi-modal RAG
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Index a PDF into the same searchable index Phase 3's index_project "
        "uses — extracts both prose text and tables (one chunk per table, "
        "preserving row/column structure). After this, search_codebase will "
        "return results from this PDF alongside any indexed code, with "
        "citations as (file, page) instead of (file, line range)."
    ),
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the PDF file (relative to the workspace, or absolute)."},
        },
        "required": ["path"],
    },
)
def index_pdf(path: str) -> str:
    from agent.multimodal_rag import get_multimodal_indexer
    if not os.path.isabs(path):
        path = os.path.join(WORKSPACE_DIR, path)
    return get_multimodal_indexer().index_pdf(path)


@tool(
    description=(
        "Index an image into the same searchable index Phase 3's "
        "index_project uses — OCR extracts any visible text, and (if a "
        "CLIP-family model is available) the image itself is embedded so "
        "purely visual content is searchable too. An optional caption is "
        "indexed alongside, useful for diagrams/photos where OCR alone "
        "wouldn't capture the intent."
    ),
    schema={
        "type": "object",
        "properties": {
            "path":    {"type": "string", "description": "Path to the image file (relative to the workspace, or absolute)."},
            "caption": {"type": "string", "description": "Optional human-written description of the image."},
        },
        "required": ["path"],
    },
)
def index_image(path: str, caption: str = None) -> str:
    from agent.multimodal_rag import get_multimodal_indexer
    if not os.path.isabs(path):
        path = os.path.join(WORKSPACE_DIR, path)
    return get_multimodal_indexer().index_image(path, caption=caption)


@tool(
    description=(
        "Transcribe an audio file (via openai-whisper) and index the "
        "transcript into the same searchable index Phase 3's index_project "
        "uses. Citations use timestamps (e.g. '12:34') instead of page/line "
        "numbers, so a search result tells you exactly where in the "
        "recording to find the relevant moment."
    ),
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the audio file (relative to the workspace, or absolute)."},
            "model_size": {
                "type": "string",
                "enum": ["tiny", "base", "small", "medium", "large"],
                "description": "Whisper model size — larger is slower but more accurate. Default 'base'.",
            },
        },
        "required": ["path"],
    },
)
def index_audio(path: str, model_size: str = "base") -> str:
    from agent.multimodal_rag import get_multimodal_indexer
    if not os.path.isabs(path):
        path = os.path.join(WORKSPACE_DIR, path)
    return get_multimodal_indexer().index_audio(path, model_size=model_size)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 14 TOOLS — LLM fine-tuning (LoRA / QLoRA, local open-weight models)
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Validate and save a list of {'prompt', 'completion'} examples as a "
        "fine-tuning dataset under a chosen run_id, splitting off a validation "
        "set automatically if there are 10+ examples. Catches malformed "
        "examples (missing fields, empty strings) before any training starts. "
        "This is NOT for fine-tuning Claude — it prepares data for fine-tuning "
        "a small local open-weight model via fine_tune."
    ),
    schema={
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "completion": {"type": "string"},
                    },
                    "required": ["prompt", "completion"],
                },
                "description": "List of {'prompt', 'completion'} training examples.",
            },
            "run_id": {"type": "string", "description": "A name for this fine-tuning run."},
            "validation_split": {
                "type": "number",
                "description": "Fraction held out for validation (default 0.1). Only applied with 10+ examples.",
            },
        },
        "required": ["examples", "run_id"],
    },
)
def prepare_finetune_dataset(examples: list, run_id: str, validation_split: float = 0.1) -> str:
    from agent.finetuning import get_fine_tuner
    return get_fine_tuner().prepare_dataset(examples, run_id, validation_split=validation_split)


@tool(
    description=(
        "Fine-tune a small local open-weight model (a HuggingFace model ID, "
        "e.g. a small Qwen/Llama variant — NOT Claude) on a dataset already "
        "prepared via prepare_finetune_dataset, using LoRA (or QLoRA with "
        "use_qlora=True, which requires a CUDA GPU). Saves the trained adapter "
        "under workspace/finetune/<run_id>/adapter/. This step requires real "
        "compute time and downloads the base model on first use — plan "
        "accordingly for larger models or datasets."
    ),
    schema={
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "A run_id with a dataset already prepared via prepare_finetune_dataset."},
            "base_model_id": {"type": "string", "description": "HuggingFace model ID of the local base model to fine-tune."},
            "use_qlora": {"type": "boolean", "description": "Use 4-bit QLoRA instead of plain LoRA. Requires a CUDA GPU. Default false."},
            "num_epochs": {"type": "integer", "description": "Training epochs (default 3)."},
            "learning_rate": {"type": "number", "description": "Learning rate (default 2e-4)."},
            "lora_r": {"type": "integer", "description": "LoRA rank (default 8) — higher means more trainable parameters."},
        },
        "required": ["run_id", "base_model_id"],
    },
)
def fine_tune(
    run_id: str, base_model_id: str, use_qlora: bool = False,
    num_epochs: int = 3, learning_rate: float = 2e-4, lora_r: int = 8,
) -> str:
    from agent.finetuning import get_fine_tuner
    return get_fine_tuner().fine_tune(
        run_id, base_model_id, use_qlora=use_qlora,
        num_epochs=num_epochs, learning_rate=learning_rate, lora_r=lora_r,
    )


@tool(
    description=(
        "Export a completed fine-tuning run's result. export_mode='merged' "
        "(default) folds the LoRA adapter into the base model and saves a "
        "standalone model directory anything can load normally. "
        "export_mode='adapter' just reports the adapter's location as-is "
        "(smaller on disk, but needs PEFT-aware loading code)."
    ),
    schema={
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "A run_id that has completed fine_tune."},
            "export_mode": {"type": "string", "enum": ["merged", "adapter"], "description": "Default 'merged'."},
        },
        "required": ["run_id"],
    },
)
def merge_and_export_model(run_id: str, export_mode: str = "merged") -> str:
    from agent.finetuning import get_fine_tuner
    return get_fine_tuner().merge_and_export(run_id, export_mode=export_mode)


@tool(
    description="List all fine-tuning runs in this session and their status.",
    schema={"type": "object", "properties": {}, "required": []},
)
def list_finetune_runs() -> str:
    from agent.finetuning import get_fine_tuner
    return get_fine_tuner().list_finetune_runs()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 15 TOOLS — evaluation, guardrails & observability
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Run a small canned benchmark suite testing whether the prompt- "
        "injection guardrail correctly flags known injection patterns AND "
        "correctly does NOT flag benign content (both matter — a guardrail "
        "that flags everything is as useless as one that catches nothing). "
        "Useful for verifying the guardrail layer itself is working, or "
        "after editing the pattern list in observability.py."
    ),
    schema={"type": "object", "properties": {}, "required": []},
)
def run_guardrail_benchmark() -> str:
    from agent.observability import get_benchmark_harness
    return get_benchmark_harness().run()


@tool(
    description=(
        "Show every prompt-injection pattern flagged so far in this session "
        "by the live guardrail policy (the one actually scanning real tool "
        "results during this run, not the benchmark's synthetic cases). "
        "Empty if nothing's been flagged."
    ),
    schema={"type": "object", "properties": {}, "required": []},
)
def get_guardrail_findings() -> str:
    from agent.observability import get_guardrail_policy
    return get_guardrail_policy().summary()


# ═══════════════════════════════════════════════════════════════════════════════
# V2 TOOLS — AIDE-style solution tree search (the MLE-bench engine)
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    description=(
        "Solve a complete ML task autonomously via solution tree search: the "
        "engine drafts several independent solution scripts, executes each in "
        "the sandbox, reviews the results, then iteratively debugs failures and "
        "improves the best solution until the step/time budget is spent. "
        "Returns the best validation metric, the path to the winning script, "
        "and the run report. Use this for any 'train the best possible model "
        "on this data' style task instead of hand-driving individual "
        "train_model calls."
    ),
    schema={
        "type": "object",
        "properties": {
            "task": {"type": "string",
                     "description": "Full natural-language task description including the target and metric."},
            "data_dir": {"type": "string",
                         "description": "Path to the directory holding the task's data files."},
            "steps": {"type": "integer",
                      "description": "Search budget in nodes (default 12)."},
            "time_limit_secs": {"type": "integer",
                                "description": "Optional wall-clock budget in seconds."},
        },
        "required": ["task", "data_dir"],
    },
)
def solve_ml_task(task: str, data_dir: str, steps: int = 12,
                  time_limit_secs: int | None = None) -> str:
    from agent.search import SearchConfig, run_search
    cfg = SearchConfig(steps=steps, time_limit_secs=time_limit_secs)
    try:
        result = run_search(task, data_dir=data_dir, config=cfg)
    except Exception as e:
        return f"Error: solution search failed: {e}"
    if result.best:
        return (
            f"Search complete: best validation metric {result.best.metric:.6g} "
            f"({'lower' if result.best.lower_is_better else 'higher'} is better) "
            f"after {result.steps_done} steps in {result.wall_time:.0f}s.\n"
            f"Best solution: {result.solution_path}\n"
            f"Report: {result.report_path}"
        )
    return (
        f"Search finished without a working solution after {result.steps_done} steps. "
        f"See report for failure analysis: {result.report_path}"
    )
